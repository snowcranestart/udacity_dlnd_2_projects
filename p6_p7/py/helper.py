
# coding: utf-8

# In[32]:


from keras.applications import *
from keras.layers import *
from keras.preprocessing.image import *
from keras.preprocessing import image
from keras.models import *

import math
import os
import shutil
import zipfile
import gzip
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


# In[33]:


#文件操作

def move_files(source_path, target_path):
    if not os.path.isdir(source_path):
        print("source path is not exist")
        return
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    files = os.listdir(source_path)
    for file in files:
        shutil.move(os.path.join(source_path,file), os.path.join(target_path,file))
    shutil.rmtree(source_path)
    print("finished moving")

               
def extract_file(zip_file, data_path, extracted_path):
    
    if os.path.isdir(extracted_path):
        print("Extracted files already exist.")
        return 
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    try:
        print('Extracting {}...'.format(zip_file))
        with zipfile.ZipFile(zip_file) as zf:
            zf.extractall(data_path)
    except Exception as err:
        shutil.rmtree(data_path)  # Remove extraction folder if there is an error
        raise err
    print("Finished extraction")    
    
def move_files_into_sub_classes(extracted_path, new_path, classes):      
    if not os.path.isdir(extracted_path):
        print("The extracted folder is not exist.")
        return
    
    image_files = os.listdir(extracted_path)
    if len(image_files)==0:
        print("There is no file in the folder")
        return
    
    if not os.path.isdir(new_path):
        os.makedirs(new_path)
        
    sub_dirs = [os.path.join(new_path,each) for each in classes]
    
    for path in sub_dirs:
        if not os.path.isdir(path):
            os.makedirs(path)
    
    print("Moving images to sub folders.")
    for image_file in image_files:
        if classes[0] in image_file:
            shutil.move(os.path.join(extracted_path,image_file), os.path.join(sub_dirs[0],image_file))
        else:
            shutil.move(os.path.join(extracted_path,image_file), os.path.join(sub_dirs[1],image_file))
    shutil.rmtree(extracted_path)
    print("Finish moving files to sub folders")
    
def move_validation_files_from_trainset(train_path, validation_path, classes, split_index): 
    train_sub_paths = [os.path.join(train_path, class_item) for class_item in classes]
    validation_sub_paths = [os.path.join(validation_path, class_item) for class_item in classes]        
    
    if os.path.isdir(validation_sub_paths[0]) and os.path.isdir(validation_sub_paths[1]):
        print("Already moved")
        return 
    
    for path in validation_sub_paths:
        os.makedirs(path)
        
    for ii, image_file in enumerate(os.listdir(train_sub_paths[0]), 1):
        if ii >split_index:
            shutil.move(os.path.join(train_sub_paths[0],os.path.basename(image_file)),
                        os.path.join(validation_sub_paths[0],os.path.basename(image_file)))
    for ii, image_file in enumerate(os.listdir(train_sub_paths[1]), 1):
        if ii >split_index:
            shutil.move(os.path.join(train_sub_paths[1],os.path.basename(image_file)),
                        os.path.join(validation_sub_paths[1],os.path.basename(image_file)))
    print("Moving finished.")


# In[34]:


def get_images_and_shapes(file_path):
    image_files = [os.path.join(file_path, file) for file in os.listdir(file_path)]
    images_shapes= []
    for ii, path in enumerate(image_files):
        img = image.load_img(path)
        shape = np.shape(img)
        images_shapes.append(shape)
    print("finished")
    return image_files, images_shapes


#获取单个文件夹（只包含图片文件）中所有图片，用lambda_func预处理，用于异常值处理时读取图片
def get_input_from_folder_with_image_files(file_path, img_size, lambda_func=None):
    files = [os.path.join(file_path, file) for file in os.listdir(file_path)]
    X = np.empty((len(files), img_size[0], img_size[1], 3), dtype=np.float32)
    for ii, path in enumerate(files):
        img = image.load_img(path, target_size=img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = lambda_func(x)
        X[ii]=x
    print("finished")
    return X, files

def get_basic_classifier_model(MODEL):
    model = MODEL(weights="imagenet")
    return model

def get_outliers_for_one_class(model, X_inputs, files, topN, nameList, lambda_decode):
    preds = model.predict(X_inputs)
    print("finished predicting")
    
    decoded_preds = lambda_decode(preds, top=topN)
    outliers = []
    
    for ii, predict in enumerate(decoded_preds, 0):
        flag = False
        for pre in enumerate(predict, 0):
            if pre[1][0] in nameList:
                flag=True
                break
        if flag == False:
            outliers.append(files[ii])
    print(len(outliers))
    print(outliers[:10])
    return outliers

#可视化异常值等
def visual_images(files, image_size):
    import matplotlib.pyplot as plt
    import PIL
    get_ipython().magic('matplotlib inline')
    
    images =[image.load_img(path, target_size =image_size) for path in files]
    images = [image.img_to_array(img) for img in images]
    images = np.array(images).astype(np.uint8)
    
    batches = math.ceil(len(images)*1.0/8)
    
    print(batches)
    images = images[0:batches*8]
    fig = plt.figure(figsize=(20,20))
    for i in range(1, 8*batches+1 ):
        if i-1 < len(files):
            fig.add_subplot(batches, 8, i)
            plt.xlabel(os.path.basename(files[i-1]))
            plt.imshow(images[i-1])
    plt.show()

#将单个class里的outliers移到另一个文件夹
def move_files_to_new_folder(files, target_path):
    if len(files) == 0:
        print("0 files, no moving")
        return 
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    for file in files:
        shutil.move(file, os.path.join(target_path,os.path.basename(file)))
    print("finished moving")


# In[35]:


#获取单个文件夹中（只包含子文件夹）中的所有图片，用lambda_func预处理，用作获取训练模型输入

def get_train_input_from_folder_with_subclasses(train_path, img_size, lambda_func=None):
    image_files = [[os.path.join(path, file) for file in os.listdir(path)] for path in train_path]
    cat_num = len(image_files[0])
    dog_num = len(image_files[1])
    
    image_files = np.concatenate(image_files, axis = 0)
    
    X_train = np.empty((cat_num+dog_num, img_size[0],img_size[1],3), dtype=np.float32)
    y_train = np.concatenate((np.zeros((cat_num,1),dtype=np.float32),np.ones((dog_num,1),dtype=np.float32)), axis = 0)
    
    image_files, y_train = shuffle(image_files, y_train)
    
    for ii, path in enumerate(image_files):
        img = image.load_img(path,target_size =img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = lambda_func(x)
        X_train[ii]=x
    
    print(X_train.shape)
    print(y_train.shape)
    print("finished")
    return X_train, y_train, image_files



def predict_and_update_to_csv(model, X_test, image_file_names, template_csv_path, target_csv_path):
    y_pred = model.predict(X_test, verbose=1)
    y_pred = y_pred.clip(min=0.005, max=0.995)    
        
    df = pd.read_csv(template_csv_path)
    
    for i, fname in enumerate(image_file_names):
        index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
        df.at[index-1,'label'] = y_pred[i]
        
    df.to_csv(target_csv_path, index=None)
    print("finished")


# In[36]:


def write_gap(MODEL,image_size, train_path, test_path, save_path, lambda_func=None):
    input_tensor = Input((image_size[0], image_size[1], 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory(train_path, image_size, shuffle=False, 
                                              batch_size=16)
    test_generator = gen.flow_from_directory(test_path, image_size, shuffle=False, 
                                             batch_size=16, class_mode=None)

    print("start to predict")
    train = model.predict_generator(train_generator)
    test = model.predict_generator(test_generator)
    print("finish to predict")
    with h5py.File(save_path) as h:
        h.create_dataset("X_train", data=train)
        h.create_dataset("X_test", data=test)
        h.create_dataset("y_train", data=train_generator.classes)
    print("finished")

def load_and_merge_features(feature_files):
    import h5py
    import numpy as np
    from sklearn.utils import shuffle
    np.random.seed(2017)
    
    X_train = []
    X_test = []
    
    for filename in feature_files:
        with h5py.File(filename, 'r') as h:
            X_train.append(np.array(h['X_train']))
            X_test.append(np.array(h['X_test']))
            y_train = np.array(h['y_train'])

    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)

    X_train, y_train = shuffle(X_train, y_train)
    print('finished')
    return X_train, y_train, X_test

def get_model_for_merge_features(X_input):    
    input_tensor = Input(X_input.shape[1:])
    x = Dropout(0.5)(input_tensor)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, x)

    model.compile(optimizer='adadelta',loss='binary_crossentropy',metrics=['accuracy'])
    return model


# In[37]:


def get_fine_tuning_first_model(MODEL):
    from keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from keras.models import Model
    
    print("start")
    base_model = MODEL( weights='imagenet', include_top=False)
    print(base_model.input.shape)
    print(base_model.output.shape)

    top_x = base_model.output
    top_x = GlobalAveragePooling2D()(top_x)
    top_x = Dropout(0.5)(top_x)
    top_x = Dense(1, activation='sigmoid')(top_x)
    model = Model(base_model.input, top_x)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_fine_tuning_second_model(model, layer_num):
    for layer in model.layers[:layer_num]:
        layer.trainable = False
    for layer in model.layers[layer_num:]:
        layer.trainable = True

    from keras.optimizers import SGD

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy',metrics=['accuracy'])
    return model

def visualize_model(model, model_image):
    from keras.utils.vis_utils import plot_model
    from IPython.display import Image
    
    plot_model(model, to_file=model_image, show_shapes=True)
    Image(model_image)

