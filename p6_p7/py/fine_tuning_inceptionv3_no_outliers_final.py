
# coding: utf-8

# In[1]:


import helper
import matplotlib.pyplot as plt
from keras.applications import *
from keras.callbacks import EarlyStopping
import os


# In[2]:


#设置各种参数
train_path = ['./data/train2/cat', './data/train2/dog']
test_path ='./data/test1/test1'
img_size =(299,299)
layer_num = 249
model_image ='./models/model_image_fine_tuning_inceptionv3_0403.png'
model_weights_file = './models/weights_fine_tuning_inceptionv3_no_outliers_0403.h5'
template_csv_path = './predicts/sample_submission.csv'
target_csv_path = './predicts/pred_fine_tuning_inceptionv3_no_outliers_0403.csv'
MODEL = inception_v3.InceptionV3
preprocess_func = inception_v3.preprocess_input


# In[3]:


#获取训练集数据
X_train, y_train, image_files= helper.get_train_input_from_folder_with_subclasses(train_path, img_size, lambda_func=preprocess_func)
print("finished")


# In[4]:


#构造模型，锁定base_model所有层
model = helper.get_fine_tuning_first_model(MODEL)

#可视化模型
helper.visualize_model(model, model_image)
print("finished")


# In[5]:


print("start")
#第一次训练新添加层权重
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(X_train, y_train, batch_size=128, epochs=8, validation_split=0.2, callbacks=[early_stopping])
print("finished")


# In[6]:


print("start")
#放开若干层权重，再次训练
model = helper.get_fine_tuning_second_model(model, layer_num)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
model.fit(X_train, y_train, batch_size=128, epochs=60, validation_split=0.2, callbacks=[early_stopping])
print("finished")


# In[7]:


#保存模型参数
model.save_weights(model_weights_file)
del X_train
del y_train
print("finished")


# In[8]:


print("start")
#获取测试数据和对应文件
X_test, test_files = helper.get_input_from_folder_with_image_files(test_path, img_size, lambda_func=preprocess_func)

#获取文件basename
image_file_names = [os.path.basename(path) for path in test_files]

#预测并保存预测结果到csv
helper.predict_and_update_to_csv(model, X_test, image_file_names, template_csv_path, target_csv_path)

print("finished")

