
# coding: utf-8

# In[1]:


import helper


# In[2]:


data_path = './data' 
train_save_path = './data/train.zip'
test_save_path =  './data/test.zip'
train_extracted_path = './data/train'
test_extracted_path = './data/test'
train_final_path1 = './data/train1'
train_final_path2 = './data/train2'
test_final_path1 ='./data/test1/test1'
classes=['dog', 'cat']


# In[3]:


#将训练集文件移到两个子目录下, 作为不去除异常值的训练集数据
helper.extract_file(train_save_path, data_path, train_extracted_path)
helper.move_files_into_sub_classes(train_extracted_path, train_final_path1, classes)


# In[4]:


#将测试数据集文件移动目标文件夹中
helper.extract_file(test_save_path, data_path, test_extracted_path)
helper.move_files(test_extracted_path, test_final_path1)


# In[5]:


#将训练集文件移到两个子目录下，作为将去除异常值的训练集数据
helper.extract_file(train_save_path, data_path, train_extracted_path)
helper.move_files_into_sub_classes(train_extracted_path, train_final_path2, classes)

