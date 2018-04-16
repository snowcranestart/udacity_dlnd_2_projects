
项目需要用到的软件以及库

项目需要的配置
aws p3.2xlarge

数据集：
Kaggle 猫狗大战数据集

所有代码放在根目录下
另外有如下文件夹
data, —存放训练集，测试集，压缩包和解压文件
predicts,预测结果
models, 模型参数和模型结构图片
merged_features,模型的特征向量

项目文件包括
helper.py 所有封装的方法，包括文件操作，图片预处理，生成特征向量，获取模型，可视化图片，可视化模型等等。

准备数据
1. 运行prepare_files_final.ipynb (大约2分钟）
将train.zip, test.zip解压，并按照求移动到对应文件夹， train1中包含dog,cat文件夹包含对应的猫狗训练集图片，train2与train1相同，作为将要移除异常值的文件夹备用，test1中包含test1,其中包含所有测试图片
2. 运行remove_outliers_final.ipynb (大约15分钟）
将train2中的猫狗异常值移除，作为第二训练集备用

3.在train1训练集上训练模型并预测测试集
- Fine-tuning 单个模型
fine_tuning_xception_with_outliers_final.ipynb （大约 80 分钟, 输出不小心刷新了，运行结果请参考对应的html, pre_.cv文件）
fine_tuning_inceptionv3_with_outliers_final.ipynb （大约 70 分钟）
fine_tuning_inceptionresnetv2_with_outliers_final.ipynb （大约 130 分钟）
- 特征向量融合
merged_3_models_with_outliers_final.ipynb （大约 25  分钟）

4.在train2训练集上训练模型并预测测试集
- Fine-tuning 单个模型
fine_tuning_xception_no_outliers_final.ipynb （大约 70 分钟）
fine_tuning_inceptionv3_no_outliers_final.ipynb （大约 60 分钟）
fine_tuning_inceptionresnetv2_no_outliers_final.ipynb （大约 120 分钟）
- 特征向量融合
merged_3_models_no_outliers_final.ipynb （大约 20 分钟）

对应的html文件
prepare_files_final.html
remove_outliers_final.ipynb
fine_tuning_xception_with_outliers_final.html
fine_tuning_inceptionv3_with_outliers_final.html
fine_tuning_inceptionresnetv2_with_outliers_final.html
merged_3_models_with_outliers_final.html
fine_tuning_xception_no_outliers_final.html
fine_tuning_inceptionv3_no_outliers_final.html
fine_tuning_inceptionresnetv2_no_outliers_final.html
merged_3_models_no_outliers_final.html

预测结果集
pre_fine_tuning_xception_with_outliers_final.csv
pre_fine_tuning_inceptionv3_with_outliers_final.csv
pre_fine_tuning_inceptionresnetv2_with_outliers_final.csv
pre_merged_3_models_with_outliers_final.csv
pre_fine_tuning_xception_no_outliers_final.csv
pre_fine_tuning_inceptionv3_no_outliers_final.csv
pre_fine_tuning_inceptionresnetv2_no_outliers_final.csv
pre_merged_3_models_no_outliers_final.csv




