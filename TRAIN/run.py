# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:12:25 2018

@author: wwt
"""
from help import loss, accuracy, data_generator
from models import model_build
import pandas as pd
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

####↓↓↓↓↓↓↓↓↓↓↓↓↓↓参数设置↓↓↓↓↓↓↓↓↓↓↓######
pic_size = 448 #网络设置的输入图片大小
class_num = 54 #最后划分的类别数
batch_size = 8 #每次训练的batch大小
epoch = 2 #训练轮数
dropout = 0.5 #全连接后的dropout比例
learn_rate = 0.01 #学习率初始值
momentum = 0.9 #动量优化
factors=0.2 #学习速率被降低的因数
patience=3 #没有进步的训练轮数
learn_rate=0.00001 #最小学习率
train_data_hance = True #训练时是否做数据增强
####↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑参数设置↑↑↑↑↑↑↑↑↑↑↑↑######

####↓↓↓↓↓↓↓↓↓↓↓↓↓↓模型路径↓↓↓↓↓↓↓↓↓↓↓######
model_name = r"my_model.h5"
data_divide_into_train_and_test = r"train_and_test_lable.csv" #划分训练集测试集后的保存
####↑↑↑↑↑↑↑↑↑↑↑↑↑↑模型路径↑↑↑↑↑↑↑↑↑↑↑↑######

##↓↓↓↓↓↓↓↓↓↓↓↓↓↓训练数据与测试数据↓↓↓↓↓↓↓↓↓↓↓######
train_and_test_lable = pd.read_csv(data_divide_into_train_and_test)#训练集的图像标签

#生成训练数据
train_data_image_path = list(train_and_test_lable[train_and_test_lable["flag"]==1]["imagepath"].reset_index(drop=True))
train_data_lable = [eval(x) for x in list(train_and_test_lable[train_and_test_lable["flag"]==1]["label"].reset_index(drop=True))]

#生成测试数据
test_data_image_path = list(train_and_test_lable[train_and_test_lable["flag"]==0]["imagepath"].reset_index(drop=True))
test_data_lable = [eval(x) for x in list(train_and_test_lable[train_and_test_lable["flag"]==0]["label"].reset_index(drop=True))]


#生成train_datagen
train_datagen = data_generator( data_image_path=train_data_image_path,
                                data_lable=train_data_lable,
                                batch_size=batch_size,
                                target_size=(pic_size, pic_size),
                                ifenhance=train_data_hance)

#生成validation_datagen
validation_datagen = data_generator( data_image_path=test_data_image_path, 
                                     data_lable=test_data_lable, 
                                     batch_size=batch_size, 
                                     target_size=(pic_size, pic_size), 
                                     ifenhance=False)
##↑↑↑↑↑↑↑↑↑↑↑↑↑↑训练数据与测试数据↑↑↑↑↑↑↑↑↑↑↑↑######


##↓↓↓↓↓↓↓↓↓↓↓↓↓↓生成模型↓↓↓↓↓↓↓↓↓↓↓######
model = model_build(input_shape=(pic_size, pic_size, 3), classes=class_num, dropout=dropout)
optimizer = SGD(lr=learn_rate, momentum=momentum)
model.compile(loss=loss, optimizer=optimizer, metrics=[accuracy])
##↑↑↑↑↑↑↑↑↑↑↑↑↑↑生成模型↑↑↑↑↑↑↑↑↑↑↑↑######

##↓↓↓↓↓↓↓↓↓↓↓↓↓↓生成call_back参数↓↓↓↓↓↓↓↓↓↓↓######
check_point = ModelCheckpoint(model_name, save_best_only=True,save_weights_only=True, verbose=1)
reduce_learn_rate_value = ReduceLROnPlateau(monitor='val_loss', factor=factors, patience=patience, min_lr=learn_rate)
when_stop = EarlyStopping(monitor='val_loss', patience=2 * patience)
call_back=[check_point, reduce_learn_rate_value, when_stop]
##↑↑↑↑↑↑↑↑↑↑↑↑↑↑生成call_back参数↑↑↑↑↑↑↑↑↑↑↑↑######

##↓↓↓↓↓↓↓↓↓↓↓↓↓↓训练模型↓↓↓↓↓↓↓↓↓↓↓######
model.fit_generator(generator=train_datagen,
                    steps_per_epoch= len(train_data_image_path) // batch_size,
                    epochs=epoch,
                    callbacks=call_back,
                    validation_data=validation_datagen,
                    validation_steps = len(test_data_image_path) // batch_size,
                    verbose=1)
##↑↑↑↑↑↑↑↑↑↑↑↑↑↑训练模型↑↑↑↑↑↑↑↑↑↑↑↑######

##↓↓↓↓↓↓↓↓↓↓↓↓↓↓保存模型↓↓↓↓↓↓↓↓↓↓↓######
model.save_weights(model_name) 
print("train_over")
##↑↑↑↑↑↑↑↑↑↑↑↑↑↑保存模型↑↑↑↑↑↑↑↑↑↑↑######
