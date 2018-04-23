# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 23:59:19 2018

@author: wwt
"""
import random
from math import floor
from PIL import Image
import keras.backend as K
import numpy as np
import tensorflow as tf


###↓↓↓↓↓↓↓↓↓↓↓↓↓↓返回位置↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓####
def label_dict(label):
    if label=="skirt_length_labels":
        return 0,6
    if label=="coat_length_labels":
        return 6,14
    if label=="collar_design_labels":
        return 14,19
    if label=="lapel_design_labels":
        return 19,24
    if label=="neck_design_labels":
        return 24,29
    if label=="neckline_design_labels":
        return 29,39
    if label=="pant_length_labels":
        return 39,45
    if label=="sleeve_length_labels":
        return 45,54
    
###↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑返回位置↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑####


def image_reforce(image,prop,percentage,version):
    value = random.random() < prop
    if version==0:
        if value:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return image
    else:
        if value:
            wight,hight = image.size
            update_wight_left=(wight/2)-(int(floor(wight * percentage))/2)
            update_hight_left=(hight/2)-(int(floor(hight * percentage))/2)
            update_wight_right=(wight/2)+(int(floor(wight * percentage))/2)
            update_hight_right=(hight/2)+(int(floor(hight * percentage))/2)
            return image.crop((update_wight_left, 
                               update_hight_left, 
                               update_wight_right, 
                               update_hight_right))
        else:
            return image

def data_generator(data_image_path, data_lable, batch_size, target_size, reforce=True):
    while True:
        Input = []
        output = []
        for _ in range(batch_size):
            random_image_index= random.randint(0, len(data_image_path)-1)
            image = Image.open(data_image_path[random_image_index])
            if reforce:
                image = image_reforce(image,prop=0.5,percentage=0.98,version=0)
                image = image_reforce(image,prop=0.2,percentage=0.98,version=1)
            image = image.resize(target_size, Image.ANTIALIAS)
            image = np.array(image)
            input.append(image)
            output.append(data_lable[random_image_index])
        Input = np.asarray(Input)
        output = np.asarray(output)
        Input = input.astype('float32')
        Input /= 255
        yield (Input, output)

####↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓用于模型建立↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓#########

####↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑用于模型建立↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑#########


###↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓损失函数↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓#####
def loss(y_true, y_pred):
    value = tf.cast(y_true > -1, dtype=tf.float32)
    return K.binary_crossentropy(y_true * value, y_pred * value)
###↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑损失函数↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑#####

####↓↓↓↓↓↓↓↓↓↓↓↓↓准确度函数↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓####
def accuracy(y_true, y_pred):
    value = tf.cast(y_true > -1, dtype=tf.float32)
    return tf.cast(tf.equal(tf.argmax(y_true * value, axis=-1),
                          tf.argmax(y_pred * value, axis=-1)),
                           tf.float32)
###↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑准确度函数↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑#####   




