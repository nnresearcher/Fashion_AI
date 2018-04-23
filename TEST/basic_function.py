# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 23:59:19 2018

@author: wwt
"""
from PIL import Image
import numpy as np


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

####↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓用于模型预测↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓#########
def predict(image_path, model, resize_shape):
    image = Image.open(image_path)
    image = image.resize(resize_shape, Image.ANTIALIAS)
    image = np.array(image)
    image = image.astype('float32')
    image /= 255
    image = np.expand_dims(image, axis=0)
    predict_output = model.predict(image)
    return predict_output
####↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑用于模型预测↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑#########



