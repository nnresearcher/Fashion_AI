'''
Created on 2018Äê4ÔÂ23ÈÕ

@author: hasee
'''
from densenet_updated_GN import DenseNet121
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model

def model_build(input_shape, classes, dropout):
    DenseNet121model = DenseNet121(include_top=False,
                                   weights="imagenet",
                                   input_tensor=None,
                                   input_shape=input_shape,
                                   pooling=None)
    x = DenseNet121model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(1024, activation='relu')(x)
    predicts = Dense(classes, activation='sigmoid')(x)
    model = Model(inputs=DenseNet121model.input, outputs=predicts)
    return model