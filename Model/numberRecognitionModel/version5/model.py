import os
import tensorflow as tf
from keras.layers import *
from keras.regularizers import l2

def create_model(path_to_weights='', load_weights=True) :
    if (load_weights):
        assert(path_to_weights is not None and 
           os.path.isfile(path_to_weights)), "path_to_weights must exist and not be empty if load_weights is True, otherwise change load_weights to False"
        
    model = tf.keras.models.Sequential([
        Input((28,12,1)),
        Dropout(0.2),
        
        Conv2D(8, (3,3), activation='relu',
            kernel_regularizer=l2(1e-3), 
            padding='same'),
        BatchNormalization(),
        Conv2D(8, (3,3), activation='relu', strides=(2,2),
            kernel_regularizer=l2(1e-3),
            padding='same'),
        BatchNormalization(),
        SpatialDropout2D(0.4),
        
        Conv2D(16, (3,3), activation='relu', 
            kernel_regularizer=l2(1e-3), 
            padding='same'),
        BatchNormalization(),
        Conv2D(16, (3,3), activation='relu', strides=(2,2),
            kernel_regularizer=l2(1e-3), 
            padding='same'),
        BatchNormalization(),
        SpatialDropout2D(0.4),
        
        
        Conv2D(32, (3,3), activation='relu', 
            kernel_regularizer=l2(1e-3), 
            padding='same'),
        BatchNormalization(),
        Conv2D(32, (1,1), activation='relu', 
            kernel_regularizer=l2(1e-3), 
            padding='same'),
        BatchNormalization(),
        Conv2D(10, (1,1), activation='relu', 
        kernel_regularizer=l2(1e-3), 
        padding='same'),
        BatchNormalization(),
        Flatten(),
        
        Dense(10, activation='softmax')
    ])
    
    if load_weights:
        model.load_weights(path_to_weights)

    return model