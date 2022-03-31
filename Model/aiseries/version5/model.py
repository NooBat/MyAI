import tensorflow as tf
from tensorflow import keras
from keras import regularizers
import os

def create_model(path_to_weights='', load_weights=True):
    """Function to create a model

    Returns a compiled and optionally loaded model

    Keyword arguments:

    path_to_weights -- (Optional, only used when load_weights is True) -- Path to weight file (.hdf5 files)

    load_weights -- Whether to load weights or not (default to True)
    """
    if (load_weights):
        assert(path_to_weights is not None and 
           os.path.isfile(path_to_weights)), "path_to_weights must exist and not be empty if load_weights is True, otherwise change load_weights to False"

    model = keras.models.Sequential([
        Input((32,32,3)),
        Dropout(0.2),
        Conv2D(96, (3,3), padding='same',
                            kernel_regularizer=l2(1e-3),
#                             activity_regularizer=l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        BatchNormalization(),
        Conv2D(96, (3,3), padding='same',
                            kernel_regularizer=l2(1e-3),
#                             activity_regularizer=l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        BatchNormalization(),
        Conv2D(96, (3,3), padding='same', strides=(2,2),
                            kernel_regularizer=l2(1e-3),
#                             activity_regularizer=l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Conv2D(192, (3,3), padding='same',
                            kernel_regularizer=l2(1e-3),
#                             activity_regularizer=l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        BatchNormalization(),
        Conv2D(192, (3,3), padding='same',
                            kernel_regularizer=l2(1e-3),
#                             activity_regularizer=l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        BatchNormalization(),
        Conv2D(192, (3,3), padding='same', strides=(2,2),
                            kernel_regularizer=l2(1e-3),
#                             activity_regularizer=l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Conv2D(192, (3,3), padding='same',
                            kernel_regularizer=l2(1e-3),
#                             activity_regularizer=l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        BatchNormalization(),
        Conv2D(192, (1,1), padding='same',
                            kernel_regularizer=l2(1e-3),
#                             activity_regularizer=l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        BatchNormalization(),
        Conv2D(10, (1,1), padding='same',
                            kernel_regularizer=l2(1e-3),
#                             activity_regularizer=l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    if load_weights:
        model.load_weights(path_to_weights)

    model.compile(optimizer=Adam(learning_rate=0.001),
                                 loss='sparse_categorical_crossentropy',
                                 metrics=[
                                 SparseCategoricalCrossentropy(name='sparse'),
                                 'accuracy'])

    return model
