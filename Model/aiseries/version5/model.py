import tensorflow as tf
from tensorflow import keras
from keras import regularizers
import os

def create_model(path_to_weights=None, load_weights=True):
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
        keras.layers.Input((32,32,3)),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(96, (3,3), padding='same',
                            kernel_regularizer=regularizers.l2(1e-3),
#                             activity_regularizer=regularizers.l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(96, (3,3), padding='same',
                            kernel_regularizer=regularizers.l2(1e-3),
#                             activity_regularizer=regularizers.l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(96, (3,3), strides=(2,2), padding='same',
                            kernel_regularizer=regularizers.l2(1e-3),
#                             activity_regularizer=regularizers.l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.SpatialDropout2D(0.5),

        keras.layers.Conv2D(192, (3,3), padding='same',
                            kernel_regularizer=regularizers.l2(1e-3),
#                             activity_regularizer=regularizers.l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(192, (3,3), padding='same',
                            kernel_regularizer=regularizers.l2(1e-3),
#                             activity_regularizer=regularizers.l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(192, (3,3), strides=(2,2), padding='same',
                            kernel_regularizer=regularizers.l2(1e-3),
#                             activity_regularizer=regularizers.l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.SpatialDropout2D(0.5),
        
        keras.layers.Conv2D(192, (3,3),
                            kernel_regularizer=regularizers.l2(1e-3),
#                             activity_regularizer=regularizers.l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(192, (1,1), padding='same',
                            kernel_regularizer=regularizers.l2(1e-3),
#                             activity_regularizer=regularizers.l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(10, (1,1),
                            kernel_regularizer=regularizers.l2(1e-3),
#                             activity_regularizer=regularizers.l2(1e-3),
                            kernel_initializer='he_normal',
                            activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    if load_weights:
        model.load_weights(path_to_weights)

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                                                loss='sparse_categorical_crossentropy',
                                                metrics=[
                                                    tf.keras.losses.SparseCategoricalCrossentropy(name='sparse_categorical_crossentropy'),
                                                    'accuracy'])

    return model
