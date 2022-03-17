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
        assert(len(path_to_weights) > 0 and 
           os.path.isfile(path_to_weights)), "path_to_weights must exist and not be empty if load_weights is True, otherwise change load_weights to False"

    model = keras.models.Sequential([
        keras.layers.Input((32,32,1)),
        keras.layers.Conv2D(32, (3,3), padding='same',
                            kernel_regularizer=regularizers.l2(1e-3),
                            activity_regularizer=regularizers.l2(1e-4),
                            kernel_initializer='he_normal',
                            activation='elu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3,3), padding='same',
                            kernel_regularizer=regularizers.l2(1e-3),
                            activity_regularizer=regularizers.l2(1e-4),
                            kernel_initializer='he_normal',
                            activation='elu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2,2)),

        keras.layers.Conv2D(64, (3,3), padding='same',
                            kernel_regularizer=regularizers.l2(1e-3),
                            activity_regularizer=regularizers.l2(1e-4),
                            kernel_initializer='he_normal',
                            activation='elu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3,3), padding='same',
                            kernel_regularizer=regularizers.l2(1e-3),
                            activity_regularizer=regularizers.l2(1e-4),
                            kernel_initializer='he_normal',
                            activation='elu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2,2)),

        keras.layers.Conv2D(128, (1,1), padding='same',
                            kernel_regularizer=regularizers.l2(1e-3),
                            activity_regularizer=regularizers.l2(1e-4),
                            kernel_initializer='he_normal',
                            activation='elu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (1,1), padding='same',
                            kernel_regularizer=regularizers.l2(1e-3),
                            activity_regularizer=regularizers.l2(1e-4),
                            kernel_initializer='he_normal',
                            activation='elu'),
        keras.layers.BatchNormalization(),
        keras.layers.AveragePooling2D(pool_size=(2,2)),

        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='elu',
                        kernel_regularizer=regularizers.l2(1e-3),
                        activity_regularizer=regularizers.l2(1e-4)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='elu',
                        kernel_regularizer=regularizers.l2(1e-3),
                        activity_regularizer=regularizers.l2(1e-4)),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                                                loss='sparse_categorical_crossentropy',
                                                metrics=['accuracy'])

    if load_weights:
        model.load_weights(path_to_weights)

    return model
