import tensorflow as tf

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((28,12,1)),
        tf.keras.layers.Conv2D(16, (3,3), activation='elu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-3), 
                            padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(16, (3,3), activation='elu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-3), 
                            padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3,3), activation='elu', 
                            kernel_regularizer=tf.keras.regularizers.l2(1e-3), 
                            padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.load_weights('version9.hdf5')

    return 0;