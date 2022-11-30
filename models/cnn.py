import numpy as np
import tensorflow as tf

def get_CNN_model():
    model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(scale=1/255),
            tf.keras.layers.Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'), 
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            # tf.keras.layers.MaxPool2D(pool_size=(2,2)),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(2),
            tf.keras.layers.Activation('softmax')
        ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),  ## feel free to change
        loss="binary_crossentropy",  ## do not change loss/metrics
        metrics=["binary_accuracy"],
    )

    return model