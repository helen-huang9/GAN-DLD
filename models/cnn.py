import numpy as np
import tensorflow as tf
from utils import euclidean_distance

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

    # TODO: 
    # - Look into triplet loss
    # - Add data augmentation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),  ## feel free to change
        loss="binary_crossentropy",  ## do not change loss/metrics
        metrics=["binary_accuracy"],
    )

    return model


def make_siamese_model(embed_size=48):
    inputs = tf.keras.layers.Input((64,128,1))
    x = tf.keras.layers.Conv2D(64, (2,2), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)


    x = tf.keras.layers.Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    pooled_out = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(embed_size)(pooled_out)

    model = tf.keras.models.Model(inputs, outputs)
    return model

def get_siamese_model(embed_size=48):
    im_a = tf.keras.layers.Input(shape=(64,128,1))
    im_b = tf.keras.layers.Input(shape=(64,128,1))
    feature_extractor = make_siamese_model(embed_size)
    feats_a = feature_extractor(im_a)
    feats_b = feature_extractor(im_b)
    distance = tf.keras.layers.Lambda(euclidean_distance)([feats_a, feats_b])
    outputs = tf.keras.layers.Dense(1, activation='softmax')(distance)
    model = tf.keras.models.Model(inputs=[im_a, im_b], outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  ## feel free to change
        loss="binary_crossentropy",  ## do not change loss/metrics
        metrics=["accuracy"],
    )
    
    return model