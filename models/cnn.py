# Based on: https://pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/
import numpy as np
import tensorflow as tf
from utils import euclidean_distance

def get_cnn_model(embed_size=48):
    cnn = tf.keras.Sequential([
        # First convolution layers
        tf.keras.layers.Conv2D(64, (2,2), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.2),
        # Second convolution layer
        tf.keras.layers.Conv2D(64, (2, 2), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        # Output embeddings
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(embed_size)
    ])
    return cnn

def get_siamese_model(embed_size=48):
    im_a = tf.keras.layers.Input(shape=(64,128,1))
    im_b = tf.keras.layers.Input(shape=(64,128,1))
    cnn = get_cnn_model(embed_size)
    feats_a = cnn(im_a)
    feats_b = cnn(im_b)
    distance = tf.keras.layers.Lambda(euclidean_distance)([feats_a, feats_b])
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
    model = tf.keras.models.Model(inputs=[im_a, im_b], outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  ## feel free to change
        loss="binary_crossentropy",  ## do not change loss/metrics
        metrics=["accuracy"],
    )
    
    return model