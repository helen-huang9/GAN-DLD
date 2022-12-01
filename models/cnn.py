import numpy as np
import tensorflow as tf
from utils import euclidean_distance

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
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
    model = tf.keras.models.Model(inputs=[im_a, im_b], outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  ## feel free to change
        loss="binary_crossentropy",  ## do not change loss/metrics
        metrics=["accuracy"],
    )
    
    return model