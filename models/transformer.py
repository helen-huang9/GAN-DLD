# Based on: https://keras.io/examples/vision/image_classification_with_vision_transformer/
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import euclidean_distance

image_size = (64,128)  # We'll resize input images to this size
patch_size = 4  # Size of the patches to be extract from the input images
num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
projection_dim = 64
num_heads = 3

class Patches(keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded 

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, num_heads):
        super(EncoderBlock, self).__init__()
        self.ff_layer = tf.keras.layers.Dense(emb_sz)
        self.self_atten = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=emb_sz)
        self.layer_norm = tf.keras.layers.LayerNormalization() # Could make 2 different layer norms
    def call(self, inputs):
        attention = self.self_atten(inputs,inputs)
        attention += inputs
        attention = self.layer_norm(attention)
        out = self.ff_layer(attention)
        out += attention
        out = self.layer_norm(out)
        return out

def get_encoder_model(embed_size=48):
    encoder = tf.keras.Sequential([
        # Rescale input
        tf.keras.layers.Rescaling(scale=1/255),
        # Encoding
        Patches(patch_size),
        PatchEncoder(num_patches, projection_dim),
        EncoderBlock(emb_sz=projection_dim, num_heads=num_heads),
        tf.keras.layers.Dense(embed_size)
    ])
    return encoder

def get_transformer_model(embed_size=48):
    # Inputs
    im_a = tf.keras.layers.Input(shape=(64,128,1))
    im_b = tf.keras.layers.Input(shape=(64,128,1))
    # Encoder
    encoder = get_encoder_model(embed_size)
    feats_a = encoder(im_a)
    feats_b = encoder(im_b)
    # Difference module
    distance = tf.keras.layers.Lambda(euclidean_distance)([feats_a, feats_b])
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
    model = tf.keras.models.Model(inputs=[im_a, im_b], outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  ## feel free to change
        loss="binary_crossentropy",  ## do not change loss/metrics
        metrics=["binary_accuracy"],
    )
    return model              
