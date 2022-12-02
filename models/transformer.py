# Based on: https://keras.io/examples/vision/image_classification_with_vision_transformer/
import numpy as np
import tensorflow as tf
from tensorflow import keras

num_classes = 2
input_shape = (64,128,3)

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = (32,64)  # We'll resize input images to this size
patch_size = 4  # Size of the patches to be extract from the input images
num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 3
mlp_head_units = [512, 2]  # Size of the dense layers of the final classifier


data_augmentation = keras.Sequential(
    [
        tf.keras.layers.Normalization(),
        tf.keras.layers.Resizing(*image_size),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(factor=0.02),
        tf.keras.layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

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

def get_transformer_model():
    im_A = keras.layers.Input(shape=input_shape)
    im_B = keras.layers.Input(shape=input_shape)
    # Augment data.
    aug_A = data_augmentation(im_A)
    aug_B = data_augmentation(im_B)
    # Create patches.
    patch_A = Patches(patch_size)(aug_A)
    patch_B = Patches(patch_size)(aug_B)
    # Encode patches.
    encoded_patch_A = PatchEncoder(num_patches, projection_dim)(patch_A)
    encoded_patch_B = PatchEncoder(num_patches, projection_dim)(patch_B)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1_A = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patch_A)
        x1_B = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patch_B)
        # Create a multi-head attention layer.
        attention_output_A = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1_A, x1_A)
        attention_output_B = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1_B, x1_B)
        # Residual connection 1.
        x2_A = tf.keras.layers.Add()([attention_output_A, encoded_patch_A])
        x2_B = tf.keras.layers.Add()([attention_output_B, encoded_patch_B])
        # Layer normalization 2.
        x3_A = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2_A)
        x3_B = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2_B)
        # MLP.
        x3_A = mlp(x3_A, hidden_units=transformer_units, dropout_rate=0.1)
        x3_B = mlp(x3_B, hidden_units=transformer_units, dropout_rate=0.1)
        # Residual connection 2.
        x3_A = tf.keras.layers.Add()([x3_A, x2_A])
        x3_B = tf.keras.layers.Add()([x3_B, x2_B])
        # Layer normalization 2.
        x3_A = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x3_A)
        x3_B = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x3_B)

    cross_attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
    )(x1_A, x1_B)
    # Create a [batch_size, projection_dim] tensor.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(cross_attention_output)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = tf.keras.layers.Dense(1, activation='sigmoid')(features)
    # Create the Keras model.
    model = tf.keras.Model(inputs=[im_A, im_B], outputs=logits)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),  ## feel free to change
        loss="binary_crossentropy",  ## do not change loss/metrics
        metrics=["binary_accuracy"],
    )
    return model              
