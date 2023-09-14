import math

import keras
import tensorflow as tf
from keras import layers

from apneaDetection_transformer.models.transformer import Patches, PatchEncoder, mlp


def mlp(x, hidden_units, dropout_rate, name):
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation=tf.nn.gelu, name=name + "mlpl_" + str(i))(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
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


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, name):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim, name=name + "_lll1")
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim, name=name + "_lll2"
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_decoder_2d(modality_str, input_shape=(16, 16, 1), num_heads=4, transformer_layers=3, projection_dim=16,
                      image_size=16, patch_size=4, output_shape=(128, 16, 1), cnn=True, trainable=True):
    n = modality_str + "_dec"
    transformer_units = [projection_dim * 2, projection_dim, ]
    num_patches = (image_size // patch_size) ** 2

    inputs = layers.Input(shape=input_shape)
    input_norm = layers.Normalization(name=n + "_l1")(inputs)
    input_resize = layers.Resizing(image_size, image_size, name=n + "_l2")(input_norm)
    patches = Patches(patch_size)(input_resize)
    encoded_patches = PatchEncoder(num_patches, projection_dim, name=n + "_l4")(patches)

    for li in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6, name=n + "_l5_" + str(li))(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1, name=n + "_l6_" + str(li)
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6, name=n + "_l7_" + str(li))(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1, name=n + "_l8_" + str(li))
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6, name=n + "_l9")(encoded_patches)
    ##########################################################################
    if not cnn:
        x = layers.Flatten()(representation)
        pre_final = layers.Dense(units=math.prod([item for item in output_shape]), activation="sigmoid")(x)
        outputs = layers.Reshape(output_shape)(pre_final)
    ##########################################################################
    else:
        x = tf.expand_dims(representation, -1)
        x = layers.Conv2DTranspose(32, (4, 4), strides=(2,1), padding='same', activation='relu', name=n + "_l10")(x)
        x = layers.Conv2DTranspose(8, (8, 4), strides=(2,1), padding='same', activation='relu', name=n + "_l11")(x)
        outputs = layers.Conv2DTranspose(output_shape[-1], (16, 4), strides=(2,1), padding='same', activation='sigmoid', name=n + "_l13")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=n, trainable=trainable)
    return model


def create_encoder_2d(modality_str, input_shape=(128, 16, 1), num_heads=4, transformer_layers=3, projection_dim=16,
                      image_size=64,
                      patch_size=16, trainable=True):
    n = modality_str + "_enc"
    transformer_units = [projection_dim * 2, projection_dim, ]
    num_patches = (image_size // patch_size) ** 2

    inputs = layers.Input(shape=input_shape)
    input_norm = layers.Normalization(name=n + "_l1")(inputs)
    input_resize = layers.Resizing(image_size, image_size, name=n + "_l2")(input_norm)
    patches = Patches(patch_size)(input_resize)
    encoded_patches = PatchEncoder(num_patches, projection_dim, name=n + "_l4")(patches)

    for li in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6, name=n + "_l5_" + str(li))(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1, name=n + "_l6_" + str(li)
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6, name=n + "_l7_" + str(li))(x2)
        x3 = mlp(x3, hidden_units=transformer_units, name=n + "_l8_" + str(li), dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6, name=n + "_l9")(encoded_patches)
    representation = tf.expand_dims(representation, -1)
    return keras.Model(inputs=inputs, outputs=representation, name=n, trainable=trainable)
