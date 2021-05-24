import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_augmentation(dataset: str, image_size: int):
    if dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        aug_layers = [
            layers.experimental.preprocessing.Normalization(),
            layers.experimental.preprocessing.Resizing(image_size, image_size),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(factor=0.02),
            layers.experimental.preprocessing.RandomZoom(
                height_factor=0.2, width_factor=0.2)]
    else:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        aug_layers = [
            layers.experimental.preprocessing.Normalization(),
            layers.experimental.preprocessing.Resizing(image_size, image_size),
            layers.experimental.preprocessing.RandomRotation(factor=0.02),
            layers.experimental.preprocessing.RandomZoom(
                height_factor=0.2, width_factor=0.2)]

    data_augmentation = keras.Sequential(aug_layers, name='data_augmentation')
    # Compute the mean and the variance of the training data for normalization.
    data_augmentation.layers[0].adapt(x_train)
    return data_augmentation


def mlp(x, hidden_units: list, dropout_rate: float):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)

    return x


class Patches(layers.Layer):
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
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, learn_position, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.learn_position = learn_position
        self.projection = layers.Dense(projection_dim, name='projection')
        if self.learn_position:
            self.position_embedding = layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim,
                name='embeddings'
            )

    def build(self, input_shape):
        super(PatchEncoder, self).build(input_shape)

    def call(self, patch):
        input_shape = tf.shape(patch)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        proj = self.projection(patch)
        if self.learn_position:
            pos = self.position_embedding(positions)
            encoded = proj + pos
            return encoded
        else:
            return tf.reshape(proj, self.compute_output_shape(input_shape))

    def compute_output_shape(self, input_shape):
        return (-1, self.num_patches, self.projection_dim)


def create_vit_classifier(
        image_size: int, patch_size: int,
        learn_position: bool, transformer_layers: list,
        mlp_head_units: list, num_heads: int,
        projection_dim: int, transformer_units: int,
        dataset: str):

    if dataset == 'mnist':
        input_shape = (28, 28, 1)
        num_classes = 10
        data_augmentation = create_augmentation(dataset, image_size)
    else:
        input_shape = (32, 32, 3)
        num_classes = 100
        data_augmentation = create_augmentation(dataset, image_size)

    num_patches = (image_size // patch_size) ** 2

    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(
        num_patches, projection_dim,
        learn_position=True, name='PatchEncoder')(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes, activation='softmax')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
