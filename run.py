import argparse

import tensorflow as tf
from tensorflow import keras

from vit import create_vit_classifier


parser = argparse.ArgumentParser(description='Simple script to train ViT models')
parser.add_argument('--dataset', choices=['mnist', 'cifar100'], default='mnist')
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--patch_size', type=int, default=8)
parser.add_argument('--projection_dim', type=int, default=128)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--transformer_layers', type=int, default=8)
parser.add_argument('--mlp_head_units', nargs='+', default=[2048, 1024])
parser.add_argument('--learn_position', action='store_true')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args()
print('Args', args)


transformer_units = [
    args.projection_dim * 2,
    args.projection_dim,
]

if args.dataset == 'mnist':
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
else:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f'x_train shape: {x_train.shape} - y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape} - y_test shape: {y_test.shape}')

model = create_vit_classifier(
    args.image_size, args.patch_size, args.learn_position,
    args.transformer_layers, args.mlp_head_units, args.num_heads,
    args.projection_dim, transformer_units, args.dataset)
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name='top-5-accuracy'),
    ],
)
print(model.summary())

checkpoint_filepath = '/tmp/checkpoint'
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
)

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=args.batch_size,
    epochs=args.epochs,
    validation_split=0.1,
    callbacks=[checkpoint_callback],
)

model.load_weights(checkpoint_filepath)
_, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {round(accuracy * 100, 2)}%')
print(f'Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%')
