import argparse
import json
import os
import shutil

import tensorflow as tf
from tensorflow.keras import callbacks, datasets, layers, models, optimizers

parser = argparse.ArgumentParser(
    description='Distributed training of Keras model for MNIST with '
    'MultiWorkerMirroredStrategy.')
parser.add_argument('--log_dir',
                    type=str,
                    help='Path of the TensorBoard log directory.')
parser.add_argument('--no_cuda',
                    action='store_true',
                    default=False,
                    help='Disable CUDA training.')
args = parser.parse_args()

if args.no_cuda:
    tf.config.set_visible_devices([], 'GPU')

strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Get world size and index of current worker.
tf_config = json.loads(os.environ['TF_CONFIG'])
world_size = len(tf_config['cluster']['worker'])
task_index = tf_config['task']['index']

with strategy.scope():
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001 * world_size),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'mnist.npz')
(train_images,
 train_labels), (test_images,
                 test_labels) = datasets.mnist.load_data(path=dataset_path)
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255
train_images, val_images = tf.split(train_images, [48000, 12000], axis=0)
train_labels, val_labels = tf.split(train_labels, [48000, 12000], axis=0)
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)).shuffle(48000).repeat().batch(128)
val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_images, val_labels)).batch(400)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_images, test_labels)).batch(1000)

train_callbacks = []
if args.log_dir and task_index == 0:
    log_dir = args.log_dir
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir, ignore_errors=True)
    train_callbacks.append(callbacks.TensorBoard(log_dir=log_dir))

model.fit(train_dataset,
          epochs=10,
          steps_per_epoch=375,
          validation_data=val_dataset,
          callbacks=train_callbacks,
          verbose=2)

model.evaluate(test_images, test_labels, verbose=2)
