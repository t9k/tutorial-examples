import argparse
import json
import logging
import os
import time

import tensorflow as tf
from tensorflow.keras import callbacks, datasets, layers, models, optimizers

from t9k import tuner

parser = argparse.ArgumentParser(
    description='Distributed training of Keras model for MNIST with '
    'MultiWorkerMirroredStrategy.')
parser.add_argument('--no_cuda',
                    action='store_true',
                    default=False,
                    help='Disable CUDA training.')
parser.add_argument('--save_path',
                    type=str,
                    default=None,
                    help='Save path of the trained model.')
args = parser.parse_args()
logger = logging.getLogger('print')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.propagate = False

if args.no_cuda:
    tf.config.set_visible_devices([], 'GPU')
gpus = tf.config.get_visible_devices('GPU')
if gpus:
    # Print GPU info
    logger.info('NVIDIA_VISIBLE_DEVICES: {}'.format(
        os.getenv('NVIDIA_VISIBLE_DEVICES')))
    logger.info('T9K_GPU_PERCENT: {}'.format(os.getenv('T9K_GPU_PERCENT')))
    logger.info('Visible GPUs: {}'.format(
        tf.config.get_visible_devices('GPU')))
    # Set memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Get information for current worker.
tf_config = json.loads(os.environ['TF_CONFIG'])
world_size = len(tf_config['cluster']['worker'])
task_index = tf_config['task']['index']

tuner_params = tuner.get_next_parameter()
params = {
    # Search space:
    # 'batch_size': ...
    # 'learning_rate': ...
    # 'conv_channels1': ...
    'epochs': 10,
    'conv_channels2': 64,
    'conv_channels3': 64,
    'conv_kernel_size': 3,
    'maxpool_size': 2,
    'linear_features1': 64,
    'seed': 1,
}
params.update(tuner_params)

with strategy.scope():
    model = models.Sequential([
        layers.Conv2D(params['conv_channels1'],
                      params['conv_kernel_size'],
                      activation='relu',
                      input_shape=(28, 28, 1)),
        layers.MaxPooling2D((params['maxpool_size'], params['maxpool_size'])),
        layers.Conv2D(params['conv_channels2'],
                      params['conv_kernel_size'],
                      activation='relu'),
        layers.MaxPooling2D((params['maxpool_size'], params['maxpool_size'])),
        layers.Conv2D(params['conv_channels3'],
                      params['conv_kernel_size'],
                      activation='relu'),
        layers.Flatten(),
        layers.Dense(params['linear_features1'], activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=params['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'mnist.npz')
(train_images,
 train_labels), (test_images,
                 test_labels) = datasets.mnist.load_data(path=dataset_path)
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255
train_images, val_images = train_images[:48000], train_images[48000:]
train_labels, val_labels = train_labels[:48000], train_labels[48000:]
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)).shuffle(
        48000, seed=params['seed']).repeat().batch(params['batch_size'])
val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_images, val_labels)).batch(400)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_images, test_labels)).batch(1000)

train_callbacks = []
test_callbacks = []

if task_index == 0:
    from t9k.tuner.keras import AutoTuneFitCallback, AutoTuneEvalCallback
    train_callbacks.append(AutoTuneFitCallback(metric='accuracy'))
    test_callbacks.append(AutoTuneEvalCallback(metric='accuracy'))

model.fit(train_images,
          train_labels,
          batch_size=params['batch_size'],
          epochs=params['epochs'],
          validation_split=0.2,
          callbacks=train_callbacks,
          verbose=2)

model.evaluate(test_images, test_labels, callbacks=test_callbacks, verbose=2)

if task_index > 0:
    # wait a while for index 0
    time.sleep(1)
