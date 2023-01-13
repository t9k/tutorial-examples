import argparse
import os
import shutil

import horovod.tensorflow.keras as hvd
import tensorflow as tf
from tensorflow.keras import callbacks, datasets, layers, models, optimizers

parser = argparse.ArgumentParser(
    description='Distributed training of Keras model for MNIST using horovod.')
parser.add_argument('--log_dir',
                    type=str,
                    help='Path of the TensorBoard log directory.')
parser.add_argument('--no_cuda',
                    action='store_true',
                    default=False,
                    help='Disables CUDA training.')
args = parser.parse_args()

hvd.init()

if args.no_cuda:
    # Sets all GPUs invisible
    tf.config.set_visible_devices([], 'GPU')
else:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()],
                                                   'GPU')

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

opt = optimizers.Adam(learning_rate=0.001 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'mnist.npz')
(train_images,
 train_labels), (test_images,
                 test_labels) = datasets.mnist.load_data(path=dataset_path)
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_images, val_images = tf.split(train_images, [48000, 12000], axis=0)
train_labels, val_labels = tf.split(train_labels, [48000, 12000], axis=0)
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images,
     train_labels)).shuffle(48000).shard(num_shards=hvd.size(),
                                         index=hvd.rank()).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_images, val_labels)).shard(num_shards=hvd.size(),
                                    index=hvd.rank()).batch(100)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_images, test_labels)).batch(1000)

train_callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(initial_lr=0.001 * hvd.size(),
                                             warmup_epochs=3),
]

if args.log_dir and hvd.rank() == 0:
    log_dir = args.log_dir
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir, ignore_errors=True)
    train_callbacks.append(callbacks.TensorBoard(log_dir=log_dir))

verbose = 1 if hvd.rank() == 0 else 0

model.fit(train_dataset,
          epochs=10,
          validation_data=val_dataset,
          callbacks=train_callbacks,
          verbose=verbose)

if hvd.rank() == 0:
    model.evaluate(test_dataset)
