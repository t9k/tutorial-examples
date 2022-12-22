import os
import tensorflow as tf

tf.keras.datasets.mnist.load_data(os.path.join(os.getcwd(), 'mnist.npz'))
