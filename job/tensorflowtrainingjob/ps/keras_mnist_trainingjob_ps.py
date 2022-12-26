import argparse
import os

import tensorflow as tf

parser = argparse.ArgumentParser(
    description='Distributed training of Keras model for MNIST with '
    'ParameterServerStrategy.')
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

cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

if cluster_resolver.task_type in ("worker", "ps"):
    # Set the environment variable to allow reporting worker and ps failure to the
    # coordinator. This is a workaround and won't be necessary in the future.
    os.environ["GRPC_FAIL_FAST"] = "use_caller"

    server = tf.distribute.Server(cluster_resolver.cluster_spec(),
                                  job_name=cluster_resolver.task_type,
                                  task_index=cluster_resolver.task_id,
                                  protocol=cluster_resolver.rpc_layer
                                  or "grpc",
                                  start=True)
    server.join()

else:  # task_type == "chief"
    import shutil

    from tensorflow.keras import callbacks, datasets, layers, models, optimizers

    variable_partitioner = (
        tf.distribute.experimental.partitioners.MinSizePartitioner(
            min_shard_bytes=(256 << 10), max_shards=2))
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver, variable_partitioner=variable_partitioner)
    worker_num = cluster_resolver.cluster_spec().num_tasks('worker')

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
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001 *
                                                worker_num),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    def dataset_fn(input_context):
        dataset_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'mnist.npz')
        (train_images,
         train_labels), _ = datasets.mnist.load_data(path=dataset_path)

        train_images = train_images.reshape((60000, 28, 28, 1))
        train_images = train_images / 255.0

        batch_size = input_context.get_per_replica_batch_size(32 * worker_num)

        dataset = tf.data.Dataset.from_tensor_slices(
            (train_images, train_labels)).shuffle(60000).repeat()
        dataset = dataset.shard(input_context.num_input_pipelines,
                                input_context.input_pipeline_id)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2)
        return dataset

    dc = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

    train_callbacks = []
    if args.log_dir:
        log_dir = args.log_dir
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir, ignore_errors=True)
        train_callbacks.append(callbacks.TensorBoard(log_dir=log_dir))

    model.fit(dc,
              epochs=10,
              steps_per_epoch=450,
              callbacks=train_callbacks,
              verbose=2)
