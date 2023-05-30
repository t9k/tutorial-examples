import argparse
import json
import os
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma

from tensorflow.python.lib.io import file_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
# from tensorflow_transform.saved import input_fn_maker

IMAGE_EMBEDDING_SIZE = 2048
CLASSIFICATION_TARGET_TYPES = [tf.bool, tf.int32, tf.int64]
REGRESSION_TARGET_TYPES = [tf.float32, tf.float64]
TARGET_TYPES = CLASSIFICATION_TARGET_TYPES + REGRESSION_TARGET_TYPES

# Categorical features are assumed to each have a maximum value in the dataset.
MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]
CATEGORICAL_FEATURE_KEYS = [
    'trip_start_hour', 'trip_start_day', 'trip_start_month',
    'pickup_census_tract', 'dropoff_census_tract', 'pickup_community_area',
    'dropoff_community_area'
]

DENSE_FLOAT_FEATURE_KEYS = ['trip_miles', 'fare', 'trip_seconds']

# Number of buckets used by tf.transform for encoding each feature.
FEATURE_BUCKET_COUNT = 10

BUCKET_FEATURE_KEYS = [
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
    'dropoff_longitude'
]

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

VOCAB_FEATURE_KEYS = [
    'payment_type',
    'company',
]
LABEL_KEY = 'tips'
FARE_KEY = 'fare'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='Output directory.')
    parser.add_argument(
        '--transformed-data-dir',
        type=str,
        required=True,
        help='Directory path containing tf-transformed training and eval data.'
    )
    parser.add_argument('--schema',
                        type=str,
                        required=True,
                        help='Schema file path.')
    parser.add_argument('--predict-data',
                        type=str,
                        required=True,
                        help='Directory path containing the predict data.')
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        help='The name of the column to predict in training data.')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.1,
                        help='Learning rate for training.')
    parser.add_argument(
        '--hidden-layer-size',
        type=str,
        default='100',
        help='comma separated hidden layer sizes. For example "200,100,50".')
    parser.add_argument(
        '--steps',
        type=int,
        help='Maximum number of training steps to perform. If unspecified, will '
        'honor epochs.')
    parser.add_argument(
        '--epochs',
        type=int,
        help='Maximum number of training data epochs on which to train. If '
        'both "steps" and "epochs" are specified, the training '
        'job will run for "steps" or "epochs", whichever occurs first.')

    args = parser.parse_args()
    args.hidden_layer_size = [
        int(x.strip()) for x in args.hidden_layer_size.split(',')
    ]
    return args


def is_classification(transformed_data_dir, target):
    """Whether the scenario is classification (vs regression).

    Returns:
        The number of classes if the target represents a classification
        problem, or None if it does not.
  """
    #   transformed_metadata = metadata_io.read_metadata(
    #       os.path.join(transformed_data_dir, transform_fn_io.TRANSFORMED_METADATA_DIR))
    tf_transform_output = tft.TFTransformOutput(transformed_data_dir)
    transformed_feature_spec = tf_transform_output.transformed_feature_spec()
    if target not in transformed_feature_spec:
        raise ValueError('Cannot find target "%s" in transformed data.' %
                         target)

    feature = transformed_feature_spec[target]
    if (not isinstance(feature, tf.FixedLenFeature) or feature.shape != []
            or feature.dtype not in TARGET_TYPES):
        raise ValueError('target "%s" is of invalid type.' % target)

    if feature.dtype in CLASSIFICATION_TARGET_TYPES:
        if feature.dtype == tf.bool:
            return 2
        return get_vocab_size(transformed_data_dir, target)

    return None


def make_training_input_fn(transformed_data_dir,
                           mode,
                           batch_size,
                           target_name,
                           num_epochs=None):
    """Creates an input function reading from transformed data.

    Args:
        transformed_data_dir: Directory to read transformed data and metadata
            from.
        mode: 'train' or 'eval'.
        batch_size: Batch size.
        target_name: name of the target column.
        num_epochs: number of training data epochs.
    Returns:
        The input function for training or eval.
  """
    tf_transform_output = tft.TFTransformOutput(transformed_data_dir)

    def _input_fn():
        """Input function for training and eval."""
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=os.path.join(transformed_data_dir, mode + '*'),
            batch_size=batch_size,
            features=tf_transform_output.transformed_feature_spec(),
            reader=tf.data.TFRecordDataset,
            shuffle=False,
            num_epochs=1)
        transformed_features = tf.compat.v1.data.make_one_shot_iterator(
            dataset).get_next()

        # Extract features and label from the transformed tensors.
        transformed_labels = transformed_features.pop(target_name)
        return transformed_features, transformed_labels

    return _input_fn


def make_serving_input_fn(transformed_data_dir, target_name):
    """Creates an input function reading from transformed data.

    Args:
        transformed_data_dir: Directory to read transformed data and metadata
            from.
        target_name: name of the target column.
    Returns:
        The input function for serving.
  """
    tf_transform_output = tft.TFTransformOutput(transformed_data_dir)
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_feature_spec.pop(target_name)
    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)
    serving_input_receiver = raw_input_fn()
    transformed_features = tf_transform_output.transform_raw_features(
        serving_input_receiver.features)

    serving_input_fn = tf.estimator.export.ServingInputReceiver(
        transformed_features, serving_input_receiver.receiver_tensors)

    return serving_input_fn


def get_vocab_size(transformed_data_dir, feature_name):
    """Get vocab size of a given text or category column."""
    vocab_file = os.path.join(transformed_data_dir,
                              transform_fn_io.TRANSFORM_FN_DIR, 'assets',
                              'vocab_' + feature_name)
    with file_io.FileIO(vocab_file, 'r') as f:
        return sum(1 for _ in f)


def get_feature_columns(transformed_data_dir):
    """Callback that returns a list of feature columns for building a
    tf.estimator.

    Args:
        transformed_data_dir: The directory holding the output of the tft
            transformation.

    Returns:
        A list of tf.feature_column.
  """
    real_valued_columns = [
        tf.feature_column.numeric_column(key, shape=())
        for key in DENSE_FLOAT_FEATURE_KEYS
    ]

    categorical_columns = [
        tf.feature_column.categorical_column_with_identity(
            key, num_buckets=VOCAB_SIZE + OOV_SIZE, default_value=0)
        for key in VOCAB_FEATURE_KEYS
    ]
    categorical_columns += [
        tf.feature_column.categorical_column_with_identity(
            key, num_buckets=FEATURE_BUCKET_COUNT, default_value=0)
        for key in BUCKET_FEATURE_KEYS
    ]
    categorical_columns += [
        tf.feature_column.categorical_column_with_identity(
            key, num_buckets=num_buckets, default_value=0) for key, num_buckets
        in zip(CATEGORICAL_FEATURE_KEYS, MAX_CATEGORICAL_FEATURE_VALUES)
    ]

    return categorical_columns, real_valued_columns


def write_to_schema(schema, output):
    output_schema = list(filter(lambda x: x['name'] != 'tips', schema))

    output_schema.append({'name': 'target', 'type': 'CATEGORY'})
    output_schema.append({'name': 'predicted', 'type': 'CATEGORY'})
    output_schema.append({'name': 'false', 'type': 'NUMBER'})
    output_schema.append({'name': 'true', 'type': 'NUMBER'})
    string = json.dumps(output_schema)
    with open(os.path.join(output, 'output_schema.json'), 'w') as f:
        f.write(string)
    return output_schema


def write_to_pvc(predict_data, output, predictions):
    with open(predict_data) as f:
        with open(os.path.join(output, 'prediction_results-00000-of-00001'),
                  'w') as fo:
            i = 0
            for line in f:
                if predictions[i]['probabilities'][0] > 0.5:
                    predicted = 'false'
                fo.write(line.strip() + ',' + predicted + ',' +
                         str(predictions[i]['probabilities'][0]) + ',' +
                         str(predictions[i]['probabilities'][1]) + '\n')
                i += 1


def get_estimator(transformed_data_dir, target_name, output_dir, hidden_units,
                  learning_rate, categorical_columns, real_valued_columns):
    # Set how often to run checkpointing in terms of steps.
    config = tf.estimator.RunConfig(save_checkpoints_steps=1000)
    n_classes = 2
    if n_classes:
        estimator = tf.estimator.DNNLinearCombinedClassifier(
            dnn_feature_columns=real_valued_columns,
            linear_feature_columns=categorical_columns,
            dnn_hidden_units=hidden_units,
            n_classes=n_classes,
            config=config,
            model_dir=os.path.join(output_dir, 'model'))

    return estimator


def eval_input_receiver_fn(tf_transform_dir, target):
    """Build everything needed for the tf-model-analysis to run the model.

    Args:
        tf_transform_dir: directory in which the tf-transform model was written
            during the preprocessing step.
        target: name of the target column.

    Returns:
        EvalInputReceiver function, which contains:
        - Tensorflow graph which parses raw untranformed features, applies the
          tf-transform preprocessing operators.
        - Set of raw, untransformed features.
        - Label against which predictions will be compared.
    """
    tf_transform_output = tft.TFTransformOutput(tf_transform_dir)
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    serialized_tf_example = tf.compat.v1.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')
    features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = tf_transform_output.transform_raw_features(features)
    features.update(transformed_features)
    receiver_tensors = {'examples': serialized_tf_example}
    return tfma.export.EvalInputReceiver(features=features,
                                         receiver_tensors=receiver_tensors,
                                         labels=transformed_features[target])


def main():
    # configure the TF_CONFIG such that the tensorflow recoginzes the MASTER in the yaml file as the chief.
    # TODO: kubeflow is working on fixing the problem and this TF_CONFIG can be
    # removed then.
    import logging
    logger = tf.get_logger()
    logger.setLevel(logging.INFO)

    args = parse_arguments()
    #   tf.logging.set_verbosity(tf.logging.INFO)

    categorical_columns, real_valued_columns = get_feature_columns(
        args.transformed_data_dir)

    estimator = get_estimator(args.transformed_data_dir, args.target,
                              args.output, args.hidden_layer_size,
                              args.learning_rate, categorical_columns,
                              real_valued_columns)

    # TODO: Expose batch size.
    train_input_fn = make_training_input_fn(args.transformed_data_dir,
                                            'train',
                                            32,
                                            args.target,
                                            num_epochs=args.epochs)

    eval_input_fn = make_training_input_fn(args.transformed_data_dir, 'eval',
                                           32, args.target)
    serving_input_fn = lambda: make_serving_input_fn(args.transformed_data_dir,
                                                     args.target)

    exporter = tf.estimator.FinalExporter('export', serving_input_fn)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=args.steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      exporters=[exporter],
                                      name='chicago-taxi-eval')
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    predictions = estimator.predict(input_fn=eval_input_fn,
                                    yield_single_examples=True)
    write_to_pvc(args.predict_data, args.output, list(predictions))
    schema = json.loads(file_io.read_file_to_string(args.schema))
    write_to_schema(schema, args.output)
    eval_model_dir = os.path.join(args.output, 'model/tfma_eval_model_dir')
    tfma.export.export_eval_savedmodel(
        estimator=estimator,
        export_dir_base=eval_model_dir,
        eval_input_receiver_fn=(lambda: eval_input_receiver_fn(
            args.transformed_data_dir, args.target)))


if __name__ == '__main__':
    main()
