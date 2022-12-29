import apache_beam as beam
import argparse
import datetime
import csv
import json
import logging
import os
import tensorflow as tf
import tensorflow_transform as tft

import tensorflow_transform as transform


from apache_beam.io import textio
from apache_beam.io import tfrecordio
from apache_beam.options.pipeline_options import PipelineOptions

from tensorflow.python.lib.io import file_io
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders.csv_coder import CsvCoder
from tensorflow_transform.coders.example_proto_coder import ExampleProtoCoder
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io


def parse_arguments():
  """Parse command line arguments."""

  parser = argparse.ArgumentParser()
  parser.add_argument('--output',
                      type=str,
                      required=True,
                      help='GCS or local directory.')
  parser.add_argument('--train',
                      type=str,
                      required=True,
                      help='GCS path of train file patterns.')
  parser.add_argument('--eval',
                      type=str,
                      required=True,
                      help='GCS path of eval file patterns.')
  parser.add_argument('--schema',
                      type=str,
                      required=True,
                      help='GCS json schema file path.')

  args = parser.parse_args()
  return args


# Categorical features are assumed to each have a maximum value in the dataset.
MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]
CATEGORICAL_FEATURE_KEYS = [
    'trip_start_hour',
    'trip_start_day',
    'trip_start_month',
    'dropoff_census_tract',
    'pickup_community_area',
    'dropoff_community_area'
]

DENSE_FLOAT_FEATURE_KEYS = [
    'trip_miles',
    'fare',
    'trip_seconds'
]

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

def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)

def preprocess(inputs):
  """tf.transform's callback function for preprocessing inputs.
  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in DENSE_FLOAT_FEATURE_KEYS:
    # Preserve this feature as a dense float, setting nan's to the mean.
    outputs[key] = transform.scale_to_z_score(_fill_in_missing(inputs[key]))

  for key in VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    outputs[key] = transform.compute_and_apply_vocabulary(
        _fill_in_missing(inputs[key]),
        top_k=VOCAB_SIZE,
        num_oov_buckets=OOV_SIZE)

  for key in BUCKET_FEATURE_KEYS:
    outputs[key] = transform.bucketize(_fill_in_missing(inputs[key]), FEATURE_BUCKET_COUNT,
        always_return_num_quantiles=False)

  for key in CATEGORICAL_FEATURE_KEYS:
    outputs[key] = tf.cast(_fill_in_missing(inputs[key]), tf.int64)

  taxi_fare = _fill_in_missing(inputs[FARE_KEY])
  tips = _fill_in_missing(inputs[LABEL_KEY])
  # Test if the tip was > 20% of the fare.
  outputs[LABEL_KEY] = tf.where(
      tf.math.is_nan(taxi_fare),
      tf.cast(tf.zeros_like(taxi_fare), tf.int64),
      # Test if the tip was > 20% of the fare.
      tf.cast(
          tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))), tf.int64))

  return outputs


def make_tft_input_metadata(schema):
  """Make a TFT Schema object
  In the tft framework, this is where default values are recoreded for training.
  Args:
    schema: schema list of training data.
  Returns:
    TFT metadata object.
  """
  tft_schema = {}

  for col_schema in schema:
    col_type = col_schema['type']
    col_name = col_schema['name']
    if col_type == 'NUMBER':
      tft_schema[col_name] = tf.io.VarLenFeature(tf.float32)
    elif col_type in ['CATEGORY', 'TEXT', 'IMAGE_URL', 'KEY']:
      tft_schema[col_name] = tf.io.VarLenFeature(tf.string)
  return dataset_schema.schema_utils.schema_from_feature_spec(tft_schema)


def run_transform(output_dir, schema_list, train_data_file, eval_data_file,
                  preprocessing_fn=None):
  """Writes a tft transform fn, and metadata files.
  Args:
    output_dir: output folder
    schema_list: schema list.
    train_data_file: training data file pattern.
    eval_data_file: eval data file pattern.
    local: whether the job should be local or cloud.
    preprocessing_fn: a function used to preprocess the raw data. If not
                      specified, a function will be automatically inferred
                      from the schema_list.
  """

  tft_input_metadata = tft.tf_metadata.dataset_metadata.DatasetMetadata(make_tft_input_metadata(schema_list))
  temp_dir = os.path.join(output_dir, 'tmp')
  preprocessing_fn = preprocessing_fn

  with beam.Pipeline() as p:
    with beam_impl.Context(temp_dir=temp_dir):
      names = [x['name'] for x in schema_list]
      converter = CsvCoder(names, tft_input_metadata.schema)
      train_data = (
          p
          | 'ReadTrainData' >> textio.ReadFromText(train_data_file)
          | 'DecodeTrainData' >> beam.Map(converter.decode))

      train_dataset = (train_data, tft_input_metadata)
      transformed_dataset, transform_fn = (
          train_dataset | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))
      transformed_data, transformed_metadata = transformed_dataset

      # Writes transformed_metadata and transfrom_fn folders
      _ = (transform_fn | 'WriteTransformFn' >> transform_fn_io.WriteTransformFn(output_dir))

      # Write the raw_metadata
      metadata_io.write_metadata(
          metadata=tft_input_metadata,
          path=os.path.join(output_dir, 'metadata'))

      _ = transformed_data | 'WriteTrainData' >> tfrecordio.WriteToTFRecord(
          os.path.join(output_dir, 'train'),
          coder=ExampleProtoCoder(transformed_metadata.schema))

      eval_data = (
          p
          | 'ReadEvalData' >> textio.ReadFromText(eval_data_file)
          | 'DecodeEvalData' >> beam.Map(converter.decode))

      eval_dataset = (eval_data, tft_input_metadata)

      transformed_eval_dataset = (
          (eval_dataset, transform_fn) | beam_impl.TransformDataset())
      transformed_eval_data, transformed_metadata = transformed_eval_dataset

      _ = transformed_eval_data | 'WriteEvalData' >> tfrecordio.WriteToTFRecord(
          os.path.join(output_dir, 'eval'),
          coder=ExampleProtoCoder(transformed_metadata.schema))


def main():
  logging.getLogger().setLevel(logging.INFO)
  args = parse_arguments()
  schema = json.loads(file_io.read_file_to_string(args.schema))

  def wrapped_preprocessing_fn(inputs):
    outputs = preprocess(inputs)
    for key in outputs:
      if outputs[key].dtype == tf.bool:
        print('****************bool feature:', key)
        outputs[key] = tft.string_to_int(tf.as_string(outputs[key]),
                                          vocab_filename='vocab_' + key)
    return outputs

  preprocessing_fn = wrapped_preprocessing_fn

  run_transform(args.output, schema, args.train, args.eval,
                preprocessing_fn=preprocessing_fn)

if __name__== "__main__":
  main()
