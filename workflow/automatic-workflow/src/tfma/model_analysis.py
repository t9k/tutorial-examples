from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os

import apache_beam as beam
from ipywidgets.embed import embed_minimal_html
import ipywidgets
import tensorflow as tf
from tensorflow.python.lib.io import file_io

import tensorflow_model_analysis as tfma
import tensorflow_transform as tft

from tensorflow_transform.coders.csv_coder import CsvCoder
from tensorflow_transform.coders.example_proto_coder import ExampleProtoCoder
from tensorflow_transform.tf_metadata import dataset_schema

_OUTPUT_HTML_FILE = 'output_display.html'


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='GCS or local directory.')
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='GCS path to the model which will be evaluated.')
    parser.add_argument('--eval',
                        type=str,
                        required=True,
                        help='GCS path of eval files.')
    parser.add_argument('--schema',
                        type=str,
                        required=True,
                        help='GCS json schema file path.')
    parser.add_argument(
        '--slice-columns',
        type=str,
        action='append',
        required=True,
        help='one or more columns on which to slice for analysis.')

    return parser.parse_args()


def get_raw_feature_spec(schema):
    feature_spec = {}
    for column in schema:
        column_name = column['name']
        column_type = column['type']

        feature = tf.FixedLenFeature(shape=[],
                                     dtype=tf.string,
                                     default_value='')
        if column_type == 'NUMBER':
            feature = tf.FixedLenFeature(shape=[],
                                         dtype=tf.float32,
                                         default_value=0.0)
        feature_spec[column_name] = feature
    return feature_spec


def make_tft_input_metadata(schema):
    """Make a TFT Schema object.

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
    return tft_schema, dataset_schema.schema_utils.schema_from_feature_spec(
        tft_schema)


def run_analysis(output_dir, model_dir, eval_path, schema, slice_columns):
    pipeline_options = None
    runner = 'DirectRunner'

    column_names = [x['name'] for x in schema]
    for slice_column in slice_columns:
        if slice_column not in column_names:
            raise ValueError('Unknown slice column: %s' % slice_column)

    slice_spec = [
        tfma.slicer.SingleSliceSpec(
        ),  # An empty spec is required for the 'Overall' slice
        tfma.slicer.SingleSliceSpec(columns=slice_columns)
    ]

    with beam.Pipeline(runner=runner, options=pipeline_options) as pipeline:
        raw_feature_spec, schema_from = make_tft_input_metadata(schema)
        example_coder = ExampleProtoCoder(
            tft.tf_metadata.dataset_metadata.DatasetMetadata(
                schema_from).schema)
        csv_coder = CsvCoder(
            column_names,
            tft.tf_metadata.dataset_metadata.DatasetMetadata(
                schema_from).schema)

        eval_shared_model = tfma.default_eval_shared_model(
            eval_saved_model_path=model_dir)

        raw_data = (
            pipeline
            | 'ReadFromText' >> beam.io.ReadFromText(eval_path)
            | 'ParseCSV' >> beam.Map(csv_coder.decode)
            # | 'CleanData' >> beam.Map(clean_raw_data_dict(raw_feature_spec))
            | 'ToSerializedTFExample' >> beam.Map(example_coder.encode)
            | 'EvaluateAndWriteResults' >> tfma.ExtractEvaluateAndWriteResults(
                eval_shared_model=eval_shared_model,
                slice_spec=slice_spec,
                output_path=output_dir))


def generate_static_html_output(output_dir, slicing_columns):
    result = tfma.load_eval_result(output_path=output_dir)
    #   slicing_metrics_views = [
    #       tfma.view.render_slicing_metrics(result, slicing_column=slicing_column)
    #       for slicing_column in slicing_columns
    #   ]
    slicing_metrics_view = tfma.view.render_slicing_metrics(
        result, slicing_column=slicing_columns[0])
    static_html_path = os.path.join(output_dir, _OUTPUT_HTML_FILE)
    embed_minimal_html(static_html_path,
                       views=[slicing_metrics_view],
                       title='Slicing Metrics')

    print('ipywidgets.version:', ipywidgets.__version__)

    if os.path.isdir(static_html_path):
        for filename in os.listdir(static_html_path):
            print('################', filename)
    elif os.path.isfile(static_html_path):
        print('###################it is file #################')


def main():
    logger = tf.get_logger()
    logger.setLevel(logging.INFO)
    print('TFMA version: {}'.format(tfma.version.VERSION_STRING))
    #   tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_arguments()
    args.slice_columns = [
        column for column_group in args.slice_columns
        for column in column_group.split(',')
    ]
    schema = json.loads(file_io.read_file_to_string(args.schema))
    eval_model_parent_dir = args.model
    model_export_dir = os.path.join(
        eval_model_parent_dir,
        file_io.list_directory(eval_model_parent_dir)[0])
    run_analysis(args.output, model_export_dir, args.eval, schema,
                 args.slice_columns)
    generate_static_html_output(args.output, args.slice_columns)


if __name__ == '__main__':
    main()
