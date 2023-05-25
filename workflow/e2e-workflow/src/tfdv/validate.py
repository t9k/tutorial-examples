import argparse
import json
import logging
import os

import tensorflow_data_validation as tfdv
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--output',
                        default='./output',
                        type=str,
                        required=True,
                        help='GCS or local directory.')
    parser.add_argument(
        '--csv-data-for-inference',
        default='./eval.csv',
        type=str,
        required=True,
        help='GCS path of the CSV file from which to infer the schema.')
    parser.add_argument(
        '--csv-data-to-validate',
        type=str,
        help='GCS path of the CSV file whose contents should be validated.')
    parser.add_argument(
        '--column-names',
        type=str,
        help='GCS json file containing a list of column names.')

    args = parser.parse_args()
    return args


def convert_feature_to_json(feature):
    feature_json = {'name': feature.name}
    feature_type = schema_pb2.FeatureType.Name(feature.type)
    if (feature_type == 'INT' or feature_type == 'FLOAT'
            or feature.HasField('int_domain')
            or feature.HasField('float_domain')):
        feature_json['type'] = 'NUMBER'
    elif feature.HasField('bool_domain'):
        feature_json['type'] = 'CATEGORY'
    elif feature_type == 'BYTES':
        if (feature.HasField('domain') or feature.HasField('string_domain') or
            (feature.HasField('distribution_constraints')
             and feature.distribution_constraints.min_domain_mass > 0.95)):
            feature_json['type'] = 'CATEGORY'
        else:
            feature_json['type'] = 'TEXT'
    else:
        feature_json['type'] = 'KEY'
    return feature_json


def convert_schema_proto_to_json(schema, column_names):
    column_schemas = {}
    for feature in schema.feature:
        column_schemas[feature.name] = (convert_feature_to_json(feature))
    schema_json = []
    for column_name in column_names:
        schema_json.append(column_schemas[column_name])
    return schema_json


def run_validator(output_dir, column_names, csv_data_file,
                  csv_data_file_to_validate):
    """Writes a TFDV-generated schema.

    Args:
      output_dir: output folder
      column_names: list of names for the columns in the CSV file. If omitted,
          the first line is treated as the column names.
      csv_data_file: name of the CSV file to analyze and generate a schema.
      csv_data_file_to_validate: name of a CSV file to validate
          against the schema.
    """
    stats = tfdv.generate_statistics_from_csv(data_location=csv_data_file,
                                              column_names=column_names,
                                              delimiter=',',
                                              output_path=os.path.join(
                                                  output_dir,
                                                  'data_stats.tfrecord'))
    schema = tfdv.infer_schema(stats)
    schema_json = convert_schema_proto_to_json(schema, column_names)
    schema_json_file = os.path.join(output_dir, 'schema.json')
    with file_io.FileIO(schema_json_file, 'w+') as f:
        logging.getLogger().info('Writing JSON schema to %s', f.name)
        json.dump(schema_json, f)

    if not csv_data_file_to_validate:
        return

    validation_stats = tfdv.generate_statistics_from_csv(
        data_location=csv_data_file_to_validate,
        column_names=column_names,
        delimiter=',',
        output_path=os.path.join(output_dir, 'validation_data_stats.tfrecord'))
    anomalies = tfdv.validate_statistics(validation_stats, schema)

    with file_io.FileIO(os.path.join(output_dir, 'anomalies.pb2'), 'w+') as f:
        logging.getLogger().info('Writing anomalies to %s', f.name)
        f.write(anomalies.SerializeToString())
    for feature_name, anomaly_info in anomalies.anomaly_info.items():
        logging.getLogger().error('Anomaly in feature "%s": %s', feature_name,
                                  anomaly_info.description)


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = parse_arguments()
    column_names = None
    if args.column_names:
        column_names = json.loads(
            file_io.read_file_to_string(args.column_names))

    run_validator(args.output, column_names, args.csv_data_for_inference,
                  args.csv_data_to_validate)


if __name__ == '__main__':
    main()
