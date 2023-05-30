# A program to generate confusion matrix data out of prediction results.
# Usage:
# python confusion_matrix.py  \
#   --predictions=gs://bradley-playground/sfpd/predictions/part-* \
#   --output=gs://bradley-playground/sfpd/cm/ \
#   --target=resolution \
#   --analysis=gs://bradley-playground/sfpd/analysis \

import argparse
import json
import os

import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.python.lib.io import file_io


def main(argv=None):
    parser = argparse.ArgumentParser(description='ML Trainer')
    parser.add_argument('--predictions',
                        type=str,
                        help='Path of prediction file pattern.')
    parser.add_argument('--output',
                        type=str,
                        help='Path of the output directory.')
    parser.add_argument('--output-schema',
                        type=str,
                        help='Schema for confusion matrix.')
    args = parser.parse_args()

    schema = json.loads(file_io.read_file_to_string(args.output_schema))
    names = [x['name'] for x in schema]
    dfs = []
    files = file_io.get_matching_files(args.predictions)
    for file in files:
        with file_io.FileIO(file, 'r') as f:
            dfs.append(pd.read_csv(f, names=names))

    df = pd.concat(dfs)
    target_lambda = """lambda x: (x['target'] > x['fare'] * 0.2)"""
    df['target'] = df.apply(eval(target_lambda), axis=1)

    vocab = list(df['target'].unique())
    cm = confusion_matrix(df['target'], df['predicted'], labels=vocab)
    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))

    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    cm_file = os.path.join(args.output, 'confusion_matrix.csv')
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(cm_file, 'w') as f:
        df_cm.to_csv(f,
                     columns=['target', 'predicted', 'count'],
                     header=False,
                     index=False)


if __name__ == '__main__':
    main()
