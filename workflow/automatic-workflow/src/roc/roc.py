# A program to generate ROC data out of prediction results.
# Usage:
# python roc.py  \
#   --predictions=gs://bradley-playground/sfpd/predictions/part-* \
#   --trueclass=ACTION \
#   --output=gs://bradley-playground/sfpd/roc/ \


import argparse
import json
import os
import pandas as pd
import tarfile

from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.python.lib.io import file_io


def main(argv=None):
  parser = argparse.ArgumentParser(description='ML Trainer')
  parser.add_argument('--predictions', type=str, help='Path of prediction file pattern.')
  parser.add_argument('--trueclass', type=str, default='true',
                      help='The name of the class as true value. If missing, assuming it is ' +
                           'binary classification and default to "true".')
  parser.add_argument('--true_score_column', type=str, default='true',
                      help='The name of the column for positive prob. If missing, assuming it is ' +
                           'binary classification and defaults to "true".')
  parser.add_argument('--output-schema', type=str, help='Output schema file.')
  parser.add_argument('--output', type=str, help='Path of the output directory.')
  args = parser.parse_args()

  if not os.path.exists(args.output):
    os.makedirs(args.output)

  schema = json.loads(file_io.read_file_to_string(args.output_schema))
  names = [x['name'] for x in schema]

  if args.true_score_column not in names:
    raise ValueError('Cannot find column name "%s"' % args.true_score_column)

  dfs = []
  files = file_io.get_matching_files(args.predictions)
  for file in files:
    with file_io.FileIO(file, 'r') as f:
      dfs.append(pd.read_csv(f, names=names))

  df = pd.concat(dfs)
  target_lambda = """lambda x: 1 if (x['target'] > x['fare'] * 0.2) else 0"""
  df['target'] = df.apply(eval(target_lambda), axis=1)
  fpr, tpr, thresholds = roc_curve(df['target'], df[args.true_score_column])
  roc_auc = roc_auc_score(df['target'], df[args.true_score_column])
  df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
  roc_file = os.path.join(args.output, 'roc.csv')
  with file_io.FileIO(roc_file, 'w') as f:
    df_roc.to_csv(f, columns=['fpr', 'tpr', 'thresholds'], header=False, index=False)


if __name__== "__main__":
  main()
