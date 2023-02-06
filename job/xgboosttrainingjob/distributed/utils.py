# Copyright (c) 2019 Kubeflow
# Modifications copyright (C) 2023 TensorStack
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import joblib
import xgboost as xgb
import os
import pandas as pd

from sklearn import datasets

logger = logging.getLogger(__name__)


def extract_xgbooost_cluster_env():
    """
    Extract the cluster env from pod
    :return: the related cluster env to build rabit
    """

    logger.info("starting to extract system env")

    master_addr = os.environ.get("MASTER_ADDR", "{}")
    master_port = int(os.environ.get("MASTER_PORT", "{}"))
    rank = int(os.environ.get("RANK", "{}"))
    world_size = int(os.environ.get("WORLD_SIZE", "{}"))

    logger.info(
        "extract the Rabit env from cluster :"
        " %s, port: %d, rank: %d, word_size: %d ", master_addr, master_port,
        rank, world_size)

    return master_addr, master_port, rank, world_size


def read_train_data(rank, num_workers, path):
    """
    Read file based on the rank of worker.
    We use the sklearn.iris data for demonstration
    You can extend this to read distributed data source like HDFS, HIVE etc
    :param rank: the id of each worker
    :param num_workers: total number of workers in this cluster
    :param path: the input file name or the place to read the data
    :return: XGBoost Dmatrix
    """
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    start, end = get_range_data(len(x), rank, num_workers)
    x = x[start:end, :]
    y = y[start:end]

    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    dtrain = xgb.DMatrix(data=x, label=y)

    logging.info("Read data from IRIS data source with range from %d to %d",
                 start, end)

    return dtrain


def read_predict_data(rank, num_workers, path):
    """
    Read file based on the rank of worker.
    We use the sklearn.iris data for demonstration
    You can extend this to read distributed data source like HDFS, HIVE etc
    :param rank: the id of each worker
    :param num_workers: total number of workers in this cluster
    :param path: the input file name or the place to read the data
    :return: XGBoost Dmatrix, and real value
    """
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    start, end = get_range_data(len(x), rank, num_workers)
    x = x[start:end, :]
    y = y[start:end]
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    logging.info("Read data from IRIS datasource with range from %d to %d",
                 start, end)

    predict = xgb.DMatrix(x, label=y)

    return predict, y


def get_range_data(num_row, rank, num_workers):
    """
    compute the data range based on the input data size and worker id
    :param num_row: total number of dataset
    :param rank: the worker id
    :param num_workers: total number of workers
    :return: begin and end range of input matrix
    """
    num_per_partition = int(num_row / num_workers)

    x_start = rank * num_per_partition
    x_end = (rank + 1) * num_per_partition

    if x_end > num_row:
        x_end = num_row

    return x_start, x_end


def dump_model(model, model_path, args):
    """
    dump the trained model into local place
    you can update this function to store the model into a remote place
    :param model: the xgboost trained booster
    :param model_path: place to store model
    :param args: configuration for model storage
    :return: True if the dump process success
    """
    if model is None:
        raise Exception("fail to get the XGBoost train model")
    else:
        joblib.dump(model, model_path)
        logging.info("Dump model into local place %s", model_path)

    return True


def read_model(model_path, args):
    """
    read model from physical storage
    :param model_path: place to store the model
    :param args: configuration to read model
    :return: XGBoost model
    """

    model = joblib.load(model_path)
    logging.info("Read model from local place %s", model_path)

    return model


def parse_parameters(input, splitter_between, splitter_in):
    """
    helper function parse the input parameter
    :param input: the string of configuration like key-value pairs
    :param splitter_between: the splitter between config for input string
    :param splitter_in: the splitter inside config for input string
    :return: key-value pair configuration
    """

    ky_pairs = input.split(splitter_between)

    confs = {}

    for kv in ky_pairs:
        conf = kv.split(splitter_in)
        key = conf[0].strip(" ")
        if key == "objective" or key == "endpoint":
            value = conf[1].strip("'") + ":" + conf[2].strip("'")
        else:
            value = conf[1]

        confs[key] = value
    return confs
