# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
import time


COLUMNS = ["UserID", "AnimeID", "UserRating",
           "Genre0", "Genre1", "Genre2", "Genre3", "Genre4", "Genre5", "Genre6", "Genre7", "Genre8", "Genre9", "Genre10",
           "Genre11", "Genre12", "Genre13", "Genre14", "Genre15", "Genre16", "Genre17", "Genre18", "Genre19", "Genre20",
           "Genre21", "Genre22", "Genre23", "Genre24", "Genre25", "Genre26", "Genre27", "Genre28", "Genre29", "Genre30",
           "Genre31", "Genre32", "Genre33", "Genre34", "Genre35", "Genre36", "Genre37", "Genre38", "Genre39", "Genre40",
           "Genre41", "Genre42",
           "MediaType", "Episodes", "OverallRating", "ListMembership"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["UserID", "AnimeID",
                       "Genre0", "Genre1", "Genre2", "Genre3", "Genre4", "Genre5", "Genre6", "Genre7", "Genre8", "Genre9", "Genre10",
                       "Genre11", "Genre12", "Genre13", "Genre14", "Genre15", "Genre16", "Genre17", "Genre18", "Genre19", "Genre20",
                       "Genre21", "Genre22", "Genre23", "Genre24", "Genre25", "Genre26", "Genre27", "Genre28", "Genre29", "Genre30",
                       "Genre31", "Genre32", "Genre33", "Genre34", "Genre35", "Genre36", "Genre37", "Genre38", "Genre39", "Genre40",
                       "Genre41", "Genre42",
                       "MediaType"]
CONTINUOUS_COLUMNS = ["Episodes", "OverallRating", "ListMembership"]
print ("Start:" + str(time.strftime("%H:%M:%S")))


def build_estimator(model_dir, model_type, user_ids, anime_ids):
    """Build an estimator."""
    # Sparse base columns.
    UserID = tf.contrib.layers.sparse_column_with_integerized_feature(
        "UserID", user_ids + 1, combiner="sqrtn")
    AnimeID = tf.contrib.layers.sparse_column_with_integerized_feature(
        "AnimeID", anime_ids + 1, combiner="sqrtn")
    Genres = []
    for i in range(43):
        Genres.append(tf.contrib.layers.sparse_column_with_integerized_feature(
            "Genre" + str(i), 2, combiner="sqrtn"))
    MediaType = tf.contrib.layers.sparse_column_with_keys(column_name="MediaType",
        keys=["TV", "Movie", "Special", "OVA", "ONA", "Music"], combiner="sqrtn")

    # Continuous base columns.
    Episodes = tf.contrib.layers.real_valued_column("Episodes")
    OverallRating = tf.contrib.layers.real_valued_column("OverallRating")
    ListMembership = tf.contrib.layers.real_valued_column("ListMembership")

    # Transformations.
    #age_buckets = tf.contrib.layers.bucketized_column(age,
    #                                                boundaries=[
    #                                                    18, 25, 30, 35, 40, 45,
    #                                                    50, 55, 60, 65
    #                                                ])

    # Wide columns
    wide_columns = [UserID, AnimeID,
        tf.contrib.layers.crossed_column(Genres, hash_bucket_size=int(1e4), combiner="sqrtn")]

    # Deep columns
    deep_columns = []
    for GenreX in Genres:
        deep_columns.append(tf.contrib.layers.embedding_column(GenreX, dimension=8, combiner="sqrtn"))
    deep_columns.append(tf.contrib.layers.embedding_column(UserID, dimension=8, combiner="sqrtn"))
    deep_columns.append(tf.contrib.layers.embedding_column(AnimeID, dimension=8, combiner="sqrtn"))
    deep_columns.append(tf.contrib.layers.embedding_column(MediaType, dimension=8, combiner="sqrtn"))
    deep_columns.append(Episodes)
    deep_columns.append(OverallRating)
    deep_columns.append(ListMembership)

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                            feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                        feature_columns=deep_columns,
                                        hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50],
            fix_global_step_increment_bug=True)
    return m


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
    """Train and evaluate the model."""
    
    #df_all = pd.read_csv("file:///C:/Users/jaden/Documents/SYDE%20522/Data%20Set/data_user.csv", names = COLUMNS);
    df_all = pd.read_csv("file:///C:/Users/jaden/Documents/SYDE%20522/Data%20Set/data_user.csv", names = COLUMNS, nrows = 100000);

    split_perc=0.9
    mask = np.random.rand(len(df_all)) < split_perc
    df_train = df_all[mask]
    df_test = df_all[~mask]

    # Split into positive and negative
    df_train[LABEL_COLUMN] = (
        df_train["UserRating"].apply(lambda x: 1 if x >= 7 else 0))
    df_test[LABEL_COLUMN] = (
        df_test["UserRating"].apply(lambda x: 1 if x >= 7 else 0))

    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir, model_type, df_train['UserID'].max(), df_train['AnimeID'].max())
    # Still needs a monitor
    print ("FitBegin:" + str(time.strftime("%H:%M:%S")))
    m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    print ("FitEnd:" + str(time.strftime("%H:%M:%S")))
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
    print ("End:" + str(time.strftime("%H:%M:%S")))
    #freeze_graph(m) #doesn't work

def freeze_graph(model):
    # We precise the file fullname of our freezed graph
    output_graph = "/tmp/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = "InputData/X,FullyConnected/Softmax"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = model.net.graph
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    # We use a built-in TF helper to export variables to constants
    sess = model.session
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, # The session is used to retrieve the weights
        input_graph_def, # The graph_def is used to retrieve the nodes 
        output_node_names.split(",") # The output node names are used to select the usefull nodes
    ) 

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))


FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=200,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)