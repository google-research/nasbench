# Copyright 2018 The Google Research Authors.
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

"""User interface for the NAS Benchmark dataset.

Before using this API, download the data at:
  [data not yet available for download]

Usage:
  # Load the data from file (this will take some time)
  dataset = api.NASBench('/path/to/nasbench.tfrecord')

  # Create an Inception-like module (5x5 convolution replaced with two 3x3
  # convolutions).
  model_spec = api.ModelSpec(
      # Adjacency matrix of the module
      matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
              [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
              [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
              [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
              [0, 0, 0, 0, 0, 0, 0]],   # output layer
      # Operations at the vertices of the module, matches order of matrix
      ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])


  # Query this model from dataset
  data = dataset.query(model_spec)

Adjacency matrices are expected to be upper-triangular 0-1 matrices within the
defined search space (7 vertices, 9 edges, 3 allowed ops). The first and last
operations must be 'input' and 'output'. The other operations should be from
config['available_ops']. Currently, the available operations are:
  CONV3X3 = "conv3x3-bn-relu"
  CONV1X1 = "conv1x1-bn-relu"
  MAXPOOL3X3 = "maxpool3x3"

When querying a spec, the spec will first be automatically pruned (removing
unused vertices and edges along with ops). If the pruned spec is still out of
the search space, an OutOfDomainError will be raised, otherwise the data is
returned.

The returned data object is a dictionary with the following keys:
  - module_adjacency: numpy array for the adjacency matrix
  - module_operations: list of operation labels
  - trainable_parameters: number of trainable parameters in the model
  - training_time: the total training time up to this point
  - train_accuracy: training accuracy
  - validation_accuracy: validation_accuracy
  - test_accuracy: testing accuracy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import random
import time

from nasbench.lib import config
from nasbench.lib import evaluate
from nasbench.lib import model_metrics_pb2
from nasbench.lib import model_spec as _model_spec
from nasbench.scripts import run_evaluation
import numpy as np
import tensorflow as tf

# Bring ModelSpec to top-level for convenience. See lib/model_spec.py.
ModelSpec = _model_spec.ModelSpec


class OutOfDomainError(Exception):
  """Indicates that the requested graph is outside of the search domain."""


class NASBench(object):
  """User-facing API for accessing the NASBench dataset."""

  def __init__(self, dataset_file, seed=None):
    """Initialize dataset, this should only be done once per experiment.

    Args:
      dataset_file: path to .tfrecord file containing the dataset.
      seed: random seed used for sampling queried models. Two NASBench objects
        created with the same seed will return the same data points when queried
        with the same models in the same order. By default, the seed is randomly
        generated.
    """
    # TODO(chrisying): download the file directly if dataset_file not provided
    self.config = config.build_config()
    random.seed(seed)

    print('Loading dataset from file... This may take a few minutes...')
    start = time.time()
    self.dataset = {}
    self.valid_epochs = set()
    for serialized_row in tf.python_io.tf_record_iterator(dataset_file):
      data_point = {}

      row = json.loads(serialized_row)
      module_hash = row[0]
      max_epochs = row[1]
      self.valid_epochs.add(max_epochs)
      metrics = model_metrics_pb2.ModelMetrics.FromString(
          row[4].decode('base64'))

      # TODO(chrisying): is it useful to keep the original adjacency and
      # operations actually evaluated? Leaving this commented out because it
      # saves a lot of memory.
      '''
      module_adjacency = row[2]
      module_operations = row[3]
      adjacency = np.array(map(int, list(metrics['module_adjacency'])))
      dim = int(math.sqrt(len(adjacency)))
      assert dim * dim == len(adjacency)
      data_point['module_adjacency'] = adjacency.reshape((dim, dim))
      data_point['module_operations'] = metrics['module_operations'].split(',')
      '''
      data_point['trainable_parameters'] = metrics.trainable_parameters

      half_evaluation = metrics.evaluation_data[1]
      final_evaluation = metrics.evaluation_data[2]
      data_point['training_time'] = (half_evaluation.training_time,
                                     final_evaluation.training_time)
      data_point['train_accuracy'] = (half_evaluation.train_accuracy,
                                      final_evaluation.train_accuracy)
      data_point['validation_accuracy'] = (half_evaluation.validation_accuracy,
                                           final_evaluation.validation_accuracy)
      data_point['test_accuracy'] = (half_evaluation.test_accuracy,
                                     final_evaluation.test_accuracy)

      key = (module_hash, max_epochs)
      if key not in self.dataset:
        self.dataset[key] = []
      self.dataset[key].append(data_point)

    elapsed = time.time() - start
    print('Loaded dataset in %d seconds' % elapsed)

    self.history = {}
    self.training_time_spent = 0
    self.total_epochs_spent = 0
    # TODO(chrisying): add GCS readers for checkpoint data

  def query(self, model_spec, num_epochs=108, stop_halfway=False):
    """Fetch one of the evaluations for this model spec.

    Each call will sample one of the config['num_repeats'] evaluations of the
    model. This means that repeated queries of the same model (or isomorphic
    models) may return identical metrics.

    This function will increment the budget counters for benchmarking purposes.
    See self.training_time_spent, and self.total_epochs_spent.

    This function also allows querying the evaluation metrics at the halfway
    point of training using stop_halfway. Using this option will increment the
    budget counters only up to the halfway point.
    # TODO(chrisying): support "resume" training which gives only increments the
    # budget by the second half of the cost. How should the dataset handle the
    # case where the user queries the same halfway model multiple times?

    Args:
      model_spec: ModelSpec object.
      num_epochs: number of epochs trained. Must be one of the evaluated number
        of epochs, [4, 12, 36, 108] for the full dataset.
      stop_halfway: if True, returned dict will only contain the training time
        and accuracies at the halfway point of training (num_epochs/2).
        Otherwise, returns the time and accuracies at the end of training
        (num_epochs).

    Returns:
      dict containing the evaluated data for this object.

    Raises:
      OutOfDomainError: if model_spec or num_epochs is outside the search space.
    """
    self._check_spec(model_spec)
    if num_epochs not in self.valid_epochs:
      raise OutOfDomainError('invalid number of epochs, must be one of %s'
                             % self.valid_epochs)

    key = (model_spec.hash_spec(self.config['available_ops']), num_epochs)
    sampled_index = random.randint(0, self.config['num_repeats'] - 1)
    data = copy.deepcopy(self.dataset[key][sampled_index])
    if stop_halfway:
      data['training_time'] = data['training_time'][0]
      data['train_accuracy'] = data['train_accuracy'][0]
      data['validation_accuracy'] = data['validation_accuracy'][0]
      data['test_accuracy'] = data['test_accuracy'][0]
    else:
      data['training_time'] = data['training_time'][1]
      data['train_accuracy'] = data['train_accuracy'][1]
      data['validation_accuracy'] = data['validation_accuracy'][1]
      data['test_accuracy'] = data['test_accuracy'][1]

    self.training_time_spent += data['training_time']
    if stop_halfway:
      self.total_epochs_spent += num_epochs // 2
    else:
      self.total_epochs_spent += num_epochs

    return data

  def is_valid(self, model_spec):
    """Checks the validity of the model_spec.

    For the purposes of benchmarking, this does not increment the budget
    counters.

    Args:
      model_spec: ModelSpec object.

    Returns:
      True if model is within space.
    """
    try:
      self._check_spec(model_spec)
    except OutOfDomainError:
      return False

    return True

  def evaluate(self, model_spec, model_dir):
    """Trains and evaluates a model spec from scratch (does not query dataset).

    This function runs the same procedure that was used to generate each
    evaluation in the dataset.  Because we are not querying the generated
    dataset of trained models, there are no limitations on number of vertices,
    edges, operations, or epochs. Note that the results will not exactly match
    the dataset due to randomness. By default, this uses TPUs for evaluation but
    CPU/GPU can be used by setting --use_tpu=false (GPU will require installing
    tensorflow-gpu).

    Args:
      model_spec: ModelSpec object.
      model_dir: directory to store the checkpoints, summaries, and logs.

    Returns:
      dict contained the evaluated data for this object, same structure as
      returned by query().
    """
    # Metadata contains additional metrics that aren't reported normally.
    # However, these are stored in the JSON file at the model_dir.
    metadata = evaluate.train_and_evaluate(model_spec, self.config, model_dir)
    metadata_file = os.path.join(model_dir, 'metadata.json')
    with tf.gfile.Open(metadata_file, 'w') as f:
      json.dump(metadata, f, cls=run_evaluation.NumpyEncoder)

    data_point = {}
    data_point['trainable_parameters'] = metadata['trainable_params']

    final_evaluation = metadata['evaluation_results'][-1]
    data_point['training_time'] = final_evaluation['training_time']
    data_point['train_accuracy'] = final_evaluation['train_accuracy']
    data_point['validation_accuracy'] = final_evaluation['validation_accuracy']
    data_point['test_accuracy'] = final_evaluation['test_accuracy']

    return data_point

  def get_model_metrics(self, model_spec):
    """Returns the computed metrics for all epochs and all repeats of a model.

    This method is for dataset analysis and should not be used for benchmarking.
    As such, it does not increment any of the budget counters.

    Args:
      model_spec: ModelSpec object.

    Returns:
      dict of number of epochs (one of [4, 12, 36, 108]) to a list of data
      points, each data point is a dict of the same format as described in the
      docstring. Note that training_time, train_accuracy, validation_accuracy,
      test_accuracy are tuples, with the first element being the value at the
      halfway point of training and the second at the end.
    """
    self._check_spec(model_spec)
    data = {}
    for num_epochs in self.valid_epochs:
      key = (model_spec.hash_spec(self.config['available_ops']), num_epochs)
      data[num_epochs] = self.dataset[key]

    return data

  def _check_spec(self, model_spec):
    """Checks that the model spec is within the dataset."""
    if not model_spec.valid_spec:
      raise OutOfDomainError('invalid spec, provided graph is disconnected.')

    num_vertices = len(model_spec.ops)
    num_edges = np.sum(model_spec.matrix)

    if num_vertices > self.config['module_vertices']:
      raise OutOfDomainError('too many vertices, got %d (max vertices = %d)'
                             % (num_vertices, config['module_vertices']))

    if num_edges > self.config['max_edges']:
      raise OutOfDomainError('too many edges, got %d (max edges = %d)'
                             % (num_edges, self.config['max_edges']))

    if model_spec.ops[0] != 'input':
      raise OutOfDomainError('first operation should be \'input\'')
    if model_spec.ops[-1] != 'output':
      raise OutOfDomainError('last operation should be \'output\'')
    for op in model_spec.ops[1:-1]:
      if op not in self.config['available_ops']:
        raise OutOfDomainError('unsupported op %s (available ops = %s)'
                               % (op, self.config['available_ops']))

