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
  - total_time: total time for training and evaluation
  - trainable_parameters: number of trainable parameters in the model
  - training_time: the total training time up to this point
  - train_accuracy: training accuracy
  - validation_accuracy: validation_accuracy
  - test_accuracy: testing accuracy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
import time

from nasbench.lib import config
from nasbench.lib import model_metrics_pb2
from nasbench.lib import model_spec as _model_spec
import numpy as np
import tensorflow as tf

VALID_EPOCHS = frozenset([108])

# Bring ModelSpec to top-level for convenience. See lib/model_spec.py.
ModelSpec = _model_spec.ModelSpec


class OutOfDomainError(Exception):
  """Indicates that the requested graph is outside of the search domain."""


class NASBench(object):
  """User-facing API for accessing the NASBench dataset."""

  def __init__(self, dataset_file):
    """Initialize dataset, this should only be done once."""
    # TODO(chrisying): download the file directly if dataset_file not provided
    self.config = config.build_config()

    print('Loading dataset from file... This may take a few minutes...')
    start = time.time()
    self.dataset = {}
    for serialized_row in tf.python_io.tf_record_iterator(dataset_file):
      data_point = {}

      row = json.loads(serialized_row)
      module_hash = row[0]
      metrics = model_metrics_pb2.ModelMetrics.FromString(
          row[3].decode('base64'))

      # TODO(chrisying): is it useful to keep the original adjacency and
      # operations actually evaluated? Leaving this commented out because it
      # saves a lot of memory.
      '''
      module_adjacency = row[1]
      module_operations = row[2]
      adjacency = np.array(map(int, list(metrics['module_adjacency'])))
      dim = int(math.sqrt(len(adjacency)))
      assert dim * dim == len(adjacency)
      data_point['module_adjacency'] = adjacency.reshape((dim, dim))
      data_point['module_operations'] = metrics['module_operations'].split(',')
      '''

      data_point['trainable_parameters'] = metrics.trainable_parameters

      final_evaluation = metrics.evaluation_data[-1]
      data_point['training_time'] = final_evaluation.training_time
      data_point['train_accuracy'] = final_evaluation.train_accuracy
      data_point['validation_accuracy'] = final_evaluation.validation_accuracy
      data_point['test_accuracy'] = final_evaluation.test_accuracy

      # TODO(chrisying): when we support multiple epoch lengths, dict key will
      # include the epoch count as well
      if module_hash not in self.dataset:
        self.dataset[module_hash] = []
      self.dataset[module_hash].append(data_point)

    elapsed = time.time() - start
    print('Loaded dataset in %d seconds' % elapsed)

    self.history = {}
    self.total_time_spent = 0
    self.total_epochs_spent = 0
    # TODO(chrisying): add GCS readers for checkpoint data

  # TODO(chrisying): support additional num_epochs when the data is available.
  def query(self, model_spec, num_epochs=108):
    """Fetch one of the evaluations for this model spec.

    Each call will sample one of the config['num_repeats'] evaluations of the
    model. This means that repeated queries of the same model (or isomorphic
    models) may return identical metrics.

    This function will increment the budget counters for benchmarking purposes.
    See self.total_time_spent and self.total_epochs_spent.

    Args:
      model_spec: ModelSpec object
      num_epochs: number of epochs trained. Must be one of [108].

    Returns:
      dict containing the evaluated data for this object.

    Raises:
      OutOfDomainError: if model_spec or num_epochs is outside the search space.
    """
    self._check_spec(model_spec)
    if num_epochs not in VALID_EPOCHS:
      raise OutOfDomainError('invalid number of epochs, must be one of %s'
                             % VALID_EPOCHS)

    key = model_spec.hash_spec(self.config['available_ops'])
    sampled_index = random.randint(0, self.config['num_repeats'] - 1)
    data = self.dataset[key][sampled_index]

    self.total_time_spent += data['training_time']
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
