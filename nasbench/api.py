# Copyright 2019 The Google Research Authors.
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

Before using this API, download the data files from the links in the README.

Usage:
  # Load the data from file (this will take some time)
  nasbench = api.NASBench('/path/to/nasbench.tfrecord')

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
  data = nasbench.query(model_spec)

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
  - training_time: the total training time in seconds up to this point
  - train_accuracy: training accuracy
  - validation_accuracy: validation_accuracy
  - test_accuracy: testing accuracy

Instead of querying the dataset for a single run of a model, it is also possible
to retrieve all metrics for a given spec, using:

  fixed_stats, computed_stats = nasbench.get_metrics_from_spec(model_spec)

The fixed_stats is a dictionary with the keys:
  - module_adjacency
  - module_operations
  - trainable_parameters

The computed_stats is a dictionary from epoch count to a list of metric
dicts. For example, computed_stats[108][0] contains the metrics for the first
repeat of the provided model trained to 108 epochs. The available keys are:
  - halfway_training_time
  - halfway_train_accuracy
  - halfway_validation_accuracy
  - halfway_test_accuracy
  - final_training_time
  - final_train_accuracy
  - final_validation_accuracy
  - final_test_accuracy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import copy
import json
import os
import random
import time

from nasbench.lib import config
from nasbench.lib import evaluate
from nasbench.lib import model_metrics_pb2
from nasbench.lib import model_spec as _model_spec
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
    self.config = config.build_config()
    random.seed(seed)

    print('Loading dataset from file... This may take a few minutes...')
    start = time.time()

    # Stores the fixed statistics that are independent of evaluation (i.e.,
    # adjacency matrix, operations, and number of parameters).
    # hash --> metric name --> scalar
    self.fixed_statistics = {}

    # Stores the statistics that are computed via training and evaluating the
    # model on CIFAR-10. Statistics are computed for multiple repeats of each
    # model at each max epoch length.
    # hash --> epochs --> repeat index --> metric name --> scalar
    self.computed_statistics = {}

    # Valid queriable epoch lengths. {4, 12, 36, 108} for the full dataset or
    # {108} for the smaller dataset with only the 108 epochs.
    self.valid_epochs = set()

    for serialized_row in tf.compat.v1.python_io.tf_record_iterator(dataset_file):
      # Parse the data from the data file.
      module_hash, epochs, raw_adjacency, raw_operations, raw_metrics = (
          json.loads(serialized_row.decode('utf-8')))

      dim = int(np.sqrt(len(raw_adjacency)))
      adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
      adjacency = np.reshape(adjacency, (dim, dim))
      operations = raw_operations.split(',')
      metrics = model_metrics_pb2.ModelMetrics.FromString(
          base64.b64decode(raw_metrics))

      if module_hash not in self.fixed_statistics:
        # First time seeing this module, initialize fixed statistics.
        new_entry = {}
        new_entry['module_adjacency'] = adjacency
        new_entry['module_operations'] = operations
        new_entry['trainable_parameters'] = metrics.trainable_parameters
        self.fixed_statistics[module_hash] = new_entry
        self.computed_statistics[module_hash] = {}

      self.valid_epochs.add(epochs)

      if epochs not in self.computed_statistics[module_hash]:
        self.computed_statistics[module_hash][epochs] = []

      # Each data_point consists of the metrics recorded from a single
      # train-and-evaluation of a model at a specific epoch length.
      data_point = {}

      # Note: metrics.evaluation_data[0] contains the computed metrics at the
      # start of training (step 0) but this is unused by this API.

      # Evaluation statistics at the half-way point of training
      half_evaluation = metrics.evaluation_data[1]
      data_point['halfway_training_time'] = half_evaluation.training_time
      data_point['halfway_train_accuracy'] = half_evaluation.train_accuracy
      data_point['halfway_validation_accuracy'] = (
          half_evaluation.validation_accuracy)
      data_point['halfway_test_accuracy'] = half_evaluation.test_accuracy

      # Evaluation statistics at the end of training
      final_evaluation = metrics.evaluation_data[2]
      data_point['final_training_time'] = final_evaluation.training_time
      data_point['final_train_accuracy'] = final_evaluation.train_accuracy
      data_point['final_validation_accuracy'] = (
          final_evaluation.validation_accuracy)
      data_point['final_test_accuracy'] = final_evaluation.test_accuracy

      self.computed_statistics[module_hash][epochs].append(data_point)

    elapsed = time.time() - start
    print('Loaded dataset in %d seconds' % elapsed)

    self.history = {}
    self.training_time_spent = 0.0
    self.total_epochs_spent = 0

  def query(self, model_spec, epochs=108, stop_halfway=False):
    """Fetch one of the evaluations for this model spec.

    Each call will sample one of the config['num_repeats'] evaluations of the
    model. This means that repeated queries of the same model (or isomorphic
    models) may return identical metrics.

    This function will increment the budget counters for benchmarking purposes.
    See self.training_time_spent, and self.total_epochs_spent.

    This function also allows querying the evaluation metrics at the halfway
    point of training using stop_halfway. Using this option will increment the
    budget counters only up to the halfway point.

    Args:
      model_spec: ModelSpec object.
      epochs: number of epochs trained. Must be one of the evaluated number of
        epochs, [4, 12, 36, 108] for the full dataset.
      stop_halfway: if True, returned dict will only contain the training time
        and accuracies at the halfway point of training (num_epochs/2).
        Otherwise, returns the time and accuracies at the end of training
        (num_epochs).

    Returns:
      dict containing the evaluated data for this object.

    Raises:
      OutOfDomainError: if model_spec or num_epochs is outside the search space.
    """
    if epochs not in self.valid_epochs:
      raise OutOfDomainError('invalid number of epochs, must be one of %s'
                             % self.valid_epochs)

    fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)
    sampled_index = random.randint(0, self.config['num_repeats'] - 1)
    computed_stat = computed_stat[epochs][sampled_index]

    data = {}
    data['module_adjacency'] = fixed_stat['module_adjacency']
    data['module_operations'] = fixed_stat['module_operations']
    data['trainable_parameters'] = fixed_stat['trainable_parameters']

    if stop_halfway:
      data['training_time'] = computed_stat['halfway_training_time']
      data['train_accuracy'] = computed_stat['halfway_train_accuracy']
      data['validation_accuracy'] = computed_stat['halfway_validation_accuracy']
      data['test_accuracy'] = computed_stat['halfway_test_accuracy']
    else:
      data['training_time'] = computed_stat['final_training_time']
      data['train_accuracy'] = computed_stat['final_train_accuracy']
      data['validation_accuracy'] = computed_stat['final_validation_accuracy']
      data['test_accuracy'] = computed_stat['final_test_accuracy']

    self.training_time_spent += data['training_time']
    if stop_halfway:
      self.total_epochs_spent += epochs // 2
    else:
      self.total_epochs_spent += epochs

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

  def get_budget_counters(self):
    """Returns the time and budget counters."""
    return self.training_time_spent, self.total_epochs_spent

  def reset_budget_counters(self):
    """Reset the time and epoch budget counters."""
    self.training_time_spent = 0.0
    self.total_epochs_spent = 0

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
      json.dump(metadata, f, cls=_NumpyEncoder)

    data_point = {}
    data_point['module_adjacency'] = model_spec.matrix
    data_point['module_operations'] = model_spec.ops
    data_point['trainable_parameters'] = metadata['trainable_params']

    final_evaluation = metadata['evaluation_results'][-1]
    data_point['training_time'] = final_evaluation['training_time']
    data_point['train_accuracy'] = final_evaluation['train_accuracy']
    data_point['validation_accuracy'] = final_evaluation['validation_accuracy']
    data_point['test_accuracy'] = final_evaluation['test_accuracy']

    return data_point

  def hash_iterator(self):
    """Returns iterator over all unique model hashes."""
    return self.fixed_statistics.keys()

  def get_metrics_from_hash(self, module_hash):
    """Returns the metrics for all epochs and all repeats of a hash.

    This method is for dataset analysis and should not be used for benchmarking.
    As such, it does not increment any of the budget counters.

    Args:
      module_hash: MD5 hash, i.e., the values yielded by hash_iterator().

    Returns:
      fixed stats and computed stats of the model spec provided.
    """
    fixed_stat = copy.deepcopy(self.fixed_statistics[module_hash])
    computed_stat = copy.deepcopy(self.computed_statistics[module_hash])
    return fixed_stat, computed_stat

  def get_metrics_from_spec(self, model_spec):
    """Returns the metrics for all epochs and all repeats of a model.

    This method is for dataset analysis and should not be used for benchmarking.
    As such, it does not increment any of the budget counters.

    Args:
      model_spec: ModelSpec object.

    Returns:
      fixed stats and computed stats of the model spec provided.
    """
    self._check_spec(model_spec)
    module_hash = self._hash_spec(model_spec)
    return self.get_metrics_from_hash(module_hash)

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

  def _hash_spec(self, model_spec):
    """Returns the MD5 hash for a provided model_spec."""
    return model_spec.hash_spec(self.config['available_ops'])


class _NumpyEncoder(json.JSONEncoder):
  """Converts numpy objects to JSON-serializable format."""

  def default(self, obj):
    if isinstance(obj, np.ndarray):
      # Matrices converted to nested lists
      return obj.tolist()
    elif isinstance(obj, np.generic):
      # Scalars converted to closest Python type
      return np.asscalar(obj)
    return json.JSONEncoder.default(self, obj)
