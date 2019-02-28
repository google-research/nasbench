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

"""Runnable example, as shown in the README.md."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from nasbench import api

# Replace this string with the path to the downloaded nasbench.tfrecord before
# executing.
NASBENCH_TFRECORD = '/path/to/nasbench.tfrecord'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'


def main(argv):
  del argv  # Unused

  # Load the data from file (this will take some time)
  nasbench = api.NASBench(NASBENCH_TFRECORD)

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

  # Query this model from dataset, returns a dictionary containing the metrics
  # associated with this model.
  print('Querying an Inception-like model.')
  data = nasbench.query(model_spec)
  print(data)
  print(nasbench.get_budget_counters())   # prints (total time, total epochs)

  # Get all metrics (all epoch lengths, all repeats) associated with this
  # model_spec. This should be used for dataset analysis and NOT for
  # benchmarking algorithms (does not increment budget counters).
  print('\nGetting all metrics for the same Inception-like model.')
  fixed_metrics, computed_metrics = nasbench.get_metrics_from_spec(model_spec)
  print(fixed_metrics)
  for epochs in nasbench.valid_epochs:
    for repeat_index in range(len(computed_metrics[epochs])):
      data_point = computed_metrics[epochs][repeat_index]
      print('Epochs trained %d, repeat number: %d' % (epochs, repeat_index + 1))
      print(data_point)

  # Iterate through unique models in the dataset. Models are unqiuely identified
  # by a hash.
  print('\nIterating over unique models in the dataset.')
  for unique_hash in nasbench.hash_iterator():
    fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(
        unique_hash)
    print(fixed_metrics)

    # For demo purposes, break here instead of iterating through whole set.
    break


# If you are passing command line flags to modify the default config values, you
# must use app.run(main)
if __name__ == '__main__':
  app.run(main)
