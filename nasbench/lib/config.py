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

"""Configuration flags."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

FLAGS = flags.FLAGS

# Data flags (only required for generating the dataset)
flags.DEFINE_list(
    'train_data_files', [],
    'Training data files in TFRecord format. Multiple files can be passed in a'
    ' comma-separated list. The first file in the list will be used for'
    ' computing the training error.')
flags.DEFINE_string(
    'valid_data_file', '', 'Validation data in TFRecord format.')
flags.DEFINE_string(
    'test_data_file', '', 'Testing data in TFRecord format.')
flags.DEFINE_string(
    'sample_data_file', '', 'Sampled batch data in TFRecord format.')
flags.DEFINE_string(
    'data_format', 'channels_last',
    'Data format, one of [channels_last, channels_first] for NHWC and NCHW'
    ' tensor formats respectively.')
flags.DEFINE_integer(
    'num_labels', 10, 'Number of input class labels.')

# Search space parameters.
flags.DEFINE_integer(
    'module_vertices', 7,
    'Number of vertices in module matrix, including input and output.')
flags.DEFINE_integer(
    'max_edges', 9,
    'Maximum number of edges in the module matrix.')
flags.DEFINE_list(
    'available_ops', ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'],
    'Available op labels, see base_ops.py for full list of ops.')

# Model hyperparameters. The default values are exactly what is used during the
# exhaustive evaluation of all models.
flags.DEFINE_integer(
    'stem_filter_size', 128, 'Filter size after stem convolutions.')
flags.DEFINE_integer(
    'num_stacks', 3, 'Number of stacks of modules.')
flags.DEFINE_integer(
    'num_modules_per_stack', 3, 'Number of modules per stack.')
flags.DEFINE_integer(
    'batch_size', 256, 'Training batch size.')
flags.DEFINE_integer(
    'train_epochs', 108,
    'Maximum training epochs. If --train_seconds is reached first, training'
    ' may not reach --train_epochs.')
flags.DEFINE_float(
    'train_seconds', 4.0 * 60 * 60,
    'Maximum training seconds. If --train_epochs is reached first, training'
    ' may not reach --train_seconds. Used as safeguard against stalled jobs.'
    ' If train_seconds is 0.0, no time limit will be used.')
flags.DEFINE_float(
    'learning_rate', 0.1,
    'Base learning rate. Linearly scaled by --tpu_num_shards.')
flags.DEFINE_string(
    'lr_decay_method', 'COSINE_BY_STEP',
    '[COSINE_BY_TIME, COSINE_BY_STEP, STEPWISE], see model_builder.py for full'
    ' list of decay methods.')
flags.DEFINE_float(
    'momentum', 0.9, 'Momentum.')
flags.DEFINE_float(
    'weight_decay', 1e-4, 'L2 regularization weight.')
flags.DEFINE_integer(
    'max_attempts', 5,
    'Maximum number of times to try training and evaluating an individual'
    ' before aborting.')
flags.DEFINE_list(
    'intermediate_evaluations', ['0.5'],
    'Intermediate evaluations relative to --train_epochs. For example, to'
    ' evaluate the model at 1/4, 1/2, 3/4 of the total epochs, use [0.25, 0.5,'
    ' 0.75]. An evaluation is always done at the start and end of training.')
flags.DEFINE_integer(
    'num_repeats', 3,
    'Number of repeats evaluated for each model in the space.')

# TPU flags
flags.DEFINE_bool(
    'use_tpu', True, 'Use TPUs for train and evaluation.')
flags.DEFINE_integer(
    'tpu_iterations_per_loop', 100, 'Iterations per loop of TPU execution.')
flags.DEFINE_integer(
    'tpu_num_shards', 2,
    'Number of TPU shards, a single TPU chip has 2 shards.')


def build_config():
  """Build config from flags defined in this module."""
  config = {
      flag.name: flag.value
      for flag in FLAGS.flags_by_module_dict()[__name__]
  }

  return config
