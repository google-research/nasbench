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

"""Base operations used by the modules in this search space."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf

# Currently, only channels_last is well supported.
VALID_DATA_FORMATS = frozenset(['channels_last', 'channels_first'])
MIN_FILTERS = 8
BN_MOMENTUM = 0.997
BN_EPSILON = 1e-5


def conv_bn_relu(inputs, conv_size, conv_filters, is_training, data_format):
  """Convolution followed by batch norm and ReLU."""
  if data_format == 'channels_last':
    axis = 3
  elif data_format == 'channels_first':
    axis = 1
  else:
    raise ValueError('invalid data_format')

  net = tf.layers.conv2d(
      inputs=inputs,
      filters=conv_filters,
      kernel_size=conv_size,
      strides=(1, 1),
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      padding='same',
      data_format=data_format)

  net = tf.layers.batch_normalization(
      inputs=net,
      axis=axis,
      momentum=BN_MOMENTUM,
      epsilon=BN_EPSILON,
      training=is_training)

  net = tf.nn.relu(net)

  return net


class BaseOp(object):
  """Abstract base operation class."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, is_training, data_format='channels_last'):
    self.is_training = is_training
    if data_format.lower() not in VALID_DATA_FORMATS:
      raise ValueError('invalid data_format')
    self.data_format = data_format.lower()

  @abc.abstractmethod
  def build(self, inputs, channels):
    """Builds the operation with input tensors and returns an output tensor.

    Args:
      inputs: a 4-D Tensor.
      channels: int number of output channels of operation. The operation may
        choose to ignore this parameter.

    Returns:
      a 4-D Tensor with the same data format.
    """
    pass


class Identity(BaseOp):
  """Identity operation (ignores channels)."""

  def build(self, inputs, channels):
    del channels    # Unused
    return tf.identity(inputs, name='identity')


class Conv3x3BnRelu(BaseOp):
  """3x3 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    with tf.variable_scope('Conv3x3-BN-ReLU'):
      net = conv_bn_relu(
          inputs, 3, channels, self.is_training, self.data_format)

    return net


class Conv1x1BnRelu(BaseOp):
  """1x1 convolution with batch norm and ReLU activation."""

  def build(self, inputs, channels):
    with tf.variable_scope('Conv1x1-BN-ReLU'):
      net = conv_bn_relu(
          inputs, 1, channels, self.is_training, self.data_format)

    return net


class MaxPool3x3(BaseOp):
  """3x3 max pool with no subsampling."""

  def build(self, inputs, channels):
    del channels    # Unused
    with tf.variable_scope('MaxPool3x3'):
      net = tf.layers.max_pooling2d(
          inputs=inputs,
          pool_size=(3, 3),
          strides=(1, 1),
          padding='same',
          data_format=self.data_format)

    return net


class BottleneckConv3x3(BaseOp):
  """[1x1(/4)]+3x3+[1x1(*4)] conv. Uses BN + ReLU post-activation."""
  # TODO(chrisying): verify this block can reproduce results of ResNet-50.

  def build(self, inputs, channels):
    with tf.variable_scope('BottleneckConv3x3'):
      net = conv_bn_relu(
          inputs, 1, channels // 4, self.is_training, self.data_format)
      net = conv_bn_relu(
          net, 3, channels // 4, self.is_training, self.data_format)
      net = conv_bn_relu(
          net, 1, channels, self.is_training, self.data_format)

    return net


class BottleneckConv5x5(BaseOp):
  """[1x1(/4)]+5x5+[1x1(*4)] conv. Uses BN + ReLU post-activation."""

  def build(self, inputs, channels):
    with tf.variable_scope('BottleneckConv5x5'):
      net = conv_bn_relu(
          inputs, 1, channels // 4, self.is_training, self.data_format)
      net = conv_bn_relu(
          net, 5, channels // 4, self.is_training, self.data_format)
      net = conv_bn_relu(
          net, 1, channels, self.is_training, self.data_format)

    return net


class MaxPool3x3Conv1x1(BaseOp):
  """3x3 max pool with no subsampling followed by 1x1 for rescaling."""

  def build(self, inputs, channels):
    with tf.variable_scope('MaxPool3x3-Conv1x1'):
      net = tf.layers.max_pooling2d(
          inputs=inputs,
          pool_size=(3, 3),
          strides=(1, 1),
          padding='same',
          data_format=self.data_format)

      net = conv_bn_relu(net, 1, channels, self.is_training, self.data_format)

    return net


# Commas should not be used in op names
OP_MAP = {
    'identity': Identity,
    'conv3x3-bn-relu': Conv3x3BnRelu,
    'conv1x1-bn-relu': Conv1x1BnRelu,
    'maxpool3x3': MaxPool3x3,
    'bottleneck3x3': BottleneckConv3x3,
    'bottleneck5x5': BottleneckConv5x5,
    'maxpool3x3-conv1x1': MaxPool3x3Conv1x1,
}
