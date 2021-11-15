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

"""Builds the TensorFlow computational graph.

Tensors flowing into a single vertex are added together for all vertices
except the output, which is concatenated instead. Tensors flowing out of input
are always added.

If interior edge channels don't match, drop the extra channels (channels are
guaranteed non-decreasing). Tensors flowing out of the input as always
projected instead.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nasbench.lib import base_ops
from nasbench.lib import training_time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def build_model_fn(spec, config, num_train_images):
  """Returns a model function for Estimator."""
  if config['data_format'] == 'channels_last':
    channel_axis = 3
  elif config['data_format'] == 'channels_first':
    # Currently this is not well supported
    channel_axis = 1
  else:
    raise ValueError('invalid data_format')

  def model_fn(features, labels, mode, params):
    """Builds the model from the input features."""
    del params  # Unused
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Store auxiliary activations increasing in depth of network. First
    # activation occurs immediately after the stem and the others immediately
    # follow each stack.
    aux_activations = []

    # Initial stem convolution
    with tf.variable_scope('stem'):
      net = base_ops.conv_bn_relu(
          features, 3, config['stem_filter_size'],
          is_training, config['data_format'])
      aux_activations.append(net)

    for stack_num in range(config['num_stacks']):
      channels = net.get_shape()[channel_axis].value

      # Downsample at start (except first)
      if stack_num > 0:
        net = tf.layers.max_pooling2d(
            inputs=net,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            data_format=config['data_format'])

        # Double output channels each time we downsample
        channels *= 2

      with tf.variable_scope('stack{}'.format(stack_num)):
        for module_num in range(config['num_modules_per_stack']):
          with tf.variable_scope('module{}'.format(module_num)):
            net = build_module(
                spec,
                inputs=net,
                channels=channels,
                is_training=is_training)
        aux_activations.append(net)

    # Global average pool
    if config['data_format'] == 'channels_last':
      net = tf.reduce_mean(net, [1, 2])
    elif config['data_format'] == 'channels_first':
      net = tf.reduce_mean(net, [2, 3])
    else:
      raise ValueError('invalid data_format')

    # Fully-connected layer to labels
    logits = tf.layers.dense(
        inputs=net,
        units=config['num_labels'])

    if mode == tf.estimator.ModeKeys.PREDICT and not config['use_tpu']:
      # It is a known limitation of Estimator that the labels
      # are not passed during PREDICT mode when running on CPU/GPU
      # (https://github.com/tensorflow/tensorflow/issues/17824), thus we cannot
      # compute the loss or anything dependent on it (i.e., the gradients).
      loss = tf.constant(0.0)
    else:
      loss = tf.losses.softmax_cross_entropy(
          onehot_labels=tf.one_hot(labels, config['num_labels']),
          logits=logits)

      loss += config['weight_decay'] * tf.add_n(
          [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    # Use inference mode to compute some useful metrics on a fixed sample
    # Due to the batch being sharded on TPU, these metrics should be run on CPU
    # only to ensure that the metrics are computed on the whole batch. We add a
    # leading dimension because PREDICT expects batch-shaped tensors.
    if mode == tf.estimator.ModeKeys.PREDICT:
      parameter_norms = {
          'param:' + tensor.name:
          tf.expand_dims(tf.norm(tensor, ord=2), 0)
          for tensor in tf.trainable_variables()
      }

      # Compute gradients of all parameters and the input simultaneously
      all_params_names = []
      all_params_tensors = []
      for tensor in tf.trainable_variables():
        all_params_names.append('param_grad_norm:' + tensor.name)
        all_params_tensors.append(tensor)
      all_params_names.append('input_grad_norm')
      all_params_tensors.append(features)

      grads = tf.gradients(loss, all_params_tensors)

      param_gradient_norms = {}
      for name, grad in list(zip(all_params_names, grads))[:-1]:
        if grad is not None:
          param_gradient_norms[name] = (
              tf.expand_dims(tf.norm(grad, ord=2), 0))
        else:
          param_gradient_norms[name] = (
              tf.expand_dims(tf.constant(0.0), 0))

      if grads[-1] is not None:
        input_grad_norm = tf.sqrt(tf.reduce_sum(
            tf.square(grads[-1]), axis=[1, 2, 3]))
      else:
        input_grad_norm = tf.expand_dims(tf.constant(0.0), 0)

      covariance_matrices = {
          'cov_matrix_%d' % i:
          tf.expand_dims(_covariance_matrix(aux), 0)
          for i, aux in enumerate(aux_activations)
      }

      predictions = {
          'logits': logits,
          'loss': tf.expand_dims(loss, 0),
          'input_grad_norm': input_grad_norm,
      }
      predictions.update(parameter_norms)
      predictions.update(param_gradient_norms)
      predictions.update(covariance_matrices)

      return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
      global_step = tf.train.get_or_create_global_step()
      base_lr = config['learning_rate']
      if config['use_tpu']:
        base_lr *= config['tpu_num_shards']

      if config['lr_decay_method'] == 'COSINE_BY_STEP':
        total_steps = int(config['train_epochs'] * num_train_images /
                          config['batch_size'])
        progress_fraction = tf.cast(global_step, tf.float32) / total_steps
        learning_rate = (0.5 * base_lr *
                         (1 + tf.cos(np.pi * progress_fraction)))

      elif config['lr_decay_method'] == 'COSINE_BY_TIME':
        # Requires training_time.limit hooks to be added to Estimator
        elapsed_time = tf.cast(training_time.get_total_time(), dtype=tf.float32)
        progress_fraction = elapsed_time / config['train_seconds']
        learning_rate = (0.5 * base_lr *
                         (1 + tf.cos(np.pi * progress_fraction)))

      elif config['lr_decay_method'] == 'STEPWISE':
        # divide LR by 10 at 1/2, 2/3, and 5/6 of total epochs
        total_steps = (config['train_epochs'] * num_train_images /
                       config['batch_size'])
        boundaries = [int(0.5 * total_steps),
                      int(0.667 * total_steps),
                      int(0.833 * total_steps)]
        values = [1.0 * base_lr,
                  0.1 * base_lr,
                  0.01 * base_lr,
                  0.0001 * base_lr]
        learning_rate = tf.train.piecewise_constant(
            global_step, boundaries, values)

      else:
        raise ValueError('invalid lr_decay_method')

      # Set LR to 0 for step 0 to initialize the weights without training
      learning_rate = tf.where(tf.equal(global_step, 0), 0.0, learning_rate)

      optimizer = tf.train.RMSPropOptimizer(
          learning_rate=learning_rate,
          momentum=config['momentum'],
          epsilon=1.0)
      if config['use_tpu']:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

      # Update ops required for batch norm moving variables
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)

      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(labels, logits):
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.metrics.accuracy(labels, predictions)

        return {'accuracy': accuracy}

      eval_metrics = (metric_fn, [labels, logits])

      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=eval_metrics)

  return model_fn


def build_module(spec, inputs, channels, is_training):
  """Build a custom module using a proposed model spec.

  Builds the model using the adjacency matrix and op labels specified. Channels
  controls the module output channel count but the interior channels are
  determined via equally splitting the channel count whenever there is a
  concatenation of Tensors.

  Args:
    spec: ModelSpec object.
    inputs: input Tensors to this module.
    channels: output channel count.
    is_training: bool for whether this model is training.

  Returns:
    output Tensor from built module.

  Raises:
    ValueError: invalid spec
  """
  num_vertices = np.shape(spec.matrix)[0]

  if spec.data_format == 'channels_last':
    channel_axis = 3
  elif spec.data_format == 'channels_first':
    channel_axis = 1
  else:
    raise ValueError('invalid data_format')

  input_channels = inputs.get_shape()[channel_axis].value
  # vertex_channels[i] = number of output channels of vertex i
  vertex_channels = compute_vertex_channels(
      input_channels, channels, spec.matrix)

  # Construct tensors from input forward
  tensors = [tf.identity(inputs, name='input')]

  final_concat_in = []
  for t in range(1, num_vertices - 1):
    with tf.variable_scope('vertex_{}'.format(t)):
      # Create interior connections, truncating if necessary
      add_in = [truncate(tensors[src], vertex_channels[t], spec.data_format)
                for src in range(1, t) if spec.matrix[src, t]]

      # Create add connection from projected input
      if spec.matrix[0, t]:
        add_in.append(projection(
            tensors[0],
            vertex_channels[t],
            is_training,
            spec.data_format))

      if len(add_in) == 1:
        vertex_input = add_in[0]
      else:
        vertex_input = tf.add_n(add_in)

      # Perform op at vertex t
      op = base_ops.OP_MAP[spec.ops[t]](
          is_training=is_training,
          data_format=spec.data_format)
      vertex_value = op.build(vertex_input, vertex_channels[t])

    tensors.append(vertex_value)
    if spec.matrix[t, num_vertices - 1]:
      final_concat_in.append(tensors[t])

  # Construct final output tensor by concating all fan-in and adding input.
  if not final_concat_in:
    # No interior vertices, input directly connected to output
    assert spec.matrix[0, num_vertices - 1]
    with tf.variable_scope('output'):
      outputs = projection(
          tensors[0],
          channels,
          is_training,
          spec.data_format)

  else:
    if len(final_concat_in) == 1:
      outputs = final_concat_in[0]
    else:
      outputs = tf.concat(final_concat_in, channel_axis)

    if spec.matrix[0, num_vertices - 1]:
      outputs += projection(
          tensors[0],
          channels,
          is_training,
          spec.data_format)

  outputs = tf.identity(outputs, name='output')
  return outputs


def projection(inputs, channels, is_training, data_format):
  """1x1 projection (as in ResNet) followed by batch normalization and ReLU."""
  with tf.variable_scope('projection'):
    net = base_ops.conv_bn_relu(inputs, 1, channels, is_training, data_format)

  return net


def truncate(inputs, channels, data_format):
  """Slice the inputs to channels if necessary."""
  if data_format == 'channels_last':
    input_channels = inputs.get_shape()[3].value
  else:
    assert data_format == 'channels_first'
    input_channels = inputs.get_shape()[1].value

  if input_channels < channels:
    raise ValueError('input channel < output channels for truncate')
  elif input_channels == channels:
    return inputs   # No truncation necessary
  else:
    # Truncation should only be necessary when channel division leads to
    # vertices with +1 channels. The input vertex should always be projected to
    # the minimum channel count.
    assert input_channels - channels == 1
    if data_format == 'channels_last':
      return tf.slice(inputs, [0, 0, 0, 0], [-1, -1, -1, channels])
    else:
      return tf.slice(inputs, [0, 0, 0, 0], [-1, channels, -1, -1])


def compute_vertex_channels(input_channels, output_channels, matrix):
  """Computes the number of channels at every vertex.

  Given the input channels and output channels, this calculates the number of
  channels at each interior vertex. Interior vertices have the same number of
  channels as the max of the channels of the vertices it feeds into. The output
  channels are divided amongst the vertices that are directly connected to it.
  When the division is not even, some vertices may receive an extra channel to
  compensate.

  Args:
    input_channels: input channel count.
    output_channels: output channel count.
    matrix: adjacency matrix for the module (pruned by model_spec).

  Returns:
    list of channel counts, in order of the vertices.
  """
  num_vertices = np.shape(matrix)[0]

  vertex_channels = [0] * num_vertices
  vertex_channels[0] = input_channels
  vertex_channels[num_vertices - 1] = output_channels

  if num_vertices == 2:
    # Edge case where module only has input and output vertices
    return vertex_channels

  # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
  # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
  in_degree = np.sum(matrix[1:], axis=0)
  interior_channels = output_channels // in_degree[num_vertices - 1]
  correction = output_channels % in_degree[num_vertices - 1]  # Remainder to add

  # Set channels of vertices that flow directly to output
  for v in range(1, num_vertices - 1):
    if matrix[v, num_vertices - 1]:
      vertex_channels[v] = interior_channels
      if correction:
        vertex_channels[v] += 1
        correction -= 1

  # Set channels for all other vertices to the max of the out edges, going
  # backwards. (num_vertices - 2) index skipped because it only connects to
  # output.
  for v in range(num_vertices - 3, 0, -1):
    if not matrix[v, num_vertices - 1]:
      for dst in range(v + 1, num_vertices - 1):
        if matrix[v, dst]:
          vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
    assert vertex_channels[v] > 0

  tf.logging.info('vertex_channels: %s', str(vertex_channels))

  # Sanity check, verify that channels never increase and final channels add up.
  final_fan_in = 0
  for v in range(1, num_vertices - 1):
    if matrix[v, num_vertices - 1]:
      final_fan_in += vertex_channels[v]
    for dst in range(v + 1, num_vertices - 1):
      if matrix[v, dst]:
        assert vertex_channels[v] >= vertex_channels[dst]
  assert final_fan_in == output_channels or num_vertices == 2
  # num_vertices == 2 means only input/output nodes, so 0 fan-in

  return vertex_channels


def _covariance_matrix(activations):
  """Computes the unbiased covariance matrix of the samples within the batch.

  Computes the sample covariance between the samples in the batch. Specifically,

    C(i,j) = (x_i - mean(x_i)) dot (x_j - mean(x_j)) / (N - 1)

  Matches the default behavior of np.cov().

  Args:
    activations: tensor activations with batch dimension first.

  Returns:
    [batch, batch] shape tensor for the covariance matrix.
  """
  batch_size = activations.get_shape()[0].value
  flattened = tf.reshape(activations, [batch_size, -1])
  means = tf.reduce_mean(flattened, axis=1, keepdims=True)

  centered = flattened - means
  squared = tf.matmul(centered, tf.transpose(centered))
  cov = squared / (tf.cast(tf.shape(flattened)[1], tf.float32) - 1)

  return cov

