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

"""CIFAR-10 data pipeline with preprocessing.

The data is generated via generate_cifar10_tfrecords.py.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

WIDTH = 32
HEIGHT = 32
RGB_MEAN = [125.31, 122.95, 113.87]
RGB_STD = [62.99, 62.09, 66.70]


class CIFARInput(object):
  """Wrapper class for input_fn passed to TPUEstimator."""

  def __init__(self, mode, config):
    """Initializes a CIFARInput object.

    Args:
      mode: one of [train, valid, test, augment, sample]
      config: config dict built from config.py

    Raises:
      ValueError: invalid mode or data files
    """
    self.mode = mode
    self.config = config
    if mode == 'train':         # Training set (no validation & test)
      self.data_files = config['train_data_files']
    elif mode == 'train_eval':  # For computing train error
      self.data_files = [config['train_data_files'][0]]
    elif mode == 'valid':       # For computing validation error
      self.data_files = [config['valid_data_file']]
    elif mode == 'test':        # For computing the test error
      self.data_files = [config['test_data_file']]
    elif mode == 'augment':     # Training set (includes validation, no test)
      self.data_files = (config['train_data_files'] +
                         [config['valid_data_file']])
    elif mode == 'sample':      # Fixed batch of 100 samples from validation
      self.data_files = [config['sample_data_file']]
    else:
      raise ValueError('invalid mode')

    if not self.data_files:
      raise ValueError('no data files provided')

  @property
  def num_images(self):
    """Number of images in the dataset (depends on the mode)."""
    if self.mode == 'train':
      return 40000
    elif self.mode == 'train_eval':
      return 10000
    elif self.mode == 'valid':
      return 10000
    elif self.mode == 'test':
      return 10000
    elif self.mode == 'augment':
      return 50000
    elif self.mode == 'sample':
      return 100

  def input_fn(self, params):
    """Returns a CIFAR tf.data.Dataset object.

    Args:
      params: parameter dict pass by Estimator.

    Returns:
      tf.data.Dataset object
    """
    batch_size = params['batch_size']
    is_training = (self.mode == 'train' or self.mode == 'augment')

    dataset = tf.data.TFRecordDataset(self.data_files)
    dataset = dataset.prefetch(buffer_size=batch_size)

    # Repeat dataset for training modes
    if is_training:
      # Shuffle buffer with whole dataset to ensure full randomness per epoch
      dataset = dataset.cache().apply(
          tf.contrib.data.shuffle_and_repeat(
              buffer_size=self.num_images))

    # This is a hack to allow computing metrics on a fixed batch on TPU. Because
    # TPU shards the batch acrosss cores, we replicate the fixed batch so that
    # each core contains the whole batch.
    if self.mode == 'sample':
      dataset = dataset.repeat()

    # Parse, preprocess, and batch images
    parser_fn = functools.partial(_parser, is_training)
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            parser_fn,
            batch_size=batch_size,
            num_parallel_batches=self.config['tpu_num_shards'],
            drop_remainder=True))

    # Assign static batch size dimension
    dataset = dataset.map(functools.partial(_set_batch_dimension, batch_size))

    # Prefetch to overlap in-feed with training
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def _preprocess(image):
  """Perform standard CIFAR preprocessing.

  Pads the image then performs a random crop.
  Then, image is flipped horizontally randomly.

  Args:
    image: image Tensor with shape [height, width, 3]

  Returns:
    preprocessed image with the same dimensions.
  """
  # Pad 4 pixels on all sides with 0
  image = tf.image.resize_image_with_crop_or_pad(
      image, HEIGHT + 8, WIDTH + 8)

  # Random crop
  image = tf.random_crop(image, [HEIGHT, WIDTH, 3], seed=0)

  # Random flip
  image = tf.image.random_flip_left_right(image, seed=0)

  return image


def _parser(use_preprocessing, serialized_example):
  """Parses a single tf.Example into image and label tensors."""
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })
  image = tf.decode_raw(features['image'], tf.uint8)
  image.set_shape([3 * HEIGHT * WIDTH])
  image = tf.reshape(image, [3, HEIGHT, WIDTH])
  # TODO(chrisying): handle NCHW format
  image = tf.transpose(image, [1, 2, 0])
  image = tf.cast(image, tf.float32)
  if use_preprocessing:
    image = _preprocess(image)
  image -= tf.constant(RGB_MEAN, shape=[1, 1, 3])
  image /= tf.constant(RGB_STD, shape=[1, 1, 3])
  label = tf.cast(features['label'], tf.int32)
  return image, label


def _set_batch_dimension(batch_size, images, labels):
  images.set_shape(images.get_shape().merge_with(
      tf.TensorShape([batch_size, None, None, None])))
  labels.set_shape(labels.get_shape().merge_with(
      tf.TensorShape([batch_size])))

  return images, labels
