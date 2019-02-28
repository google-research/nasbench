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

"""Read CIFAR-10 data from pickled numpy arrays and writes TFRecords.

Generates tf.train.Example protos and writes them to TFRecord files from the
python version of the CIFAR-10 dataset downloaded from
https://www.cs.toronto.edu/~kriz/cifar.html.

Based on script from
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py

To run:
  python generate_cifar10_tfrecords.py --data_dir=/tmp/cifar-tfrecord
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tarfile
from six.moves import cPickle as pickle
import tensorflow as tf

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'


def download_and_extract(data_dir):
  # download CIFAR-10 if not already downloaded.
  tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir,
                                                CIFAR_DOWNLOAD_URL)
  tarfile.open(os.path.join(data_dir, CIFAR_FILENAME),
               'r:gz').extractall(data_dir)


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_file_names():
  """Returns the file names expected to exist in the input_dir."""
  file_names = {}
  for i in range(1, 5):
    file_names['train_%d' % i] = 'data_batch_%d' % i
  file_names['validation'] = 'data_batch_5'
  file_names['test'] = 'test_batch'
  return file_names


def read_pickle_from_file(filename):
  with tf.gfile.Open(filename, 'rb') as f:
    if sys.version_info >= (3, 0):
      data_dict = pickle.load(f, encoding='bytes')
    else:
      data_dict = pickle.load(f)
  return data_dict


def convert_to_tfrecord(input_file, output_file):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    data_dict = read_pickle_from_file(input_file)
    data = data_dict[b'data']
    labels = data_dict[b'labels']
    num_entries_in_batch = len(labels)
    print('Converting %d images' % num_entries_in_batch)
    for i in range(num_entries_in_batch):
      example = tf.train.Example(features=tf.train.Features(
          feature={
              'image': _bytes_feature(data[i].tobytes()),
              'label': _int64_feature(labels[i])
          }))
      record_writer.write(example.SerializeToString())


def main(data_dir):
  print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
  download_and_extract(data_dir)
  file_names = _get_file_names()
  input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
  for mode, f in file_names.items():
    input_file = os.path.join(input_dir, f)
    output_file = os.path.join(data_dir, mode + '.tfrecords')
    try:
      os.remove(output_file)
    except OSError:
      pass
    # Convert to tf.train.Example and write the to TFRecords.
    convert_to_tfrecord(input_file, output_file)

  # Save fixed batch of 100 examples (first 10 of each class sampled at the
  # front of the validation set). Ordered by label, i.e. 10 "airplane" images
  # followed by 10 "automobile" images...
  images = [[] for _ in range(10)]
  num_images = 0
  input_file = os.path.join(input_dir, file_names['validation'])
  data_dict = read_pickle_from_file(input_file)
  data = data_dict[b'data']
  labels = data_dict[b'labels']
  for i in range(len(labels)):
    label = labels[i]
    if len(images[label]) < 10:
      images[label].append(
          tf.train.Example(features=tf.train.Features(
              feature={
                  'image': _bytes_feature(data[i].tobytes()),
                  'label': _int64_feature(label)
              })))
      num_images += 1
      if num_images == 100:
        break

  output_file = os.path.join(data_dir, 'sample.tfrecords')
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for label_images in images:
      for example in label_images:
        record_writer.write(example.SerializeToString())
  print('Done!')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='',
      help='Directory to download and extract CIFAR-10 to.')

  args = parser.parse_args()
  main(args.data_dir)
