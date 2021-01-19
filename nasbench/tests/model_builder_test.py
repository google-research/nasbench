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

"""Tests for lib/model_builder.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nasbench.lib import model_builder
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class ModelBuilderTest(tf.test.TestCase):

  def test_compute_vertex_channels_linear(self):
    """Tests modules with no branching."""
    matrix1 = np.array([[0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0]])
    vc1 = model_builder.compute_vertex_channels(8, 8, matrix1)
    assert vc1 == [8, 8, 8, 8]

    vc2 = model_builder.compute_vertex_channels(8, 16, matrix1)
    assert vc2 == [8, 16, 16, 16]

    vc3 = model_builder.compute_vertex_channels(16, 8, matrix1)
    assert vc3 == [16, 8, 8, 8]

    matrix2 = np.array([[0, 1],
                        [0, 0]])
    vc4 = model_builder.compute_vertex_channels(1, 1, matrix2)
    assert vc4 == [1, 1]

    vc5 = model_builder.compute_vertex_channels(1, 5, matrix2)
    assert vc5 == [1, 5]

    vc5 = model_builder.compute_vertex_channels(5, 1, matrix2)
    assert vc5 == [5, 1]

  def test_compute_vertex_channels_no_output_branch(self):
    """Tests modules that branch but not at the output vertex."""
    matrix1 = np.array([[0, 1, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0]])
    vc1 = model_builder.compute_vertex_channels(8, 8, matrix1)
    assert vc1 == [8, 8, 8, 8, 8]

    vc2 = model_builder.compute_vertex_channels(8, 16, matrix1)
    assert vc2 == [8, 16, 16, 16, 16]

    vc3 = model_builder.compute_vertex_channels(16, 8, matrix1)
    assert vc3 == [16, 8, 8, 8, 8]

  def test_compute_vertex_channels_output_branching(self):
    """Tests modules that branch at output."""
    matrix1 = np.array([[0, 1, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0]])
    vc1 = model_builder.compute_vertex_channels(8, 8, matrix1)
    assert vc1 == [8, 4, 4, 8]

    vc2 = model_builder.compute_vertex_channels(8, 16, matrix1)
    assert vc2 == [8, 8, 8, 16]

    vc3 = model_builder.compute_vertex_channels(16, 8, matrix1)
    assert vc3 == [16, 4, 4, 8]

    vc4 = model_builder.compute_vertex_channels(8, 15, matrix1)
    assert vc4 == [8, 8, 7, 15]

    matrix2 = np.array([[0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0]])
    vc5 = model_builder.compute_vertex_channels(8, 8, matrix2)
    assert vc5 == [8, 3, 3, 2, 8]

    vc6 = model_builder.compute_vertex_channels(8, 15, matrix2)
    assert vc6 == [8, 5, 5, 5, 15]

  def test_compute_vertex_channels_max(self):
    """Tests modules where some vertices take the max channels of neighbors."""
    matrix1 = np.array([[0, 1, 0, 0, 0],
                        [0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0]])
    vc1 = model_builder.compute_vertex_channels(8, 8, matrix1)
    assert vc1 == [8, 4, 4, 4, 8]

    vc2 = model_builder.compute_vertex_channels(8, 9, matrix1)
    assert vc2 == [8, 5, 5, 4, 9]

    matrix2 = np.array([[0, 1, 0, 1, 0],
                        [0, 0, 1, 0, 1],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0]])

    vc3 = model_builder.compute_vertex_channels(8, 8, matrix2)
    assert vc3 == [8, 4, 4, 4, 8]

    vc4 = model_builder.compute_vertex_channels(8, 15, matrix2)
    assert vc4 == [8, 8, 7, 7, 15]

  def test_covariance_matrix_against_numpy(self):
    """Tests that the TF implementation of covariance matrix matchs np.cov."""

    # Randomized test 100 times
    for _ in range(100):
      batch = np.random.randint(50, 150)
      features = np.random.randint(500, 1500)
      matrix = np.random.random((batch, features))

      tf_matrix = tf.constant(matrix, dtype=tf.float32)
      tf_cov_tensor = model_builder._covariance_matrix(tf_matrix)

      with tf.Session() as sess:
        tf_cov = sess.run(tf_cov_tensor)

      np_cov = np.cov(matrix)
      np.testing.assert_array_almost_equal(tf_cov, np_cov)


if __name__ == '__main__':
  tf.test.main()
