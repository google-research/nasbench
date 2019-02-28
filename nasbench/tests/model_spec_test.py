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

"""Tests for lib/model_spec.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nasbench.lib import model_spec
import numpy as np
import tensorflow as tf   # Used only for tf.test


class ModelSpecTest(tf.test.TestCase):

  def test_prune_noop(self):
    """Tests graphs which require no pruning."""
    model1 = model_spec.ModelSpec(
        np.array([[0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0]]),
        [0, 0, 0])
    assert model1.valid_spec
    assert np.array_equal(model1.original_matrix, model1.matrix)
    assert model1.original_ops == model1.original_ops

    model2 = model_spec.ModelSpec(
        np.array([[0, 1, 1],
                  [0, 0, 1],
                  [0, 0, 0]]),
        [0, 0, 0])
    assert model2.valid_spec
    assert np.array_equal(model2.original_matrix, model2.matrix)
    assert model2.original_ops == model2.ops

    model3 = model_spec.ModelSpec(
        np.array([[0, 1, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]]),
        [0, 0, 0, 0])
    assert model3.valid_spec
    assert np.array_equal(model3.original_matrix, model3.matrix)
    assert model3.original_ops == model3.ops

  def test_prune_islands(self):
    """Tests isolated components are pruned."""
    model1 = model_spec.ModelSpec(
        np.array([[0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
        [1, 2, 3, 4])
    assert model1.valid_spec
    assert np.array_equal(model1.matrix,
                          np.array([[0, 1, 0],
                                    [0, 0, 1],
                                    [0, 0, 0]]))
    assert model1.ops == [1, 2, 4]

    model2 = model_spec.ModelSpec(
        np.array([[0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]),
        [1, 2, 3, 4, 5])
    assert model2.valid_spec
    assert np.array_equal(model2.matrix,
                          np.array([[0, 1, 0],
                                    [0, 0, 1],
                                    [0, 0, 0]]))
    assert model2.ops == [1, 2, 5]

  def test_prune_dangling(self):
    """Tests dangling vertices are pruned."""
    model1 = model_spec.ModelSpec(
        np.array([[0, 1, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]]),
        [1, 2, 3, 4])
    assert model1.valid_spec
    assert np.array_equal(model1.matrix,
                          np.array([[0, 1, 0],
                                    [0, 0, 1],
                                    [0, 0, 0]]))
    assert model1.ops == [1, 3, 4]

    model2 = model_spec.ModelSpec(
        np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]]),
        [1, 2, 3, 4])
    assert model2.valid_spec
    assert np.array_equal(model2.matrix,
                          np.array([[0, 1, 0],
                                    [0, 0, 1],
                                    [0, 0, 0]]))
    assert model2.ops == [1, 3, 4]

  def test_prune_disconnected(self):
    """Tests graphs where with no input to output path are marked invalid."""
    model1 = model_spec.ModelSpec(
        np.array([[0, 0],
                  [0, 0]]),
        [0, 0])
    assert not model1.valid_spec

    model2 = model_spec.ModelSpec(
        np.array([[0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]]),
        [1, 2, 3, 4])
    assert not model2.valid_spec

    model3 = model_spec.ModelSpec(
        np.array([[0, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
        [1, 2, 3, 4])
    assert not model3.valid_spec

  def test_is_upper_triangular(self):
    """Tests is_uppper_triangular correct for square graphs."""
    m0 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])
    assert model_spec.is_upper_triangular(m0)

    m1 = np.array([[0, 1, 1, 1],
                   [0, 0, 1, 1],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]])
    assert model_spec.is_upper_triangular(m1)

    m2 = np.array([[0, 1, 1, 1],
                   [0, 0, 1, 1],
                   [1, 0, 0, 1],
                   [0, 0, 0, 0]])
    assert not model_spec.is_upper_triangular(m2)

    m3 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 0]])
    assert not model_spec.is_upper_triangular(m3)

    m4 = np.array([[1, 0, 0, 0],
                   [1, 1, 0, 0],
                   [1, 1, 1, 0],
                   [1, 1, 1, 1]])
    assert not model_spec.is_upper_triangular(m4)

    m5 = np.array([[0]])
    assert model_spec.is_upper_triangular(m5)

    m6 = np.array([[1]])
    assert not model_spec.is_upper_triangular(m6)


if __name__ == '__main__':
  tf.test.main()
