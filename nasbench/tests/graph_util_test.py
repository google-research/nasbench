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

"""Tests for lib/graph_util.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from nasbench.lib import graph_util
import numpy as np
import tensorflow as tf   # Used for tf.test


class GraphUtilTest(tf.test.TestCase):

  def test_gen_is_edge(self):
    """Tests gen_is_edge generates correct graphs."""
    fn = graph_util.gen_is_edge_fn(0)     # '000'
    arr = np.fromfunction(fn, (3, 3), dtype=np.int8)
    self.assertTrue(np.array_equal(arr,
                                   np.array([[0, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, 0]])))

    fn = graph_util.gen_is_edge_fn(3)     # '011'
    arr = np.fromfunction(fn, (3, 3), dtype=np.int8)
    self.assertTrue(np.array_equal(arr,
                                   np.array([[0, 1, 1],
                                             [0, 0, 0],
                                             [0, 0, 0]])))

    fn = graph_util.gen_is_edge_fn(5)     # '101'
    arr = np.fromfunction(fn, (3, 3), dtype=np.int8)
    self.assertTrue(np.array_equal(arr,
                                   np.array([[0, 1, 0],
                                             [0, 0, 1],
                                             [0, 0, 0]])))

    fn = graph_util.gen_is_edge_fn(7)     # '111'
    arr = np.fromfunction(fn, (3, 3), dtype=np.int8)
    self.assertTrue(np.array_equal(arr,
                                   np.array([[0, 1, 1],
                                             [0, 0, 1],
                                             [0, 0, 0]])))

    fn = graph_util.gen_is_edge_fn(7)     # '111'
    arr = np.fromfunction(fn, (4, 4), dtype=np.int8)
    self.assertTrue(np.array_equal(arr,
                                   np.array([[0, 1, 1, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 0],
                                             [0, 0, 0, 0]])))

    fn = graph_util.gen_is_edge_fn(18)     # '010010'
    arr = np.fromfunction(fn, (4, 4), dtype=np.int8)
    self.assertTrue(np.array_equal(arr,
                                   np.array([[0, 0, 1, 0],
                                             [0, 0, 0, 1],
                                             [0, 0, 0, 0],
                                             [0, 0, 0, 0]])))

    fn = graph_util.gen_is_edge_fn(35)     # '100011'
    arr = np.fromfunction(fn, (4, 4), dtype=np.int8)
    self.assertTrue(np.array_equal(arr,
                                   np.array([[0, 1, 1, 0],
                                             [0, 0, 0, 0],
                                             [0, 0, 0, 1],
                                             [0, 0, 0, 0]])))

  def test_is_full_dag(self):
    """Tests is_full_dag classifies DAGs."""
    self.assertTrue(graph_util.is_full_dag(np.array(
        [[0, 1, 0],
         [0, 0, 1],
         [0, 0, 0]])))

    self.assertTrue(graph_util.is_full_dag(np.array(
        [[0, 1, 1],
         [0, 0, 1],
         [0, 0, 0]])))

    self.assertTrue(graph_util.is_full_dag(np.array(
        [[0, 1, 1, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 0]])))

    # vertex 1 not connected to input
    self.assertFalse(graph_util.is_full_dag(np.array(
        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 0]])))

    # vertex 1 not connected to output
    self.assertFalse(graph_util.is_full_dag(np.array(
        [[0, 1, 1],
         [0, 0, 0],
         [0, 0, 0]])))

    # 1, 3 are connected to each other but disconnected from main path
    self.assertFalse(graph_util.is_full_dag(np.array(
        [[0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]])))

    # no path from input to output
    self.assertFalse(graph_util.is_full_dag(np.array(
        [[0, 0, 1, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]])))

    # completely disconnected vertex
    self.assertFalse(graph_util.is_full_dag(np.array(
        [[0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]])))

  def test_hash_module(self):
    # Diamond graph with label permutation
    matrix1 = np.array(
        [[0, 1, 1, 0,],
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 0]])
    label1 = [-1, 1, 2, -2]
    label2 = [-1, 2, 1, -2]

    hash1 = graph_util.hash_module(matrix1, label1)
    hash2 = graph_util.hash_module(matrix1, label2)
    self.assertEqual(hash1, hash2)

    # Simple graph with edge permutation
    matrix1 = np.array(
        [[0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]])
    label1 = [-1, 1, 2, 3, -2]

    matrix2 = np.array(
        [[0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]])
    label2 = [-1, 2, 3, 1, -2]

    matrix3 = np.array(
        [[0, 1, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]])
    label3 = [-1, 2, 1, 3, -2]

    hash1 = graph_util.hash_module(matrix1, label1)
    hash2 = graph_util.hash_module(matrix2, label2)
    hash3 = graph_util.hash_module(matrix3, label3)
    self.assertEqual(hash1, hash2)
    self.assertEqual(hash2, hash3)

    hash4 = graph_util.hash_module(matrix1, label2)
    self.assertNotEqual(hash4, hash1)

    hash5 = graph_util.hash_module(matrix1, label3)
    self.assertNotEqual(hash5, hash1)

    # Connected non-isomorphic regular graphs on 6 interior vertices (8 total)
    matrix1 = np.array(
        [[0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0]])
    matrix2 = np.array(
        [[0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 1, 0],
         [0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0]])
    label1 = [-1, 1, 1, 1, 1, 1, 1, -2]

    hash1 = graph_util.hash_module(matrix1, label1)
    hash2 = graph_util.hash_module(matrix2, label1)
    self.assertNotEqual(hash1, hash2)

    # Non-isomorphic tricky case (breaks if you don't include self)
    hash1 = graph_util.hash_module(
        np.array([[0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]]),
        [-1, 1, 0, 0, -2])

    hash2 = graph_util.hash_module(
        np.array([[0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]]),
        [-1, 0, 0, 1, -2])
    self.assertNotEqual(hash1, hash2)

    # Non-isomorphic tricky case (breaks if you don't use directed edges)
    hash1 = graph_util.hash_module(
        np.array([[0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]]),
        [-1, 1, 0, -2])

    hash2 = graph_util.hash_module(
        np.array([[0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]]),
        [-1, 0, 1, -2])
    self.assertNotEqual(hash1, hash2)

    # Non-isomorphic tricky case (breaks if you only use out-neighbors and self)
    hash1 = graph_util.hash_module(np.array([[0, 1, 1, 1, 1, 0, 0],
                                             [0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 0]]),
                                   [-1, 1, 0, 0, 0, 0, -2])
    hash2 = graph_util.hash_module(np.array([[0, 1, 1, 1, 1, 0, 0],
                                             [0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 0, 1],
                                             [0, 0, 0, 0, 0, 0, 0]]),
                                   [-1, 0, 0, 0, 1, 0, -2])
    self.assertNotEqual(hash1, hash2)

  def test_permute_graph(self):
    # Does not have to be DAG
    matrix = np.array([[1, 1, 0],
                       [0, 0, 1],
                       [1, 0, 1]])
    labels = ['a', 'b', 'c']

    p1, l1 = graph_util.permute_graph(matrix, labels, [2, 0, 1])
    self.assertTrue(np.array_equal(p1,
                                   np.array([[0, 1, 0],
                                             [0, 1, 1],
                                             [1, 0, 1]])))
    self.assertEqual(l1, ['b', 'c', 'a'])

    p1, l1 = graph_util.permute_graph(matrix, labels, [0, 2, 1])
    self.assertTrue(np.array_equal(p1,
                                   np.array([[1, 0, 1],
                                             [1, 1, 0],
                                             [0, 1, 0]])))
    self.assertEqual(l1, ['a', 'c', 'b'])

  def test_is_isomorphic(self):
    # Reuse some tests from hash_module
    matrix1 = np.array(
        [[0, 1, 1, 0,],
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 0]])
    label1 = [-1, 1, 2, -2]
    label2 = [-1, 2, 1, -2]

    self.assertTrue(graph_util.is_isomorphic((matrix1, label1),
                                             (matrix1, label2)))

    # Simple graph with edge permutation
    matrix1 = np.array(
        [[0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]])
    label1 = [-1, 1, 2, 3, -2]

    matrix2 = np.array(
        [[0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]])
    label2 = [-1, 2, 3, 1, -2]

    matrix3 = np.array(
        [[0, 1, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]])
    label3 = [-1, 2, 1, 3, -2]

    self.assertTrue(graph_util.is_isomorphic((matrix1, label1),
                                             (matrix2, label2)))
    self.assertTrue(graph_util.is_isomorphic((matrix1, label1),
                                             (matrix3, label3)))
    self.assertFalse(graph_util.is_isomorphic((matrix1, label1),
                                              (matrix2, label1)))

    # Connected non-isomorphic regular graphs on 6 interior vertices (8 total)
    matrix1 = np.array(
        [[0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0]])
    matrix2 = np.array(
        [[0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 1, 0],
         [0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0]])
    label1 = [-1, 1, 1, 1, 1, 1, 1, -2]

    self.assertFalse(graph_util.is_isomorphic((matrix1, label1),
                                              (matrix2, label1)))

    # Connected isomorphic regular graphs on 8 total vertices (bipartite)
    matrix1 = np.array(
        [[0, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 1, 1, 0, 1],
         [0, 0, 0, 0, 1, 0, 1, 1],
         [0, 0, 0, 0, 0, 1, 1, 1],
         [1, 1, 1, 0, 0, 0, 0, 0],
         [1, 1, 0, 1, 0, 0, 0, 0],
         [1, 0, 1, 1, 0, 0, 0, 0],
         [0, 1, 1, 1, 0, 0, 0, 0]])
    matrix2 = np.array(
        [[0, 1, 0, 1, 1, 0, 0, 0],
         [1, 0, 1, 0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0, 0, 1, 0],
         [1, 0, 1, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 1, 0, 1],
         [0, 1, 0, 0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0, 1, 0, 1],
         [0, 0, 0, 1, 1, 0, 1, 0]])
    label1 = [1, 1, 1, 1, 1, 1, 1, 1]

    # Sanity check: manual permutation
    perm = [0, 5, 7, 2, 4, 1, 3, 6]
    pm1, pl1 = graph_util.permute_graph(matrix1, label1, perm)
    self.assertTrue(np.array_equal(matrix2, pm1))
    self.assertEqual(pl1, label1)

    self.assertTrue(graph_util.is_isomorphic((matrix1, label1),
                                             (matrix2, label1)))

    label2 = [1, 1, 1, 1, 2, 2, 2, 2]
    label3 = [1, 2, 1, 2, 2, 1, 2, 1]

    self.assertTrue(graph_util.is_isomorphic((matrix1, label2),
                                             (matrix2, label3)))

  def test_random_isomorphism_hashing(self):
    # Tests that hash_module always provides the same hash for randomly
    # generated isomorphic graphs.
    for _ in range(1000):
      # Generate random graph. Note: the algorithm works (i.e. same hash ==
      # isomorphic graphs) for all directed graphs with coloring and does not
      # require the graph to be a DAG.
      size = random.randint(3, 20)
      matrix = np.random.randint(0, 2, [size, size])
      labels = [random.randint(0, 10) for _ in range(size)]

      # Generate permutation of matrix and labels.
      perm = np.random.permutation(size).tolist()
      pmatrix, plabels = graph_util.permute_graph(matrix, labels, perm)

      # Hashes should be identical.
      hash1 = graph_util.hash_module(matrix, labels)
      hash2 = graph_util.hash_module(pmatrix, plabels)
      self.assertEqual(hash1, hash2)

  def test_counterexample_bipartite(self):
    # This is a counter example that shows that the hashing algorithm is not
    # perfectly identifiable (i.e. there are non-isomorphic graphs with the same
    # hash). If this tests fails, it means the algorithm must have been changed
    # in some way that allows it to identify these graphs as non-isomoprhic.
    matrix1 = np.array(
        [[0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    matrix2 = np.array(
        [[0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    labels = [-1, 1, 1, 1, 1, 2, 2, 2, 2, -2]

    # This takes far too long to run so commenting it out. The graphs are
    # non-isomorphic fairly obviously from visual inspection.
    # self.assertFalse(graph_util.is_isomorphic((matrix1, labels),
    #                                           (matrix2, labels)))
    self.assertEqual(graph_util.hash_module(matrix1, labels),
                     graph_util.hash_module(matrix2, labels))


if __name__ == '__main__':
  tf.test.main()
