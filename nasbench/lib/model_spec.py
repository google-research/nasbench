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

"""Model specification for module connectivity individuals.

This module handles pruning the unused parts of the computation graph but should
avoid creating any TensorFlow models (this is done inside model_builder.py).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from nasbench.lib import graph_util
import numpy as np

# Graphviz is optional and only required for visualization.
try:
  import graphviz   # pylint: disable=g-import-not-at-top
except ImportError:
  pass


class ModelSpec(object):
  """Model specification given adjacency matrix and labeling."""

  def __init__(self, matrix, ops, data_format='channels_last'):
    """Initialize the module spec.

    Args:
      matrix: ndarray or nested list with shape [V, V] for the adjacency matrix.
      ops: V-length list of labels for the base ops used. The first and last
        elements are ignored because they are the input and output vertices
        which have no operations. The elements are retained to keep consistent
        indexing.
      data_format: channels_last or channels_first.

    Raises:
      ValueError: invalid matrix or ops
    """
    if not isinstance(matrix, np.ndarray):
      matrix = np.array(matrix)
    shape = np.shape(matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
      raise ValueError('matrix must be square')
    if shape[0] != len(ops):
      raise ValueError('length of ops must match matrix dimensions')
    if not is_upper_triangular(matrix):
      raise ValueError('matrix must be upper triangular')

    # Both the original and pruned matrices are deep copies of the matrix and
    # ops so any changes to those after initialization are not recognized by the
    # spec.
    self.original_matrix = copy.deepcopy(matrix)
    self.original_ops = copy.deepcopy(ops)

    self.matrix = copy.deepcopy(matrix)
    self.ops = copy.deepcopy(ops)
    self.valid_spec = True
    self._prune()

    self.data_format = data_format

  def _prune(self):
    """Prune the extraneous parts of the graph.

    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
    num_vertices = np.shape(self.original_matrix)[0]

    # DFS forward from input
    visited_from_input = set([0])
    frontier = [0]
    while frontier:
      top = frontier.pop()
      for v in range(top + 1, num_vertices):
        if self.original_matrix[top, v] and v not in visited_from_input:
          visited_from_input.add(v)
          frontier.append(v)

    # DFS backward from output
    visited_from_output = set([num_vertices - 1])
    frontier = [num_vertices - 1]
    while frontier:
      top = frontier.pop()
      for v in range(0, top):
        if self.original_matrix[v, top] and v not in visited_from_output:
          visited_from_output.add(v)
          frontier.append(v)

    # Any vertex that isn't connected to both input and output is extraneous to
    # the computation graph.
    extraneous = set(range(num_vertices)).difference(
        visited_from_input.intersection(visited_from_output))

    # If the non-extraneous graph is less than 2 vertices, the input is not
    # connected to the output and the spec is invalid.
    if len(extraneous) > num_vertices - 2:
      self.matrix = None
      self.ops = None
      self.valid_spec = False
      return

    self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
    self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
    for index in sorted(extraneous, reverse=True):
      del self.ops[index]

  def hash_spec(self, canonical_ops):
    """Computes the isomorphism-invariant graph hash of this spec.

    Args:
      canonical_ops: list of operations in the canonical ordering which they
        were assigned (i.e. the order provided in the config['available_ops']).

    Returns:
      MD5 hash of this spec which can be used to query the dataset.
    """
    # Invert the operations back to integer label indices used in graph gen.
    labeling = [-1] + [canonical_ops.index(op) for op in self.ops[1:-1]] + [-2]
    return graph_util.hash_module(self.matrix, labeling)

  def visualize(self):
    """Creates a dot graph. Can be visualized in colab directly."""
    num_vertices = np.shape(self.matrix)[0]
    g = graphviz.Digraph()
    g.node(str(0), 'input')
    for v in range(1, num_vertices - 1):
      g.node(str(v), self.ops[v])
    g.node(str(num_vertices - 1), 'output')

    for src in range(num_vertices - 1):
      for dst in range(src + 1, num_vertices):
        if self.matrix[src, dst]:
          g.edge(str(src), str(dst))

    return g


def is_upper_triangular(matrix):
  """True if matrix is 0 on diagonal and below."""
  for src in range(np.shape(matrix)[0]):
    for dst in range(0, src + 1):
      if matrix[src, dst] != 0:
        return False

  return True
