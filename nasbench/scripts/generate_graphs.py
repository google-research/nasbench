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

"""Generate all graphs up to structure and label isomorphism.

The goal is to generate all unique computational graphs up to some number of
vertices and edges. Computational graphs can be represented by directed acyclic
graphs with all components connected along some path from a specially-labeled
input to output. The pseudocode for generating these is:

for V in [2, ..., MAX_VERTICES]:    # V includes input and output vertices
  generate all bitmasks of length V*(V-1)/2   # num upper triangular entries
  for each bitmask:
    convert bitmask to adjacency matrix
    if adjacency matrix has disconnected vertices from input/output:
      discard and continue to next matrix
    generate all labelings of ops to vertices
    for each labeling:
      compute graph hash from matrix and labels
      if graph hash has not been seen before:
        output graph (adjacency matrix + labeling)

This script uses a modification on Weisfeiler-Lehman color refinement
(https://ist.ac.at/mfcs13/slides/gi.pdf) for graph hashing, which is very
loosely similar to the hashing approach described in
https://arxiv.org/pdf/1606.00001.pdf. The general idea is to assign each vertex
a hash based on the in-degree, out-degree, and operation label then iteratively
hash each vertex with the hashes of its neighbors.

In more detail, the iterative update involves repeating the following steps a
number of times greater than or equal to the diameter of the graph:
  1) For each vertex, sort the hashes of the in-neighbors.
  2) For each vertex, sort the hashes of the out-neighbors.
  3) For each vertex, concatenate the sorted hashes from (1), (2) and the vertex
     operation label.
  4) For each vertex, compute the MD5 hash of the concatenated values in (3).
  5) Assign the newly computed hashes to each vertex.

Finally, sort the hashes of all the vertices and concat and hash one more time
to obtain the final graph hash. This hash is a graph invariant as all operations
are invariant under isomorphism, thus we expect no false negatives (isomorphic
graphs hashed to different values).

We have empirically verified that, for graphs up to 7 vertices, 9 edges, 3 ops,
this algorithm does not cause "false positives" (graphs that hash to the same
value but are non-isomorphic). For such graphs, this algorithm yields 423,624
unique computation graphs, which is roughly 1/3rd of the total number of
connected DAGs before de-duping using this hash algorithm.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import sys

from absl import app
from absl import flags
from absl import logging

from nasbench.lib import graph_util
import numpy as np
import tensorflow as tf   # For gfile

flags.DEFINE_string('output_file', '/tmp/generated_graphs.json',
                    'Output file name.')
flags.DEFINE_integer('max_vertices', 7,
                     'Maximum number of vertices including input/output.')
flags.DEFINE_integer('num_ops', 3, 'Number of operation labels.')
flags.DEFINE_integer('max_edges', 9, 'Maximum number of edges.')
flags.DEFINE_boolean('verify_isomorphism', True,
                     'Exhaustively verifies that each detected isomorphism'
                     ' is truly an isomorphism. This operation is very'
                     ' expensive.')
FLAGS = flags.FLAGS


def main(_):
  total_graphs = 0    # Total number of graphs (including isomorphisms)
  # hash --> (matrix, label) for the canonical graph associated with each hash
  buckets = {}

  logging.info('Using %d vertices, %d op labels, max %d edges',
               FLAGS.max_vertices, FLAGS.num_ops, FLAGS.max_edges)
  for vertices in range(2, FLAGS.max_vertices+1):
    for bits in range(2 ** (vertices * (vertices-1) // 2)):
      # Construct adj matrix from bit string
      matrix = np.fromfunction(graph_util.gen_is_edge_fn(bits),
                               (vertices, vertices),
                               dtype=np.int8)

      # Discard any graphs which can be pruned or exceed constraints
      if (not graph_util.is_full_dag(matrix) or
          graph_util.num_edges(matrix) > FLAGS.max_edges):
        continue

      # Iterate through all possible labelings
      for labeling in itertools.product(*[range(FLAGS.num_ops)
                                          for _ in range(vertices-2)]):
        total_graphs += 1
        labeling = [-1] + list(labeling) + [-2]
        fingerprint = graph_util.hash_module(matrix, labeling)

        if fingerprint not in buckets:
          buckets[fingerprint] = (matrix.tolist(), labeling)

        # This catches the "false positive" case of two models which are not
        # isomorphic hashing to the same bucket.
        elif FLAGS.verify_isomorphism:
          canonical_graph = buckets[fingerprint]
          if not graph_util.is_isomorphic(
              (matrix.tolist(), labeling), canonical_graph):
            logging.fatal('Matrix:\n%s\nLabel: %s\nis not isomorphic to'
                          ' canonical matrix:\n%s\nLabel: %s',
                          str(matrix), str(labeling),
                          str(canonical_graph[0]),
                          str(canonical_graph[1]))
            sys.exit()

    logging.info('Up to %d vertices: %d graphs (%d without hashing)',
                 vertices, len(buckets), total_graphs)

  with tf.gfile.Open(FLAGS.output_file, 'w') as f:
    json.dump(buckets, f, sort_keys=True)


if __name__ == '__main__':
  app.run(main)
