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

"""Script for training a large number of networks across multiple workers.

Every process running this script is assigned a monotonically increasing
worker_id starting at 0 to the total_workers (exclusive). Each full
training-and-evaluation of a module counts as a single "work unit" with repeated
runs being different units. Work units are assigned a monotonically increasing
index and each worker only computes the work units that have the same index
modulo total_workers as the worker_id.

For example, for 3 models, each with 3 repeats, and 4 total workers:
Model Number     ||  1  2  3  1  2  3  1  2  3
Repeat Number    ||  1  1  1  2  2  2  3  3  3
Work Unit Index  ||  0  1  2  3  4  5  6  7  8
Assigned Worker  ||  0  1  2  3  0  1  2  3  1

i.e. worker_id 0 will compute [model1-repeat1, model2-repeat2, model3-repeat3],
worker_id 1 will compute [model2-repeat1, model3-repeat2], etc...

--worker_id_offset is provided to allow launching workers in multiple flocks and
is added to the --worker_id flag which is assumed to start at 0 for each new
flock. --total_workers should be the total number of workers across all flocks.

For basic failure recovery, each worker stores a text file with the current work
unit index it is computing. Upon restarting, workers will resume at the
beginning of the work unit index inside the recovery file if it exists.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re

from absl import app
from absl import flags
from nasbench.lib import config as _config
from nasbench.lib import evaluate
from nasbench.lib import model_metrics_pb2
from nasbench.lib import model_spec
import numpy as np
import tensorflow as tf


flags.DEFINE_string('models_file', '',
                    'JSON file containing models.')
flags.DEFINE_string('remainders_file', '',
                    'JSON file containing list of remainders as tuples of'
                    ' (module hash, repeat num). If provided, only the runs in'
                    ' the list will be evaluated, otherwise, all models inside'
                    ' models_file will be evaluated.')
flags.DEFINE_string('model_id_regex', '^',
                    'Regex of models to train. Model IDs are MD5 hashes'
                    ' which match ([a-f0-9]{32}).')
flags.DEFINE_string('output_dir', '', 'Base output directory.')
flags.DEFINE_integer('worker_id', 0,
                     'Worker ID within this flock, starting at 0.')
flags.DEFINE_integer('worker_id_offset', 0,
                     'Worker ID offset added.')
flags.DEFINE_integer('total_workers', 1,
                     'Total number of workers, across all flocks.')
FLAGS = flags.FLAGS

CHECKPOINT_PREFIX = 'model.ckpt'
RESULTS_FILE = 'results.json'
# Checkpoint 1 is a side-effect of pre-initializing the model weights and can be
# deleted during the clean-up step.
CHECKPOINT_1_PREFIX = 'model.ckpt-1.'


class NumpyEncoder(json.JSONEncoder):
  """Converts numpy objects to JSON-serializable format."""

  def default(self, obj):
    if isinstance(obj, np.ndarray):
      # Matrices converted to nested lists
      return obj.tolist()
    elif isinstance(obj, np.generic):
      # Scalars converted to closest Python type
      return np.asscalar(obj)
    return json.JSONEncoder.default(self, obj)


class Evaluator(object):
  """Manages evaluating a subset of the total models."""

  def __init__(self,
               models_file,
               output_dir,
               worker_id=0,
               total_workers=1,
               model_id_regex='^'):
    self.config = _config.build_config()
    with tf.gfile.Open(models_file) as f:
      self.models = json.load(f)

    self.remainders = None
    self.ordered_keys = None

    if FLAGS.remainders_file:
      # Run only the modules and repeat numbers specified
      with tf.gfile.Open(FLAGS.remainders_file) as f:
        self.remainders = json.load(f)
      self.remainders = sorted(self.remainders)
      self.num_models = len(self.remainders)
      self.total_work_units = self.num_models
    else:
      # Filter keys to only those that fit the regex and order them so all
      # workers see a canonical ordering.
      regex = re.compile(model_id_regex)
      evaluated_keys = [key for key in self.models.keys() if regex.match(key)]
      self.ordered_keys = sorted(evaluated_keys)
      self.num_models = len(self.ordered_keys)
      self.total_work_units = self.num_models * self.config['num_repeats']

    self.total_workers = total_workers

    # If the worker is recovering from a restart, figure out where to restart
    worker_recovery_dir = os.path.join(output_dir, '_recovery')
    tf.gfile.MakeDirs(worker_recovery_dir)   # Silently succeeds if exists
    self.recovery_file = os.path.join(worker_recovery_dir, str(worker_id))
    if tf.gfile.Exists(self.recovery_file):
      with tf.gfile.Open(self.recovery_file) as f:
        self.current_index = int(f.read())
    else:
      self.current_index = worker_id
      with tf.gfile.Open(self.recovery_file, 'w') as f:
        f.write(str(self.current_index))

    assert self.current_index % self.total_workers == worker_id
    self.output_dir = output_dir

  def run_evaluation(self):
    """Runs the worker evaluation loop."""
    while self.current_index < self.total_work_units:
      # Perform the expensive evaluation of the model at the current index
      self._evaluate_work_unit(self.current_index)

      self.current_index += self.total_workers
      with tf.gfile.Open(self.recovery_file, 'w') as f:
        f.write(str(self.current_index))

  def _evaluate_work_unit(self, index):
    """Runs the evaluation of the model at the specified index.

    The index records the current index of the work unit being evaluated. Each
    worker will only compute the work units with index modulo total_workers
    equal to the worker_id.

    Args:
      index: int index into total work units.
    """
    if self.remainders:
      assert self.ordered_keys is None
      model_id = self.remainders[index][0]
      model_repeat = self.remainders[index][1]
    else:
      model_id = self.ordered_keys[index % self.num_models]
      model_repeat = index // self.num_models + 1

    matrix, labels = self.models[model_id]
    matrix = np.array(matrix)

    # Re-label to config['available_ops']
    labels = (['input'] +
              [self.config['available_ops'][lab] for lab in labels[1:-1]] +
              ['output'])
    spec = model_spec.ModelSpec(matrix, labels)
    assert spec.valid_spec
    assert np.sum(spec.matrix) <= self.config['max_edges']

    # Split the directory into 16^2 roughly equal subdirectories
    model_dir = os.path.join(self.output_dir,
                             model_id[:2],
                             model_id,
                             'repeat_%d' % model_repeat)
    try:
      meta = evaluate.train_and_evaluate(spec, self.config, model_dir)
    except evaluate.AbortError:
      # After hitting the retry limit, the job will continue to the next work
      # unit. These failed jobs may need to be re-run at a later point.
      return

    # Write data to model_dir
    output_file = os.path.join(model_dir, RESULTS_FILE)
    with tf.gfile.Open(output_file, 'w') as f:
      json.dump(meta, f, cls=NumpyEncoder)

    # Delete some files to reclaim space
    self._clean_model_dir(model_dir)

  def _clean_model_dir(self, model_dir):
    """Cleans the output model directory to reclaim disk space."""
    saved_prefixes = [CHECKPOINT_PREFIX, RESULTS_FILE]
    all_files = tf.gfile.ListDirectory(model_dir)
    files_to_keep = set()
    for filename in all_files:
      for prefix in saved_prefixes:
        if (filename.startswith(prefix) and
            not filename.startswith(CHECKPOINT_1_PREFIX)):
          files_to_keep.add(filename)

    for filename in all_files:
      if filename not in files_to_keep:
        full_filename = os.path.join(model_dir, filename)
        if tf.gfile.IsDirectory(full_filename):
          tf.gfile.DeleteRecursively(full_filename)
        else:
          tf.gfile.Remove(full_filename)


def main(args):
  del args  # Unused
  worker_id = FLAGS.worker_id + FLAGS.worker_id_offset
  evaluator = Evaluator(
      models_file=FLAGS.models_file,
      output_dir=FLAGS.output_dir,
      worker_id=worker_id,
      total_workers=FLAGS.total_workers,
      model_id_regex=FLAGS.model_id_regex)
  evaluator.run_evaluation()


if __name__ == '__main__':
  app.run(main)
