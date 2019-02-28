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

"""Unit tests for scripts/run_evaluation.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tempfile

from absl.testing import flagsaver
from nasbench.scripts import run_evaluation
import tensorflow as tf



class RunEvaluationTest(tf.test.TestCase):

  def setUp(self):
    """Set up files and directories that are expected by run_evaluation."""
    # Create temp directory for output files
    self.output_dir = tempfile.mkdtemp()
    self.models_file = os.path.join(self.output_dir, 'models_file.json')

    self.toy_data = {
        'abc': ([[0, 1, 1], [0, 0, 1], [0, 0, 0]], [-1, 0, -2]),
        'abd': ([[0, 1, 0], [0, 0, 1], [0, 0, 0]], [-1, 0, -2]),
        'abe': ([[0, 0, 1], [0, 0, 0], [0, 0, 0]], [-1, 0, -2]),
    }

    with tf.gfile.Open(self.models_file, 'w') as f:
      json.dump(self.toy_data, f)

    # Create files & directories which are normally created by
    # evaluate.train_and_evaluate but have been mocked out.
    for model_id in self.toy_data:
      eval_dir = os.path.join(self.output_dir, 'ab', model_id, 'repeat_1')
      tf.gfile.MakeDirs(eval_dir)
    run_evaluation.FLAGS.train_data_files = 'unused'
    run_evaluation.FLAGS.valid_data_file = 'unused'
    run_evaluation.FLAGS.test_data_file = 'unused'
    run_evaluation.FLAGS.num_repeats = 1

  @tf.test.mock.patch.object(run_evaluation, 'evaluate')
  def test_evaluate_single_worker(self, mock_eval):
    """Tests single worker code path."""
    mock_eval.train_and_evaluate.return_value = 'unused_output'
    evaluator = run_evaluation.Evaluator(
        self.models_file, self.output_dir)
    evaluator.run_evaluation()

    expected_dir = os.path.join(self.output_dir, 'ab')
    mock_eval.train_and_evaluate.assert_has_calls([
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abc', 'repeat_1')),
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abd', 'repeat_1')),
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abe', 'repeat_1'))])

    for model_id in self.toy_data:
      self.assertTrue(tf.gfile.Exists(
          os.path.join(expected_dir, model_id, 'repeat_1', 'results.json')))

  @tf.test.mock.patch.object(run_evaluation, 'evaluate')
  def test_evaluate_multi_worker_0(self, mock_eval):
    """Tests multi worker code path for worker 0."""
    mock_eval.train_and_evaluate.return_value = 'unused_output'
    evaluator = run_evaluation.Evaluator(
        self.models_file, self.output_dir, worker_id=0, total_workers=2)
    evaluator.run_evaluation()

    expected_dir = os.path.join(self.output_dir, 'ab')
    mock_eval.train_and_evaluate.assert_has_calls([
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abc', 'repeat_1')),
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abe', 'repeat_1'))])

    for model_id in ['abc', 'abe']:
      self.assertTrue(tf.gfile.Exists(
          os.path.join(expected_dir, model_id, 'repeat_1', 'results.json')))

  @tf.test.mock.patch.object(run_evaluation, 'evaluate')
  def test_evaluate_multi_worker_1(self, mock_eval):
    """Tests multi worker code path for worker 1."""
    mock_eval.train_and_evaluate.return_value = 'unused_output'
    evaluator = run_evaluation.Evaluator(
        self.models_file, self.output_dir, worker_id=1, total_workers=2)
    evaluator.run_evaluation()

    expected_dir = os.path.join(self.output_dir, 'ab')
    mock_eval.train_and_evaluate.assert_has_calls([
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abd', 'repeat_1'))])

    self.assertTrue(tf.gfile.Exists(
        os.path.join(expected_dir, 'abd', 'repeat_1', 'results.json')))

  @tf.test.mock.patch.object(run_evaluation, 'evaluate')
  def test_evaluate_regex(self, mock_eval):
    """Tests regex filters models."""
    mock_eval.train_and_evaluate.return_value = 'unused_output'
    evaluator = run_evaluation.Evaluator(
        self.models_file, self.output_dir, model_id_regex='^ab(d|e)')
    evaluator.run_evaluation()

    expected_dir = os.path.join(self.output_dir, 'ab')
    mock_eval.train_and_evaluate.assert_has_calls([
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abd', 'repeat_1')),
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abe', 'repeat_1'))])

    for model_id in ['abd', 'abe']:
      self.assertTrue(tf.gfile.Exists(
          os.path.join(expected_dir, model_id, 'repeat_1', 'results.json')))

  @tf.test.mock.patch.object(run_evaluation, 'evaluate')
  def test_evaluate_repeat(self, mock_eval):
    """Tests evaluate with repeats."""
    mock_eval.train_and_evaluate.return_value = 'unused_output'

    # Create extra directories not created in setUp for repeat_2
    for model_id in self.toy_data:
      eval_dir = os.path.join(self.output_dir, 'ab', model_id, 'repeat_2')
      tf.gfile.MakeDirs(eval_dir)

    with flagsaver.flagsaver(num_repeats=2):
      evaluator = run_evaluation.Evaluator(
          self.models_file, self.output_dir)
      evaluator.run_evaluation()

    expected_dir = os.path.join(self.output_dir, 'ab')
    mock_eval.train_and_evaluate.assert_has_calls([
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abc', 'repeat_1')),
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abd', 'repeat_1')),
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abe', 'repeat_1')),
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abc', 'repeat_2')),
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abd', 'repeat_2')),
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abe', 'repeat_2'))])

    for model_id in self.toy_data:
      for repeat in range(2):
        self.assertTrue(tf.gfile.Exists(
            os.path.join(expected_dir, model_id,
                         'repeat_%d' % (repeat + 1), 'results.json')))

  def test_clean_model_dir(self):
    """Tests clean-up of model directory keeps only intended files."""
    model_dir = os.path.join(self.output_dir, 'ab', 'abcde', 'repeat_1')
    tf.gfile.MakeDirs(model_dir)

    # Write files which will be preserved
    preserved_files = ['model.ckpt-0.index',
                       'model.ckpt-100.index',
                       'results.json']
    for filename in preserved_files:
      with tf.gfile.Open(os.path.join(model_dir, filename), 'w') as f:
        f.write('unused')

    # Write files which will be deleted
    for filename in ['checkpoint',
                     'events.out.tfevents']:
      with tf.gfile.Open(os.path.join(model_dir, filename), 'w') as f:
        f.write('unused')

    # Create subdirectory which will be deleted
    eval_dir = os.path.join(model_dir, 'eval_dir')
    tf.gfile.MakeDirs(eval_dir)
    with tf.gfile.Open(os.path.join(eval_dir, 'events.out.tfevents'), 'w') as f:
      f.write('unused')

    evaluator = run_evaluation.Evaluator(self.models_file, self.output_dir)
    evaluator._clean_model_dir(model_dir)

    # Check only intended files are preserved
    remaining_files = tf.gfile.ListDirectory(model_dir)
    self.assertItemsEqual(remaining_files, preserved_files)

  @tf.test.mock.patch.object(run_evaluation, 'evaluate')
  def test_recovery_file(self, mock_eval):
    """Tests that evaluation recovers from restart."""
    mock_eval.train_and_evaluate.return_value = 'unused_output'

    # Write recovery file
    recovery_dir = os.path.join(self.output_dir, '_recovery')
    tf.gfile.MakeDirs(recovery_dir)
    with tf.gfile.Open(os.path.join(recovery_dir, '0'), 'w') as f:
      f.write('2')    # Resume at 3rd entry

    evaluator = run_evaluation.Evaluator(
        self.models_file, self.output_dir)
    evaluator.run_evaluation()

    expected_dir = os.path.join(self.output_dir, 'ab')
    mock_eval.train_and_evaluate.assert_has_calls([
        tf.test.mock.call(tf.test.mock.ANY, tf.test.mock.ANY,
                          os.path.join(expected_dir, 'abe', 'repeat_1'))])

    # Check that only 'abe' was evaluated, 'abc' and 'abe' are skipped due to
    # recovery.
    call_args = mock_eval.train_and_evaluate.call_args_list
    self.assertEqual(len(call_args), 1)

    # Check that recovery file is updated after run
    with tf.gfile.Open(evaluator.recovery_file) as f:
      new_idx = int(f.read())
    self.assertEqual(new_idx, 3)


if __name__ == '__main__':
  tf.test.main()
