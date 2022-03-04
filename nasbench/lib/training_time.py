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

"""Tools to measure and limit the training time of a TF model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

# Name of scope where to put all timing-related ops and variables.
_SCOPE_NAME = 'timing'

# Variables names:
_START_VAR = 'start_timestamp'
_STEPS_VAR = 'steps'
_PREV_VAR = 'previous_time'
_TOTAL_VAR = 'total_time'

# The name of the TF variable that will hold the total training time so far.
# Get with estimator.get_variable_value(TOTAL_TIME_NAME) after
# running estimator.train(). Note that this time includes the time spent in
# previous calls to train() as well.
TOTAL_TIME_NAME = '%s/%s' % (_SCOPE_NAME, _TOTAL_VAR)

# We have a fixed temporal precision of one millisecond.
# We used fixed precision to represent seconds since the epoch, as a tf.int64,
# because tf.float32 lacks precision for large values and tf.float64 is not
# supported on TPU.
_INTERNAL_TIME_PRECISION = 1000


def _seconds_to_internal_time(seconds):
  """Converts seconds to fixed-precision time."""
  return tf.to_int64(tf.round(seconds * _INTERNAL_TIME_PRECISION))


def _internal_time_to_seconds(internal_time):
  """Converts fixed-precision time to seconds."""
  return tf.to_float(internal_time / _INTERNAL_TIME_PRECISION)


Timing = collections.namedtuple(  # pylint: disable=g-bad-name
    'Timing',
    [
        # A SessionRunHook instance that must be passed to estimator.train()
        # through its `hooks` arg.
        'train_hook',

        # A CheckpointSaverListener instance. This must be passed to
        # estimator.train() through its `saving_listeners` arg if and only if
        # checkpoints are being saved.
        'saving_listener',
    ])


def limit(max_train_secs=None):
  """Provides hooks and ops to measure/limit the training time of a model.

  This is done by direct measurement of the time spent on training steps. It
  excludes time spent saving checkpoints or due to pre-emptions.

  Args:
    max_train_secs: the desired training time limit. It is possible that this
      may be exceeded by the time it takes to run 1 step. If None, training will
      not be limited by time but timing variables will still be created.

  Returns:
    A Timing named tuple.
  """
  train_hook = _TimingRunHook(max_train_secs)
  saving_listener = _TimingSaverListener()
  return Timing(train_hook=train_hook, saving_listener=saving_listener)


def get_total_time():
  """Returns the timing/total_time variable, regardless of current scope.

  You may need to call force_create_timing_vars() first, or else there is a risk
  that you may try to retrieve a variable that doesn't yet exist.

  Returns:
    A TF Variable.

  Raises:
    RuntimeError: if the variable has not been created yet.
  """
  timing_vars = _get_or_create_timing_vars()
  return timing_vars.total_time


_TimingVars = collections.namedtuple(  # pylint: disable=g-bad-name
    '_TimingVars',
    [
        # TF variable to be used to store the timestamp (in seconds) of the
        # first training step after the last checkpoint save (or the first
        # training step ever if no save has happened yet). -1 means no steps
        # have been run since the last checkpoint save.
        'start_timestamp',

        # TF variable to be used to store the number of steps since the last
        # checkpoint save (or the beginning of training if no save has happened
        # yet).
        'steps',

        # TF variable to be used to store the training time up to the last
        # checkpoint saved.
        'previous_time',

        # TF variable to be used to accumulate the total training time up
        # to the last step run. This time will not include gaps resulting from
        # checkpoint saving or pre-emptions.
        'total_time',
    ])


class _TimingRunHook(tf.estimator.SessionRunHook):
  """Hook to stop the training after a certain amount of time."""

  def __init__(self, max_train_secs=None):
    """Initializes the instance.

    Args:
      max_train_secs: the maximum number of seconds to train for. If None,
        training will not be limited by time.
    """
    self._max_train_secs = max_train_secs

  def begin(self):
    with tf.name_scope(_SCOPE_NAME):
      # See _get_or_create_timing_vars for the definitions of these variables.
      timing_vars = _get_or_create_timing_vars()

      # An op to produce a tensor with the latest timestamp.
      self._end_op = _seconds_to_internal_time(tf.timestamp(name='end'))

      # An op to update the timing_vars.start_timestamp variable.
      self._start_op = tf.cond(
          pred=tf.equal(timing_vars.steps, 0),
          true_fn=lambda: timing_vars.start_timestamp.assign(self._end_op),
          false_fn=lambda: timing_vars.start_timestamp)

      # An op to update the step.
      with tf.control_dependencies([self._start_op]):
        self._step_op = timing_vars.steps.assign_add(1)

      # An op to compute the timing_vars.total_time variable.
      self._total_op = timing_vars.total_time.assign(
          timing_vars.previous_time +
          _internal_time_to_seconds(self._end_op - self._start_op))

  def before_run(self, run_context):
    return tf.train.SessionRunArgs([self._total_op, self._step_op])

  def after_run(self, run_context, run_values):
    total_time, _ = run_values.results
    if self._max_train_secs and total_time > self._max_train_secs:
      run_context.request_stop()


class _TimingSaverListener(tf.estimator.CheckpointSaverListener):
  """Saving listener to store the train time up to the last checkpoint save."""

  def begin(self):
    with tf.name_scope(_SCOPE_NAME):
      timing_vars = _get_or_create_timing_vars()

      # An op to update the timing_vars.previous_time variable.
      self._prev_op = timing_vars.previous_time.assign(timing_vars.total_time)

      # Marks that timing_vars.start_timestamp should be reset in the next step.
      self._reset_steps_op = timing_vars.steps.assign(0)

  def before_save(self, session, global_step_value):
    session.run(self._prev_op)

  def after_save(self, session, global_step_value):
    session.run(self._reset_steps_op)


def _get_or_create_timing_vars():
  """Creates variables used to measure training time.

  Returns:
    A _TimingVars named tuple.
  """
  # We always create the timing variables at root_scope / _SCOPE_NAME,
  # regardless of the scope from where this is called.
  root_scope = tf.get_variable_scope()
  with tf.variable_scope(root_scope, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(_SCOPE_NAME, reuse=tf.AUTO_REUSE):
      start_timestamp = tf.get_variable(
          _START_VAR,
          shape=[],
          dtype=tf.int64,
          initializer=tf.constant_initializer(-1),
          trainable=False)
      steps = tf.get_variable(
          _STEPS_VAR,
          shape=[],
          dtype=tf.int64,
          initializer=tf.constant_initializer(0),
          trainable=False)
      previous_time = tf.get_variable(
          _PREV_VAR,
          shape=[],
          dtype=tf.float32,
          initializer=tf.constant_initializer(0.0),
          trainable=False)
      total_time = tf.get_variable(
          _TOTAL_VAR,
          shape=[],
          dtype=tf.float32,
          initializer=tf.constant_initializer(0.0),
          trainable=False)
      return _TimingVars(
          start_timestamp=start_timestamp,
          steps=steps,
          previous_time=previous_time,
          total_time=total_time)
