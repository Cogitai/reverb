# Lint as: python3
# Copyright 2019 DeepMind Technologies Limited.
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

"""TFClient provides tf-ops for interacting with Reverb."""

from typing import Any, List, Optional, Union

from reverb import client as reverb_client, server as reverb_server
from reverb import replay_sample
import tensorflow.compat.v1 as tf
import tree

from reverb.cc.ops import gen_local_table_dataset_op


class ReplayLocalTableDataset(tf.data.Dataset):
  """A tf.data.Dataset which samples timesteps from the ReverbService.

  Note: The dataset returns `ReplaySample` where `data` with the structure of
  `dtypes` and `shapes`.

  Note: Uses of Python lists are converted into tuples as nest used by the
  tf.data API doesn't have good support for lists.

  Timesteps are streamed through the dataset as follows:

    1. Does an active prioritized item exists?
         - Yes: Go to 3
         - No: Go to 2.
    2. Sample a prioritized item from `table` using its sample-function and set
       the item as "active". Go to 3.
    3. Yield the next timestep within the active prioritized item. If the
       timestep was the last one within the item, clear its "active" status.

  This allows for items of arbitrary length to be streamed with limited memory.
  """

  def __init__(self,
               table: Union[reverb_server.Table, tf.Tensor],
               dtypes: Any,
               shapes: Any,
               sequence_length: Optional[int] = None,
               emit_timesteps: bool = True,
               flexible_batch_size: int = -1,
               rate_limiter_timeout_ms: int = -1):
    """Constructs a new ReplayLocalTableDataset.

    Args:
      table: Probability table to sample from.
      dtypes: Dtypes of the data output. Can be nested.
      shapes: Shapes of the data output. Can be nested.
      sequence_length: (Defaults to None, i.e unknown) The number of timesteps
        that each sample consists of. If set then the length of samples received
        from the server will be validated against this number.
      emit_timesteps: (Defaults to True) If set, timesteps instead of full
        sequences are returned from the dataset. Returning sequences instead of
        timesteps can be more efficient as the memcopies caused by the splitting
        and batching of tensor can be avoided. Note that if set to False then
        then all `shapes` must have dim[0] equal to `sequence_length`.
      flexible_batch_size: (Defaults to -1, i.e auto selected) is the maximum number
        of items to sampled from `Table` with a single call. Values > 1 enables
        `Table::SampleFlexibleBatch` to return more than one item (but no more than
        `flexible_batch_size`) in a single call without releasing the table lock iff
        the rate limiter allows it.
      rate_limiter_timeout_ms: (Defaults to -1: infinite).  Timeout
        (in milliseconds) to wait on the rate limiter when sampling from the
        table. If `rate_limiter_timeout_ms >= 0`, this is the timeout passed to
        `Table::Sample` describing how long to wait for the rate limiter to
        allow sampling. The first time that a request times out (across any of
        the workers), the Dataset iterator is closed and the sequence is
        considered finished.

    Raises:
      ValueError: If `dtypes` and `shapes` don't share the same structure.
      ValueError: If `sequence_length` is not a positive integer or None.
      ValueError: If `emit_timesteps is False` and not all items in `shapes` has
        `sequence_length` as its leading dimension.
      ValueError: If `flexible_batch_size < -1`.
      ValueError: If `rate_limiter_timeout_ms < -1`.
    """
    tree.assert_same_structure(dtypes, shapes, False)
    if sequence_length is not None and sequence_length < 1:
      raise ValueError(
          'sequence_length (%s) must be None or a positive integer' %
          sequence_length)
    if flexible_batch_size < -1:
        raise ValueError('flexible_batch_size (%d) must be an integer >= -1' %
                         flexible_batch_size)
    if rate_limiter_timeout_ms < -1:
      raise ValueError('rate_limiter_timeout_ms (%d) must be an integer >= -1' %
                       rate_limiter_timeout_ms)

    # Add the info fields.
    dtypes = replay_sample.ReplaySample(replay_sample.SampleInfo.tf_dtypes(),
                                        dtypes)
    shapes = replay_sample.ReplaySample(
        replay_sample.SampleInfo(
            tf.TensorShape([sequence_length] if not emit_timesteps else []),
            tf.TensorShape([sequence_length] if not emit_timesteps else []),
            tf.TensorShape([sequence_length] if not emit_timesteps else []),
            tf.TensorShape([sequence_length] if not emit_timesteps else [])),
        shapes)

    # If sequences are to be emitted then all shapes must specify use
    # sequence_length as their batch dimension.
    if not emit_timesteps:

      def _validate_batch_dim(path: str, shape: tf.TensorShape):
        if (not shape.ndims
            or tf.compat.dimension_value(shape[0]) != sequence_length):
          raise ValueError(
              'All items in shapes must use sequence_range (%s) as the leading '
              'dimension, but "%s" has shape %s' %
              (sequence_length, path[0], shape))

      tree.map_structure_with_path(_validate_batch_dim, shapes.data)

    # The tf.data API doesn't fully support lists so we convert all uses of
    # lists into tuples.
    dtypes = _convert_lists_to_tuples(dtypes)
    shapes = _convert_lists_to_tuples(shapes)

    self._table = table
    self._dtypes = dtypes
    self._shapes = shapes
    self._sequence_length = sequence_length
    self._emit_timesteps = emit_timesteps
    self._flexible_batch_size = flexible_batch_size
    self._rate_limiter_timeout_ms = rate_limiter_timeout_ms

    if _is_tf1_runtime():
      # Disabling to avoid errors given the different tf.data.Dataset init args
      # between v1 and v2 APIs.
      # pytype: disable=wrong-arg-count
      super().__init__()
    else:
      # DatasetV2 requires the dataset as a variant tensor during init.
      super().__init__(self._as_variant_tensor())
      # pytype: enable=wrong-arg-count

  # TODO(piyushk): Currently unsupported.
  # @classmethod
  # def from_table_signature(cls,
  #                          table: reverb_server.Table,
  #                          sequence_length: Optional[int] = None,
  #                          emit_timesteps: bool = True,
  #                          flexible_batch_size: int = -1,
  #                          rate_limiter_timeout_ms: int = -1,
  #                          get_signature_timeout_secs: Optional[int] = None,):
  #   """Constructs a ReplayLocalTableDataset using the table's signature to infer specs.
  #
  #   Note: The signature must be provided to `Table` at construction. See
  #   `Table.__init__` (./server.py) for more details.
  #
  #   Args:
  #     table: Table to read the signature and sample from.
  #     sequence_length: See __init__ for details.
  #     emit_timesteps: See __init__ for details.
  #     flexible_batch_size: See __init__ for details.
  #     rate_limiter_timeout_ms: See __init__ for details.
  #     get_signature_timeout_secs: Timeout in seconds to wait for server to
  #       respond when fetching the table signature. By default no timeout is set
  #       and the call will block indefinetely if the server does not respond.
  #
  #   Returns:
  #     ReplayLocalTableDataset using the specs defined by the table signature to build
  #       `shapes` and `dtypes`.
  #
  #   Raises:
  #     ValueError: If `table` does not exist on server at `server_address`.
  #     ValueError: If `table` does not have a signature.
  #     errors.DeadlineExceededError: If `get_signature_timeout_secs` provided and
  #       exceeded.
  #     ValueError: See __init__.
  #   """
  #   info = client.server_info(get_signature_timeout_secs)
  #   if table not in info:
  #     raise ValueError(
  #         f'Server at {server_address} does not contain any table named '
  #         f'{table}. Found: {", ".join(sorted(info.keys()))}.')
  #
  #   if not info[table].signature:
  #     raise ValueError(
  #         f'Table {table} at {server_address} does not have a signature.')
  #
  #   shapes = tree.map_structure(lambda x: x.shape, info[table].signature)
  #   dtypes = tree.map_structure(lambda x: x.dtype, info[table].signature)
  #
  #   if not emit_timesteps:
  #     batch_dim = tf.TensorShape([sequence_length])
  #     shapes = tree.map_structure(batch_dim.concatenate, shapes)
  #
  #   return cls(
  #       table=table,
  #       shapes=shapes,
  #       dtypes=dtypes,
  #       num_workers_per_iterator=num_workers_per_iterator,
  #       sequence_length=sequence_length,
  #       emit_timesteps=emit_timesteps,
  #       rate_limiter_timeout_ms=rate_limiter_timeout_ms)

  def _as_variant_tensor(self):
    return gen_local_table_dataset_op.reverb_local_table_dataset(
        table=self._table.get_ptr(),
        dtypes=tree.flatten(self._dtypes),
        shapes=tree.flatten(self._shapes),
        emit_timesteps=self._emit_timesteps,
        sequence_length=self._sequence_length or -1,
        flexible_batch_size=self._flexible_batch_size,
        rate_limiter_timeout_ms=self._rate_limiter_timeout_ms)

  def _inputs(self) -> List[Any]:
    return []

  @property
  def element_spec(self) -> Any:
    return tree.map_structure(tf.TensorSpec, self._shapes, self._dtypes)


def _convert_lists_to_tuples(structure: Any) -> Any:
  list_to_tuple_fn = lambda s: tuple(s) if isinstance(s, list) else s
  # Traverse depth-first, bottom-up
  return tree.traverse(list_to_tuple_fn, structure, top_down=False)


def _is_tf1_runtime() -> bool:
  """Returns True if the runtime is executing with TF1.0 APIs."""
  # TODO(b/145023272): Update when/if there is a better way.
  return hasattr(tf, 'to_float')
