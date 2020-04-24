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

"""Python wrappers for constructing Checkpointers to pass to ReverbServer."""

import abc
import tempfile

import numpy  # pylint: disable=unused-import
from reverb import pybind


class CheckpointerBase(metaclass=abc.ABCMeta):
  """Base class for Python wrappers of the Checkpointer."""

  @abc.abstractmethod
  def internal_checkpointer(self) -> pybind.CheckpointerInterface:
    """Creates the actual Checkpointer-object used by the C++ layer."""


class DefaultCheckpointer(CheckpointerBase):
  """Base class for storing checkpoints to as recordIO files.."""

  def __init__(self, path: str, group: str = ''):
    """Constructor of DefaultCheckpointer.

    Args:
      path: Root directory to store checkpoints in.
      group: MDB group to set as "group" of checkpoint directory. If empty
        (default) then no group is set.
    """
    self.path = path
    self.group = group

  def internal_checkpointer(self) -> pybind.CheckpointerInterface:
    """Creates the actual Checkpointer-object used by the C++ layer."""
    return pybind.create_default_checkpointer(self.path, self.group)


class TempDirCheckpointer(DefaultCheckpointer):
  """Stores and loads checkpoints from a temporary directory."""

  def __init__(self):
    super().__init__(tempfile.mkdtemp())


def default_checkpointer() -> CheckpointerBase:
  return TempDirCheckpointer()