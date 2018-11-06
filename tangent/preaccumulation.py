# Copyright 2018 Fabian Loeschner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
"""Functions to perform preaccumulation on the AST."""
from __future__ import absolute_import
import gast

from tangent import annotations as anno

PREACCUMULATION_ANNO = 'preaccumulation'
PREACCUMULATION_FIELD = '_preaccumulation'


def preaccumulate():
  """Decorator to mark an entire function for preaccumulation."""
  def preacc_wrapper(func):
    # TODO: Use attr to store any parameters
    setattr(func, PREACCUMULATION_FIELD, {'enabled': True})
    return func

  return preacc_wrapper


def _has_preaccumulate_decorator(func):
  has_decorator = False

  resolved_func = anno.getanno(func, 'func')
  preacc_params = getattr(resolved_func, PREACCUMULATION_FIELD, None)

  if preacc_params is not None:
    preaccumulate_decorators = [(i, dec) for i, dec in enumerate(func.decorator_list) if isinstance(dec, gast.Call) and dec.func.attr == preaccumulate.__name__]

    if len(preaccumulate_decorators) > 1:
      raise ValueError('Multiple preaccumulate decorators on one function are not allowed.')

    has_decorator = len(preaccumulate_decorators) > 0
    # Remove the preaccumulate decorator from the function's decorator list
    if has_decorator:
      # FIXME: Is this really necessary? Currently the tangent compilation does not work otherwise (NameError)
      del func.decorator_list[preaccumulate_decorators[0][0]]

  # Copy preaccumulation parameters from attr to node annotation
  anno.setanno(func, PREACCUMULATION_ANNO, preacc_params)

  return has_decorator