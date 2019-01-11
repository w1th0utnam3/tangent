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
from copy import deepcopy
import gast
import numpy

import tangent
from tangent import annotations as anno
from tangent import dag as dag_
from tangent import forward_ad
from tangent import naming
from tangent import quoting
from tangent import reverse_ad
from tangent import template
from tangent import utils

PREACCUMULATION_ANNO = 'preaccumulation'
PREACCUMULATION_FIELD = '_preaccumulation'


def preaccumulate(mode='full'):
  """Decorator to mark an entire function for preaccumulation."""
  def preacc_wrapper(func):
    # TODO: Use attr to store any parameters
    setattr(func, PREACCUMULATION_FIELD, {'enabled': True,
                                          'mode': mode,
                                          'func': func})
    return func

  return preacc_wrapper


def del_preaccumulate_fields(func):
  if anno.hasanno(func, 'func'):
    func = anno.getanno(func, 'func')

  delattr(func, PREACCUMULATION_FIELD)


def preprocess(func):
  """Write preaccumulation parameters from decorator to annotation if available."""

  resolved_func = anno.getanno(func, 'func')
  preacc_params = getattr(resolved_func, PREACCUMULATION_FIELD, {'enabled': False})

  if preacc_params is None:
    preacc_params = {'enabled': False}
  else:
    preaccumulate_decorators = [(i, dec) for i, dec in enumerate(func.decorator_list)
                                if isinstance(dec, gast.Call) and dec.func.attr == preaccumulate.__name__]

    if len(preaccumulate_decorators) > 1:
      raise ValueError('Multiple preaccumulate decorators on one function are not allowed.')

    # Remove the preaccumulate decorator from the function's decorator list
    if len(preaccumulate_decorators) > 0:
      # FIXME: Is this really necessary? Currently the tangent compilation does not work otherwise (NameError)
      del func.decorator_list[preaccumulate_decorators[0][0]]

  # Copy preaccumulation parameters from attr to node annotation
  anno.setanno(func, PREACCUMULATION_ANNO, preacc_params)
  return preacc_params


def enabled(node):
  """Check if preaccumulation is enabled for the node."""
  return anno.getanno(node, PREACCUMULATION_ANNO, {'enabled': False})['enabled']


def forward_preacc(node, wrt, motion, check_dims, verbose=0):
  """Perform forward preaccumulation in a reverse mode context"""

  if verbose >= 0:
    print('Performing forward preaccumulation')

  func = anno.getanno(node, 'func')
  # Create working copy to avoid changing the original nodes
  forward_node = deepcopy(node)
  # Generate tangent code of the function
  forward_node, required = forward_ad.forward_ad(forward_node, wrt, True,
                                                 check_dims)
  tngt_def = forward_node.body[0]

  # Get names of all involved functions
  func_name = func.__name__
  tangent_name = tngt_def.name
  primal_name = naming.primal_name(func, wrt)
  adjoint_name = naming.adjoint_name(func, wrt)

  # All arguments of original primal
  func_args = node.args.args
  # Active primal arguments
  active_args = [func_args[i] for i in wrt]
  # All arguments of the tangent
  tangent_args = tngt_def.args.args

  assert len(func_args) + len(active_args) == len(tangent_args)

  # Tangents of all active variables (dx, ...)
  tangents = tangent_args[len(func_args):]

  assert [arg.id for arg in func_args + tangents] == [arg.id for arg in tangent_args]

  # Template for driver used to accumulate the Jacobian in primal section
  def primal_driver_template(_args, _active_args, _tangents,
                             _primal_call, _tangent_call, _dz, _z):
    def _primal_call(_stack, _args):
      # FIXME: Generate/ensure unique names for the local variables?
      _dargs = tangent.init_grad([_active_args])

      n_jac_pushes = 0
      for _dseed in tangent.unit_seed_directions(_dargs):
        _dz, _z = _tangent_call(_args, *_dseed)
        raise RuntimeError('')
        tangent.push(_stack, _dz, 'op_id')
        n_jac_pushes += 1
      tangent.push(_stack, n_jac_pushes, 'jac_pushes_id')
      tangent.push(_stack, _z, 'result_id')
      return _z

  pri = template.replace(
    primal_driver_template,
    replace_grad=template.Replace.NONE,
    namer=None,
    _args=func_args,
    _active_args=active_args,
    _tangents=tangents,
    _primal_call=primal_name,
    _tangent_call=tangent_name,
    _dz='_d{}'.format(func_name),
    _z='_{}'.format(func_name)
  )[0]

  if verbose >= 2:
    print("Jacobian accumulation primal driver:")
    print(quoting.to_source(pri))

  # Template for driver used to restore the Jacobian in adjoint section and calculate vector-Jacobian product
  def adjoint_driver_template(_adjoint, _dz, _bz, _projected_jacobian):
    # FIXME: Assumes that original func only has one or multiple scalar outputs
    def _adjoint(_stack, _bz, *args):
      result = tangent.pop(_stack, 'result_id')
      n_jac_pushes = tangent.pop(_stack, 'jac_pushes_id')
      # FIXME: This has to be replaced by corresponding init_grad statements?
      _projected_jacobian = [0.0] * n_jac_pushes
      for i in range(n_jac_pushes):
        _dz = tangent.pop(_stack, 'op_id')

        # FIXME: For higher dimensions: dz * bz or bz * dz?
        # FIXME: If input variable was list, projected jacobian also has to be list
        #        -> counter i should go over variables, whereas n_jac_pushes counts all seed directions
        #        -> seperator that distinguishes input variables and dimensions is needed
        if isinstance(_dz, (tuple, list)):
          for j in range(len(_dz)):
            _projected_jacobian[i] += _dz[j] * _bz[j]
        else:
          _projected_jacobian[i] += _dz * _bz
      return tuple(reversed(_projected_jacobian)) + (result,)

  adj = template.replace(
    adjoint_driver_template,
    replace_grad=template.Replace.NONE,
    namer=None,
    _adjoint=adjoint_name,
    _dz='_d{}'.format(func_name),
    _bz='_b{}'.format(func_name),
    _projected_jacobian='_d{}s_times_b{}s'.format(func_name, func_name)
  )[0]

  if verbose >= 2:
    print("Jacobian evaluation adjoint driver:")
    print(quoting.to_source(adj))

  # TODO: What to do with op_id? Have a look at how reverse mode treats for-loops
  # TODO: Properly support split and joint motion?
  forward_node = gast.Module(body=[tngt_def, pri, adj])
  return forward_node, []


def reverse_preacc(node, wrt, motion, check_dims, verbose=0):
  """Perform reverse preaccumulation in a forward mode context"""

  if verbose >= 0:
    print('Performing reverse preaccumulation')

  func = anno.getanno(node, 'func')
  # Create working copy to avoid changing the original nodes
  adjoint_node = deepcopy(node)
  # Generate adjoint code of the function
  adjoint_node, required, stack = reverse_ad.reverse_ad(adjoint_node, wrt=wrt,
                                                        preserve_result=False,
                                                        check_dims=check_dims)
  adjoint_node = reverse_ad.split(adjoint_node, stack)
  primal_def, adjoint_def = adjoint_node.body

  # Get names of all relevant functions
  tangent_name = naming.tangent_name(func, wrt)
  primal_name = primal_def.name
  adjoint_name = adjoint_def.name

  # All arguments of original function
  func_args = node.args.args
  # Active function arguments
  active_args = [func_args[i] for i in wrt]
  # Tangents of active arguments
  arg_tangents = deepcopy(active_args)
  for it in arg_tangents:
    it.id = 'd{}'.format(it.id)

  def tangent_driver_template(_args, _arg_tangents,
                              _tangent_call, _primal_call, _adjoint_call):
    def _tangent_call(_args, _arg_tangents):
      _stack = tangent.Stack()
      _return = _primal_call(_stack, _args)

      _dargs = (_arg_tangents,)
      _dreturn = tangent.init_grad(_return)
      _dreturn = _dreturn if isinstance(_dreturn, list) else [_dreturn]

      # FIXME: Are all return values always active?
      # FIXME: Enumerate not compatible with nested lists/numpy arrays in lists
      for i, _bseed in enumerate(tangent.unit_seed_directions(_return)):
        # FIXME: Make copies of the stack instead of re-evaluating the primal
        if _stack is None:
            _stack = tangent.Stack()
            _primal_call(_stack, _args)

        _bargs = _adjoint_call(_stack, _bseed, _args)

        # FIXME: For higher dimensions: dz * bz or bz * dz?
        for j in range(len(_dargs)):
          _dreturn[i] = tangent.add_grad(_dreturn[i], tangent.mult_grad(_bargs[j], _dargs[j]))

        _stack = None

      dt = tuple(_dreturn)
      t = _return
      return dt, t

  tangent_driver_def = template.replace(
    tangent_driver_template,
    replace_grad=template.Replace.NONE,
    namer=None,
    _args=func_args,
    _arg_tangents=arg_tangents,
    _tangent_call=tangent_name,
    _primal_call=primal_name,
    _adjoint_call=adjoint_name
  )[0]

  if verbose >= 2:
    print("Jacobian accumulation driver:")
    print(quoting.to_source(tangent_driver_def))

  adjoint_node = gast.Module(body=[primal_def, adjoint_def, tangent_driver_def])
  return adjoint_node, []


def from_decorator(node, wrt, motion, check_dims, verbose=0):
  """Perform preaccumulation on a node that was marked for preaccumulation using the decorator."""

  # TODO: What about nested function calls? Nested preaccumulate function calls?
  preacc_params = anno.getanno(node, PREACCUMULATION_ANNO)

  if preacc_params['enabled']:
    preacc_mode = preacc_params['mode']
    if preacc_mode == 'forward':
      return forward_preacc(node, wrt, motion, check_dims, verbose)
    elif preacc_mode == 'reverse':
      return reverse_preacc(node, wrt, motion, check_dims, verbose)
    elif preacc_mode == 'cross_country':
      return function_to_dag(node)


def function_to_dag(func):
  """Create a DAG from a function definition."""

  print('TODO: Perform preaccumulation...')
  dag = create_dag(func.body, func.args.args, None, None)
  print('')
  print(dag_to_dot(dag))
  print('')


def create_dag(nodes, inputs, outputs, wrt):
  # It is assumed that the nodes went through ANF transformation (e.g. no
  # nested binary operations)

  dag = dag_.DiGraph()
  root = dag.add_node(dag_.Node('Inputs'))

  # Stores the node which last assigned any value to a given name
  last_assign = {}

  # Add all inputs to the DAG
  for input in inputs:
    input_id = dag.add_node(dag_.Node(input))
    dag.create_edge(root, input_id)

    last_assign[input.id] = input_id

  # Create DAG nodes for all statements
  for node in nodes:
    dag_node = dag_.Node(node)
    dag_node_id = dag.add_node(dag_node)

    node_name = None
    node_value = None

    node_inputs = []

    if isinstance(node, gast.Assign):
      # Get the target of the assign statement
      if len(node.targets) > 1:
        raise RuntimeError('Tuple unpacking currently not supported.')
      target = node.targets[0]

      node_name = target
      node_value = node.value

      if target.id in last_assign:
        raise RuntimeError('Input has to be in ANF. Cannot assign repeatedly '
                           'to the same variable!')

      last_assign[target.id] = dag_node_id
    elif isinstance(node, gast.Return):
      node_value = node.value
    else:
      raise TypeError('Encountered unsupported node type "{}" of node "{}" '
                      'during DAG creation.'.format(type(node).__name__,
                                                    node))

    dag_node.name = node_name
    dag_node.value = node_value

    # Collect inputs of the current node
    node_inputs = []
    if isinstance(node_value, gast.BinOp):
      node_inputs += [node_value.left, node_value.right]
    elif isinstance(node_value, gast.UnaryOp):
      node_inputs += [node_value.operand]
    elif isinstance(node_value, gast.Call):
      node_inputs += node_value.args
    elif isinstance(node_value, gast.Tuple):
      node_inputs += node_value.elts
    elif isinstance(node_value, gast.Name):
      node_inputs += [node_value]
    else:
      raise TypeError('Encountered unsupported node type "{}" of node "{}" '
                      'during DAG creation.'.format(type(node_value).__name__,
                                                    node_value))

    # Link inputs to the current node
    for input in node_inputs:
      if isinstance(input, gast.Name):
        if input.id not in last_assign:
          raise RuntimeError('Usage of undeclared variable "{}" in an '
                             'expression.'.format(input.id))
        dag.create_edge(last_assign[input.id], dag_node_id)
      elif isinstance(input, gast.Num):
        if input.n in last_assign:
          num_node_id = last_assign[input.n]
        else:
          num_node_id = dag.add_node(dag_.Node(input))
        dag.create_edge(num_node_id, dag_node_id)
      else:
        raise TypeError('Unsupported type "{}" of node '
                        '"{}"'.format(type(input).__name__, input))

  return dag


def dag_to_dot(dag):
  """Create a dot file representation of the given DAG."""
  strings = []

  def print_node(node):
    string_repr = ""
    if isinstance(node.name, gast.Name):
      string_repr = str(node.name.id)

    if string_repr != "" and node.value is not None:
      string_repr += ": "

    if isinstance(node.value, gast.Name):
      string_repr += str(node.value.id)
    if isinstance(node.value, gast.BinOp) or isinstance(node.value, gast.UnaryOp):
      string_repr += str(type(node.value.op).__name__)
    if isinstance(node.value, gast.Call):
      string_repr += str(node.value.func.attr)
    if isinstance(node.value, gast.Tuple):
      string_repr += str('Tuple')

    if string_repr == "":
      if isinstance(node.stmnt, gast.Name):
        string_repr = str(node.stmnt.id)
      elif isinstance(node.stmnt, gast.Num):
        string_repr = str(node.stmnt.n)
      else:
        string_repr = str(node.stmnt)

    return string_repr

  strings += ["digraph G {"]
  for node in dag.nodes():
    for succ in dag.succ_nodes(node.id):
      strings += ['\t"{}" -> "{}"'.format(print_node(node), print_node(succ))]
  strings += ["}"]

  dot_string = '\n'.join(strings)
  return dot_string
