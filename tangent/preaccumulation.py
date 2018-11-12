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


def preprocess(func):
  """Write preaccumulation parameters from decorator to annotation if available."""

  resolved_func = anno.getanno(func, 'func')
  preacc_params = getattr(resolved_func, PREACCUMULATION_FIELD, {'enabled': False})

  if preacc_params is not None:
    preaccumulate_decorators = [(i, dec) for i, dec in enumerate(func.decorator_list) if isinstance(dec, gast.Call) and dec.func.attr == preaccumulate.__name__]

    if len(preaccumulate_decorators) > 1:
      raise ValueError('Multiple preaccumulate decorators on one function are not allowed.')

    # Remove the preaccumulate decorator from the function's decorator list
    if len(preaccumulate_decorators) > 0:
      # FIXME: Is this really necessary? Currently the tangent compilation does not work otherwise (NameError)
      del func.decorator_list[preaccumulate_decorators[0][0]]

  # Copy preaccumulation parameters from attr to node annotation
  anno.setanno(func, PREACCUMULATION_ANNO, preacc_params)


def enabled(node):
  """Check if preaccumulation is enabled for the node."""
  return anno.getanno(node, PREACCUMULATION_ANNO, {'enabled': False})['enabled']


class Node(object):
  """A node in the DAG."""
  __slots__ = ['idx', 'name', 'value', 'stmnt']

  def __init__(self, stmnt):
    self.idx = None
    self.name = None
    self.value = None
    self.stmnt = stmnt

  def __repr__(self):
    return '<DAG node: {}>'.format(self.stmnt)

  def __str__(self):
    string_repr = ""
    if isinstance(self.name, gast.Name):
      string_repr = str(self.name.id)

    if string_repr != "" and self.value is not None:
      string_repr += ": "

    if isinstance(self.value, gast.Name):
      string_repr += str(self.value.id)
    if isinstance(self.value, gast.BinOp) or isinstance(self.value, gast.UnaryOp):
      string_repr += str(type(self.value.op).__name__)
    if isinstance(self.value, gast.Call):
      string_repr += str(self.value.func.attr)
    if isinstance(self.value, gast.Tuple):
      string_repr += str('Tuple')

    if string_repr == "":
      if isinstance(self.stmnt, gast.Name):
        string_repr = str(self.stmnt.id)
      elif isinstance(self.stmnt, gast.Num):
        string_repr = str(self.stmnt.n)
      else:
        string_repr = str(self.stmnt)

    return string_repr


class Edge(object):
  __slots__ = ['idx', 'source', 'target', 'value']

  def __init__(self, value = None):
    self.idx = None
    self.source = None
    self.target = None
    self.value = value


class DiGraph(object):
  def __init__(self):
    self.__nodes = {}
    self.__edges = {}
    self.__out_edges = {}
    self.__in_edges = {}
    self.__next_id = 0

  def insert_node(self, node):
    idx = self.__next_id
    node.idx = idx

    self.__nodes[idx] = node
    self.__next_id += 1

    self.__out_edges[idx] = set()
    self.__in_edges[idx] = set()
    return idx

  def create_edge(self, source, target, value = None):
    edge = Edge(value)
    assert source in self.__nodes and target in self.__nodes
    edge.source = source
    edge.target = target

    idx = self.__next_id
    edge.idx = idx

    self.__edges[idx] = edge
    self.__next_id += 1

    self.__out_edges[source].add(idx)
    self.__in_edges[target].add(idx)

    return idx

  def pop_node(self, node_id):
    def pop_all_edges(edge_set):
      while not edge_set.empty():
        edge_id = next(iter(edge_set))
        self.pop_edge(edge_id)

    pop_all_edges(self.__out_edges[node_id])
    pop_all_edges(self.__in_edges[node_id])
    
    del self.__out_edges[node_id]
    del self.__in_edges[node_id]

    return self.__nodes.pop(node_id)

  def pop_edge(self, edge_id):
    edge = self.__edges.pop(edge_id)
    self.__out_edges[edge.source].remove(edge)
    self.__in_edges[edge.target].remove(edge)
    return edge

  def node(self, node_id):
    return self.__nodes[node_id]

  def edge(self, edge_id):
    return self.__edges[edge_id]

  def nodes(self):
    def node_gen():
      for n in self.__nodes.values():
        yield n
    
    return node_gen()

  def edges(self):
    def edge_gen():
      for e in self.__edges.values():
        yield e

    return edge_gen()

  def out_edges(self, node_id):
    def out_edge_gen():
      for e in self.__out_edges[node_id]:
        yield self.edge(e)
    
    return out_edge_gen()

  def in_edges(self, node_id):
    def in_edge_gen():
      for e in self.__in_edges[node_id]:
        yield self.edge(e)

    return in_edge_gen()

  def out_nodes(self, node_id):
    def out_node_gen():
      for e in self.out_edges(node_id):
        yield self.node(e.target)
    
    return out_node_gen()

  def in_nodes(self, node_id):
    def in_node_gen():
      for e in self.in_edges(node_id):
        yield self.node(e.source)

    return in_node_gen()

def function_to_dag(func):
  """Creates a DAG from a function definition."""

  print("TODO: Perform preaccumulation...")
  dag = create_dag(func.body, func.args.args, None, None)
  print("")
  print(dag_to_dot(dag))
  print("")

def create_dag(nodes, inputs, outputs, wrt):
  # It is assumed that the nodes went through ANF transformation (e.g. no
  # nested binary operations)

  dag = DiGraph()
  root = dag.insert_node(Node('Inputs'))

  # Stores the node which last assigned any value to a given name
  last_assign = {}

  # Add all inputs to the DAG
  for input in inputs:
    input_id = dag.insert_node(Node(input))
    dag.create_edge(root, input_id)

    last_assign[input.id] = input_id

  # Create DAG nodes for all statements
  for node in nodes:
    dag_node = Node(node)
    dag_node_id = dag.insert_node(dag_node)

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
          num_node_id = dag.insert_node(Node(input))
        dag.create_edge(num_node_id, dag_node_id)
      else:
        raise TypeError('Unsupported type "{}" of node '
                        '"{}"'.format(type(input).__name__, input))

  return dag

  return root, dag_nodes

def dag_to_dot(dag):
  strings = []

  strings += ["digraph G {"]
  for node in dag.nodes():
    for succ in dag.out_nodes(node.idx):
      strings += ['\t"{}" -> "{}"'.format(str(node), str(succ))]
  strings += ["}"]

  dot_string = '\n'.join(strings)
  return dot_string
