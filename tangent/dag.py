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
"""Classes to represent and work with a directed acyclic graph."""


class Node(object):
  __slots__ = ['id', 'name', 'value', 'stmnt']

  def __init__(self, stmnt):
    self.id = None
    self.name = None
    self.value = None
    self.stmnt = stmnt

  def __repr__(self):
    return '<DAG node: {}>'.format(self.stmnt)


class Edge(object):
  __slots__ = ['id', 'source', 'target', 'value']

  def __init__(self, value = None):
    self.id = None
    self.source = None
    self.target = None
    self.value = value


class DiGraph(object):
  def __init__(self):
    """Create an empty directed graph."""
    self.__nodes = {}
    self.__edges = {}
    self.__out_edges = {}
    self.__in_edges = {}
    self.__next_id = 0

  def add_node(self, node):
    """Add a node to the graph. Assigns an id and returns it."""
    idx = self.__next_id
    node.id = idx

    self.__nodes[idx] = node
    self.__next_id += 1

    self.__out_edges[idx] = set()
    self.__in_edges[idx] = set()
    return idx

  def create_edge(self, source, target, value = None):
    """Create an edge from source to target node. Returns id of the edge."""
    edge = Edge(value)
    assert source in self.__nodes and target in self.__nodes
    edge.source = source
    edge.target = target

    idx = self.__next_id
    edge.id = idx

    self.__edges[idx] = edge
    self.__next_id += 1

    self.__out_edges[source].add(idx)
    self.__in_edges[target].add(idx)

    return id

  def node(self, node_id):
    """Get the node with the specified id."""
    return self.__nodes[node_id]

  def edge(self, edge_id):
    """Get the edge with the specified id."""
    return self.__edges[edge_id]

  def pop_node(self, node_id):
    """Remove a node and all incoming/outgoing edges and return it."""
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
    """Remove an edge and return it."""
    edge = self.__edges.pop(edge_id)
    self.__out_edges[edge.source].remove(edge)
    self.__in_edges[edge.target].remove(edge)
    return edge

  def nodes(self):
    """Generator yielding all nodes of the graph."""
    def node_gen():
      for n in self.__nodes.values():
        yield n
    
    return node_gen()

  def edges(self):
    """Generator yielding all edges of the graph."""
    def edge_gen():
      for e in self.__edges.values():
        yield e

    return edge_gen()

  def out_edges(self, source_node_id):
    """Generator yielding all outgoing edges from a node."""
    def out_edge_gen():
      for e in self.__out_edges[source_node_id]:
        yield self.edge(e)
    
    return out_edge_gen()

  def in_edges(self, target_node_id):
    """Generator yielding all incoming edges to a node."""
    def in_edge_gen():
      for e in self.__in_edges[target_node_id]:
        yield self.edge(e)

    return in_edge_gen()

  def succ_nodes(self, node_id):
    """Generator yielding all direct successor ('downstream') nodes."""
    def out_node_gen():
      for e in self.out_edges(node_id):
        yield self.node(e.target)
    
    return out_node_gen()

  def pred_nodes(self,  node_id):
    """Generator yielding all direct predecessor ('upstream') nodes."""
    def in_node_gen():
      for e in self.in_edges(node_id):
        yield self.node(e.source)

    return in_node_gen()
