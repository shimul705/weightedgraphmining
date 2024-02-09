from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys


AUTO_EDGE_ID = -1
VACANT_VERTEX_LABEL = None
VACANT_GRAPH_ID = -1


class Graph(object):
    """A simple Graph class."""

    def __init__(self, name, vertices=None, edges=None,
                 is_undirected=True, eid_auto_increment=True):
        """Initialize Graph instance."""
        self.name = name
        self.vertices = dict()
        self.edges = dict()
        self.eid_auto_increment = eid_auto_increment
        if vertices:
            for vertex in vertices:
                self.add_vertex(vertex)
        if edges:
            for edge in edges:
                self.add_edge(*edge.split())
        self.is_undirected = is_undirected

    def add_vertex(self, vertex):
        """Add a vertex to this graph."""
        vid, vlb = vertex.split()
        if vid not in self.vertices:
            self.vertices[vid] = vlb
            return True
        else:
            return False

    def add_edge(self, eid, frm, to, elb, weight=None):
        """Add an edge to this graph."""
        if eid == AUTO_EDGE_ID and self.eid_auto_increment:
            eid = len(self.edges)
        if eid in self.edges:
            return False
        if frm not in self.vertices:
            print('Source vertex {} is not in the graph.'.format(frm),
                  file=sys.stderr)
            return False
        if to not in self.vertices:
            print('Target vertex {} is not in the graph.'.format(to),
                  file=sys.stderr)
            return False
        self.edges[eid] = {'frm': frm, 'to': to, 'elb': elb, 'weight': weight}
        return True
