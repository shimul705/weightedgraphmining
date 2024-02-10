"""Definitions of Edge, Vertex and Graph."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

VACANT_EDGE_ID = -1
VACANT_VERTEX_ID = -1
VACANT_EDGE_LABEL = -1
VACANT_VERTEX_LABEL = -1
VACANT_GRAPH_ID = -1
AUTO_EDGE_ID = -1


class Edge(object):
    """Edge class."""

    def __init__(self, eid=VACANT_EDGE_ID, frm=VACANT_VERTEX_ID, to=VACANT_VERTEX_ID, elb=VACANT_EDGE_LABEL, weight=1):
        """Initialize Edge instance."""
        self.eid = eid
        self.frm = frm
        self.to = to
        self.elb = elb
        self.weight = weight


class Vertex(object):
    """Vertex class."""

    def __init__(self, vid=VACANT_VERTEX_ID, vlb=VACANT_VERTEX_LABEL):
        """Initialize Vertex instance."""
        self.vid = vid
        self.vlb = vlb
        self.edges = dict()

    def add_edge(self, eid, frm, to, elb, weight=1):
        """Add an outgoing edge."""
        self.edges[to] = Edge(eid, frm, to, elb, weight)


class Graph(object):
    """Graph class."""

    def __init__(self, gid=VACANT_GRAPH_ID, is_undirected=True, eid_auto_increment=True):
        """Initialize Graph instance."""
        self.gid = gid
        self.is_undirected = is_undirected
        self.vertices = dict()
        self.set_of_elb = collections.defaultdict(set)
        self.set_of_vlb = collections.defaultdict(set)
        self.eid_auto_increment = eid_auto_increment
        self.counter = itertools.count()

    def get_num_vertices(self):
        """Return number of vertices in the graph."""
        return len(self.vertices)

    def add_vertex(self, vid, vlb):
        """Add a vertex to the graph."""
        if vid in self.vertices:
            return self
        self.vertices[vid] = Vertex(vid, vlb)
        self.set_of_vlb[vlb].add(vid)
        return self

    def add_edge(self, eid, frm, to, elb, weight=1):
        """Add an edge to the graph."""
        if (frm is self.vertices and
                to in self.vertices and
                to in self.vertices[frm].edges):
            return self
        if self.eid_auto_increment:
            eid = next(self.counter)
        self.vertices[frm].add_edge(eid, frm, to, elb, weight)
        self.set_of_elb[elb].add((frm, to))
        if self.is_undirected:
            self.vertices[to].add_edge(eid, to, frm, elb, weight)
            self.set_of_elb[elb].add((to, frm))
        return self

    def display(self):
        """Display the graph as text."""
        display_str = ''
        print('t # {}'.format(self.gid))
        for vid in self.vertices:
            print('v {} {}'.format(vid, self.vertices[vid].vlb))
            display_str += 'v {} {} '.format(vid, self.vertices[vid].vlb)
        for frm in self.vertices:
            edges = self.vertices[frm].edges
            for to in edges:
                if self.is_undirected:
                    if frm < to:
                        print('e {} {} {} {}'.format(frm, to, edges[to].elb, edges[to].weight))
                        display_str += 'e {} {} {} {} '.format(frm, to, edges[to].elb, edges[to].weight)
                else:
                    print('e {} {} {} {}'.format(frm, to, edges[to].elb, edges[to].weight))
                    display_str += 'e {} {} {} {}'.format(frm, to, edges[to].elb, edges[to].weight)
        return display_str

    def plot(self):
        """Visualize the graph."""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except Exception as e:
            print('Can not plot graph: {}'.format(e))
            return
        gnx = nx.Graph() if self.is_undirected else nx.DiGraph()
        vlbs = {vid: v.vlb for vid, v in self.vertices.items()}
        elbs = {}
        for vid, v in self.vertices.items():
            gnx.add_node(vid, label=v.vlb)
        for vid, v in self.vertices.items():
            for to, e in v.edges.items():
                gnx.add_edge(vid, to, label=e.elb, weight=e.weight)
                elbs[(vid, to)] = e.elb
        pos = nx.spring_layout(gnx)  # positions for all nodes
        nx.draw(gnx, pos, node_size=700, labels=vlbs, with_labels=True)
        nx.draw_networkx_edge_labels(gnx, pos, edge_labels=elbs)
        plt.show()
