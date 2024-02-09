"""Definitions of Edge, Vertex and Graph."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

VACANT_VERTEX_ID = -1
VACANT_EDGE_ID = -1
VACANT_EDGE_LABEL = -1
AUTO_EDGE_ID = -1
VACANT_GRAPH_ID = -1
VACANT_VERTEX_LABEL = -1



class Vertex(object):
    """Vertex class."""

    def __init__(self, vid=VACANT_VERTEX_ID, vlb=VACANT_VERTEX_ID):
        """Initialize Vertex instance."""
        self.vid = vid
        self.vlb = vlb
        self.edges = dict()

    def add_edge(self, eid, frm, to, elb, weight=1):
        """Add an edge to the vertex."""
        self.edges[to] = Edge(eid, frm, to, elb, weight)


class Edge(object):
    """Edge class."""

    def __init__(self, eid=VACANT_EDGE_ID, frm=VACANT_VERTEX_ID, to=VACANT_VERTEX_ID, elb=VACANT_EDGE_LABEL, weight=1):
        """Initialize Edge instance."""
        self.eid = eid
        self.frm = frm
        self.to = to
        self.elb = elb
        self.weight = weight


class Graph(object):
    """Graph class."""

    def __init__(self, gid=VACANT_VERTEX_ID, is_undirected=True, eid_auto_increment=True):
        """Initialize Graph instance."""
        self.gid = gid
        self.vertices = dict()
        self.set_of_elb = defaultdict(set)
        self.is_undirected = is_undirected
        self.eid_auto_increment = eid_auto_increment
        self.counter = iter(range(1, 1000000))  # start with 1

    def add_vertex(self, vid, vlb):
        """Add a vertex to the graph."""
        self.vertices[vid] = Vertex(vid, vlb)
        return self

    def add_edge(self, eid, frm, to, elb, weight=1):
        """Add an edge to the graph."""
        if (frm in self.vertices and to in self.vertices[frm].edges):
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
        """Display the graph."""
        display_str = "Graph ID: {}\n".format(self.gid)
        for vid, vertex in self.vertices.items():
            display_str += "Vertex ID: {}, Label: {}\n".format(vid, vertex.vlb)
            for edge in vertex.edges.values():
                display_str += "  Edge ID: {}, From: {}, To: {}, Label: {}, Weight: {}\n".format(
                    edge.eid, edge.frm, edge.to, edge.elb, edge.weight
                )
        print(display_str)

    def plot(self):
        """Plot the graph."""
        G = nx.Graph()
        for vid, vertex in self.vertices.items():
            G.add_node(vid, label=vertex.vlb)
            for edge in vertex.edges.values():
                G.add_edge(edge.frm, edge.to, label=edge.elb, weight=edge.weight)

        pos = nx.spring_layout(G)  # positions for all nodes

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=700)

        # edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

        # labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.axis("off")
        plt.show()

