from collections import defaultdict
import copy
import itertools
import time
import numpy as np
import math
import pandas as pd
import codecs



from .graph import AUTO_EDGE_ID
from .graph import Graph
from .graph import VACANT_GRAPH_ID
from .graph import VACANT_VERTEX_LABEL

def record_timestamp(func):
    """Record timestamp before and after call of `func`."""
    def deco(self):
        self.timestamps[func.__name__ + '_in'] = time.time()
        func(self)
        self.timestamps[func.__name__ + '_out'] = time.time()
    return deco

class DFSedge(object):
    """DFSedge class."""

    def __init__(self, frm, to, vevlb, weight=1):
        """Initialize DFSedge instance."""
        self.frm = frm
        self.to = to
        self.vevlb = vevlb
        self.weight = weight

    def __eq__(self, other):
        """Check equivalence of DFSedge."""
        return (self.frm == other.frm and
                self.to == other.to and
                self.vevlb == other.vevlb and
                self.weight == other.weight)

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __repr__(self):
        """Represent DFScode in string way."""
        return '(frm={}, to={}, vevlb={}, weight={})'.format(
            self.frm, self.to, self.vevlb, self.weight
        )

class DFScode(list):
    """DFScode is a list of DFSedge."""

    def __init__(self):
        """Initialize DFScode."""
        self.rmpath = list()

    def __eq__(self, other):
        """Check equivalence of DFScode."""
        la, lb = len(self), len(other)
        if la != lb:
            return False
        for i in range(la):
            if self[i] != other[i]:
                return False
        return True

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __repr__(self):
        """Represent DFScode in string way."""
        return ''.join(['[', ','.join(
            [str(dfsedge) for dfsedge in self]), ']']
        )

    def push_back(self, frm, to, vevlb, weight=1):
        """Update DFScode by adding one edge."""
        self.append(DFSedge(frm, to, vevlb, weight))
        return self

    def to_graph(self, gid=VACANT_GRAPH_ID, is_undirected=True):
        """Construct a graph according to the dfs code."""
        g = Graph(gid,
                  is_undirected=is_undirected,
                  eid_auto_increment=True)
        for dfsedge in self:
            frm, to, (vlb1, elb, vlb2), weight = dfsedge.frm, dfsedge.to, dfsedge.vevlb, dfsedge.weight
            if vlb1 != VACANT_VERTEX_LABEL:
                g.add_vertex(frm, vlb1)
            if vlb2 != VACANT_VERTEX_LABEL:
                g.add_vertex(to, vlb2)
            g.add_edge(AUTO_EDGE_ID, frm, to, elb, weight)
        return g

class PDFS(object):
    """PDFS class."""

    def __init__(self, gid=VACANT_GRAPH_ID, edge=None, prev=None):
        """Initialize PDFS instance."""
        self.gid = gid
        self.edge = edge
        self.prev = prev

class Projected(list):
    """Projected is a list of PDFS.

    Each element of Projected is a projection one frequent graph in one
    original graph.
    """

    def __init__(self):
        """Initialize Projected instance."""
        super(Projected, self).__init__()

    def push_back(self, gid, edge, prev):
        """Update this Projected instance."""
        self.append(PDFS(gid, edge, prev))
        return self

class History(object):
    """History class."""

    def __init__(self, g, pdfs):
        """Initialize History instance."""
        super(History, self).__init__()
        self.edges = list()
        self.vertices_used = defaultdict(int)
        self.edges_used = defaultdict(int)
        if pdfs is None:
            return
        while pdfs:
            e = pdfs.edge
            self.edges.append(e)
            (self.vertices_used[e.frm],
             self.vertices_used[e.to],
             self.edges_used[e.eid]) = 1, 1, 1

            pdfs = pdfs.prev
        self.edges = self.edges[::-1]

    def has_vertex(self, vid):
        """Check if the vertex with vid exists in the history."""
        return self.vertices_used[vid] == 1

    def has_edge(self, eid):
        """Check if the edge with eid exists in the history."""
        return self.edges_used[eid] == 1

class gSpan(object):
    """`gSpan` algorithm."""

    def __init__(self,
                 database_file_name,
                 min_support=10,
                 min_num_vertices=1,
                 max_num_vertices=float('inf'),
                 max_ngraphs=float('inf'),
                 is_undirected=True,
                 verbose=False,
                 visualize=False,
                 where=False):
        """Initialize gSpan instance."""
        self._database_file_name = database_file_name
        self.graphs = dict()
        self._max_ngraphs = max_ngraphs
        self._is_undirected = is_undirected
        self._min_support = min_support
        self._min_num_vertices = min_num_vertices
        self._max_num_vertices = max_num_vertices
        self._DFScode = DFScode()
        self._support = 0
        self._frequent_size1_subgraphs = list()
        self._frequent_subgraphs = list()
        self._counter = itertools.count()
        self._verbose = verbose
        self._visualize = visualize
        self._where = where
        self.timestamps = dict()
        if self._max_num_vertices < self._min_num_vertices:
            print('Max number of vertices can not be smaller than '
                  'min number of that.\n'
                  'Set max_num_vertices = min_num_vertices.')
            self._max_num_vertices = self._min_num_vertices
        self._report_df = pd.DataFrame()

    def time_stats(self):
        """Print stats of time."""
        func_names = ['_read_graphs', 'run']
        time_deltas = defaultdict(float)
        for fn in func_names:
            time_deltas[fn] = round(
                self.timestamps[fn + '_out'] - self.timestamps[fn + '_in'],
                2
            )

        print('Read:\t{} s'.format(time_deltas['_read_graphs']))
        print('Mine:\t{} s'.format(
            time_deltas['run'] - time_deltas['_read_graphs']))
        print('Total:\t{} s'.format(time_deltas['run']))

        return self

    @record_timestamp
    def _read_graphs(self):
        self.graphs = dict()
        with codecs.open(self._database_file_name, 'r', 'utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            tgraph, graph_cnt = None, 0
            for i, line in enumerate(lines):
                cols = line.split(' ')
                if cols[0] == 't':
                    if tgraph is not None:
                        self.graphs[graph_cnt] = tgraph
                        graph_cnt += 1
                        tgraph = None
                    if cols[-1] == '-1' or graph_cnt >= self._max_ngraphs:
                        break
                    tgraph = Graph(graph_cnt,
                                   is_undirected=self._is_undirected,
                                   eid_auto_increment=True)
                elif cols[0] == 'v':
                    tgraph.add_vertex(cols[1], cols[2])
                elif cols[0] == 'e':
                    tgraph.add_edge(AUTO_EDGE_ID, cols[1], cols[2], cols[3], int(cols[4]))
            # adapt to input files that do not end with 't # -1'
            if tgraph is not None:
                self.graphs[graph_cnt] = tgraph
        return self

    @record_timestamp
    def _generate_1edge_frequent_subgraphs(self):
        vlb_counter = defaultdict(int)
        vevlb_counter = defaultdict(int)
        vlb_counted = set()
        vevlb_counted = set()
        for g in self.graphs.values():
            for v in g.vertices.values():
                if (g.gid, v.vlb) not in vlb_counted:
                    vlb_counter[v.vlb] += 1
                vlb_counted.add((g.gid, v.vlb))
                for to, e in v.edges.items():
                    vlb1, vlb2 = v.vlb, g.vertices[to].vlb
                    if self._is_undirected and vlb1 > vlb2:
                        vlb1, vlb2 = vlb2, vlb1
                    if (g.gid, (vlb1, e.elb, vlb2)) not in vevlb_counter:
                        vevlb_counter[(vlb1, e.elb, vlb2)] += 1
                    vevlb_counted.add((g.gid, (vlb1, e.elb, vlb2)))
        # add frequent vertices.
        for vlb, cnt in vlb_counter.items():
            if cnt >= self._min_support:
                g = Graph(gid=next(self._counter),
                          is_undirected=self._is_undirected)
                g.add_vertex(0, vlb)
                self._frequent_size1_subgraphs.append(g)
                if self._min_num_vertices <= 1:
                    self._report_size1(g, support=cnt)
            else:
                continue
        if self._min_num_vertices > 1:
            self._counter = iter(range(1, 1000000))

    @record_timestamp
    def run(self):
        """Run the gSpan algorithm."""
        self._read_graphs()
        self._generate_1edge_frequent_subgraphs()
        if self._max_num_vertices < 2:
            return
        root = defaultdict(Projected)
        for gid, g in self.graphs.items():
            for vid, v in g.vertices.items():
                edges = self._get_forward_root_edges(g, vid)
                for e in edges:
                    root[(v.vlb, e.elb, g.vertices[e.to].vlb)].append(
                        PDFS(gid, e, None)
                    )

        for vevlb, projected in root.items():
            self._DFScode.append(DFSedge(0, 1, vevlb))
            self._subgraph_mining(projected)
            self._DFScode.pop()

    def _get_support(self, projected):
        return len(set([pdfs.gid for pdfs in projected]))

    def _report_size1(self, g, support):
        g.display()
        print('\nSupport: {}'.format(support))
        print('\n-----------------\n')

    def _report(self, projected):
        self._frequent_subgraphs.append(copy.copy(self._DFScode))
        if self._DFScode.get_num_vertices() < self._min_num_vertices:
            return
        g = self._DFScode.to_graph(gid=next(self._counter),
                                   is_undirected=self._is_undirected)
        display_str = g.display()
        print('\nSupport: {}'.format(self._support))
        if self._visualize:
            g.plot()
        if self._where:
            print('where: {}'.format(list(set([p.gid for p in projected]))))
        print('\n-----------------\n')

    def _get_forward_root_edges(self, g, frm):
        result = []
        v_frm = g.vertices[frm]
        for to, e in v_frm.edges.items():
            if (not self._is_undirected) or v_frm.vlb <= g.vertices[to].vlb:
                result.append(e)
        return result

    def _get_backward_edge(self, g, e1, e2, history):
        if self._is_undirected and e1 == e2:
            return None
        for to, e in g.vertices[e2.to].edges.items():
            if history.has_edge(e.eid) or e.to != e1.to:
                continue
            if (not self._is_undirected) or (
                    g.vertices[e1.frm].vlb <= g.vertices[e.to].vlb):
                return e
        return None

    def _is_min(self, projected):
        return self._get_support(projected) >= self._min_support

    def _subgraph_mining(self, projected):
        if not projected:
            return
        history = History(self.graphs[projected[0].gid], projected[0])
        support = self._get_support(projected)
        if support >= self._min_support:
            self._report(projected)
        if not self._is_min(projected):
            return
        if not self._can_add_vertex(history):
            return
        forward_root_edges = []
        for pdfs in projected:
            for frm, v in self.graphs[pdfs.gid].vertices.items():
                if not history.has_vertex(frm):
                    forward_root_edges += [
                        (frm, e) for e in self._get_forward_root_edges(
                            self.graphs[pdfs.gid], frm)]
        for frm, e in forward_root_edges:
            new_history = copy.deepcopy(history)
            g = self._DFScode.to_graph()
            frm_vlb = g.vertices[e.frm].vlb
            to_vlb = g.vertices[e.to].vlb
            elb = e.elb
            if self._is_undirected and frm_vlb > to_vlb:
                frm_vlb, to_vlb = to_vlb, frm_vlb
            vevlb = (frm_vlb, elb, to_vlb)
            self._DFScode.append(DFSedge(frm, len(self._DFScode) + 1, vevlb))
            if not self._can_add_edge(new_history):
                self._DFScode.pop()
                continue
            extensions = []
            backward_edges = []
            for pdfs in projected:
                if pdfs.edge is None or pdfs.edge.to != frm:
                    continue
                for prev in pdfs.prev:
                    if self._is_forward_edge(prev.edge):
                        extensions.append(pdfs)
                    else:
                        backward_edges.append(pdfs)
            if extensions:
                self._subgraph_mining(extensions)
            if backward_edges:
                if self._max_num_vertices > self._DFScode.get_num_vertices():
                    if self._DFScode.get_num_vertices() >= self._min_num_vertices - 1:
                        self._subgraph_mining(backward_edges)
            self._DFScode.pop()

    def _can_add_vertex(self, history):
        if self._max_num_vertices <= self._DFScode.get_num_vertices():
            return False
        if not history.edges:
            return True
        if self._DFScode.get_num_vertices() == 1:
            return True
        return self._DFScode.get_num_vertices() < self._max_num_vertices

    def _can_add_edge(self, history):
        if self._DFScode.get_num_edges() < self._DFScode.get_num_vertices() - 1:
            return False
        if not history.edges:
            return True
        for i in range(len(self._DFScode)):
            if not self._DFScode[i].eq(history.edges[i]):
                return False
        return True

    def _is_forward_edge(self, edge):
        if not edge:
            return False
        if edge.eid != AUTO_EDGE_ID:
            return False
        return True

    def print_stats(self):
        """Print stats."""
        print('Minimum Support: {}'.format(self._min_support))
        print('Minimum Number of Vertices: {}'.format(self._min_num_vertices))
        print('Maximum Number of Vertices: {}'.format(self._max_num_vertices))
        print('Maximum Number of Graphs: {}'.format(self._max_ngraphs))
        print('Number of Graphs: {}'.format(len(self.graphs)))
        print('Number of Frequent Subgraphs: {}'.format(
            len(self._frequent_subgraphs)))
        return self
