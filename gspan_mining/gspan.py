from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import copy
import itertools
import time

from .graph import AUTO_EDGE_ID
from .graph import Graph
from .graph import VACANT_GRAPH_ID
from .graph import VACANT_VERTEX_LABEL

import pandas as pd

def record_timestamp(func):
    """Record timestamp before and after call of `func`."""
    def deco(self):
        self.timestamps[func.__name__ + '_in'] = time.time()
        func(self)
        self.timestamps[func.__name__ + '_out'] = time.time()
    return deco


class DFSedge(object):
    """DFSedge class."""

    def __init__(self, frm, to, vevlb):
        """Initialize DFSedge instance."""
        self.frm = frm
        self.to = to
        self.vevlb = vevlb

    def __eq__(self, other):
        """Check equivalence of DFSedge."""
        return (self.frm == other.frm and
                self.to == other.to and
                self.vevlb == other.vevlb)

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __repr__(self):
        """Represent DFScode in string way."""
        return '(frm={}, to={}, vevlb={})'.format(
            self.frm, self.to, self.vevlb
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

    def push_back(self, frm, to, vevlb):
        """Update DFScode by adding one edge."""
        self.append(DFSedge(frm, to, vevlb))
        return self

    def to_graph(self, gid=VACANT_GRAPH_ID, is_undirected=True):
        """Construct a graph according to the dfs code."""
        g = Graph(gid,
                  is_undirected=is_undirected,
                  eid_auto_increment=True)
        for dfsedge in self:
            frm, to, (vlb1, elb, vlb2, weight) = dfsedge.frm, dfsedge.to, dfsedge.vevlb
            if vlb1 != VACANT_VERTEX_LABEL:
                g.add_vertex(frm, vlb1)
            if vlb2 != VACANT_VERTEX_LABEL:
                g.add_vertex(to, vlb2)
            g.add_edge(AUTO_EDGE_ID, frm, to, elb, weight)
        return g

    def from_graph(self, g):
        """Build DFScode from graph `g`."""
        raise NotImplementedError('Not implemented yet.')

    def build_rmpath(self):
        """Build right most path."""
        self.rmpath = list()
        old_frm = None
        for i in range(len(self) - 1, -1, -1):
            dfsedge = self[i]
            frm, to = dfsedge.frm, dfsedge.to
            if frm < to and (old_frm is None or to == old_frm):
                self.rmpath.append(i)
                old_frm = frm
        return self

    def get_num_vertices(self):
        """Return number of vertices in the corresponding graph."""
        return len(set(
            [dfsedge.frm for dfsedge in self] +
            [dfsedge.to for dfsedge in self]
        ))


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
        self.vertices_used = collections.defaultdict(int)
        self.edges_used = collections.defaultdict(int)
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
                 min_weighted_support=10,
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
        self._min_weighted_support = min_weighted_support
        self._min_num_vertices = min_num_vertices
        self._max_num_vertices = max_num_vertices
        self._DFScode = DFScode()
        self._support = 0
        self._frequent_size1_subgraphs = list()
        # Include subgraphs with
        # any num(but >= 2, <= max_num_vertices) of vertices.
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
        time_deltas = collections.defaultdict(float)
        for fn in func_names:
            time_deltas[fn] = round(
                self.timestamps[fn + '_out'] - self.timestamps[fn + '_in'],
                2
            )

        print('Read:\t{} s'.format(time_deltas['_read_graphs']))
        print('Mine:\t{} s'.format(time_deltas['run']))

    def run(self):
        """Run the gSpan algorithm."""
        self.timestamps['run_in'] = time.time()
        self._read_graphs()
        self._frequent_size1_subgraphs = self._find_frequent_size1_subgraphs()
        if self._verbose:
            print('Frequent subgraphs of size 1: {}'.format(
                len(self._frequent_size1_subgraphs)
            ))
        if self._visualize:
            self._visualize_patterns()
        self._frequent_subgraphs = copy.deepcopy(
            self._frequent_size1_subgraphs
        )
        for frequent_subgraph in self._frequent_subgraphs:
            dfscode = DFScode().push_back(
                frequent_subgraph[0],
                frequent_subgraph[1],
                frequent_subgraph[2]
            )
            # self._DFScode = copy.deepcopy(dfscode)
            # self._support = frequent_subgraph[3]
            self._subgraph_mining(dfscode, frequent_subgraph[3])
        if self._verbose:
            print('Frequent subgraphs (with full edges): {}'.format(
                len(self._frequent_subgraphs)
            ))
        self.timestamps['run_out'] = time.time()
        self._where = False
        if self._verbose:
            self.time_stats()
        if self._where:
            return self._frequent_subgraphs, self._report_df
        return self._frequent_subgraphs

    def _read_graphs(self):
        """Read graphs from the dataset file."""
        with codecs.open(self._database_file_name, 'r', 'utf-8') as f:
            lines = f.readlines()
        graph_cnt = -1
        for line in lines:
            if line.startswith('t'):
                graph_cnt += 1
                self.graphs[graph_cnt] = Graph(gid=graph_cnt,
                                                is_undirected=self._is_undirected)
            elif line.startswith('v'):
                _, vid, vlb = line.strip().split()
                vid, vlb = int(vid), int(vlb)
                self.graphs[graph_cnt].add_vertex(vid, vlb)
            elif line.startswith('e'):
                _, frm, to, elb, weight = line.strip().split()
                frm, to, elb, weight = int(frm), int(to), int(elb), int(weight)
                self.graphs[graph_cnt].add_edge(AUTO_EDGE_ID, frm, to, elb, weight)
        self.graphs = {gid: g for gid, g in self.graphs.items() if len(g.vertices) >= self._min_num_vertices}
        return self

    def _find_frequent_size1_subgraphs(self):
        """Find frequent subgraphs with size 1."""
        vertex_labels = collections.defaultdict(int)
        edge_labels = collections.defaultdict(int)
        for gid, g in self.graphs.items():
            for vid, vertex in g.vertices.items():
                vertex_labels[vertex.vlb] += 1
                for edge in vertex.edges.values():
                    edge_labels[edge.elb] += 1
        vertex_labels = {k: v for k, v in vertex_labels.items() if v >= self._min_support}
        edge_labels = {k: v for k, v in edge_labels.items() if v >= self._min_support}
        frequent_size1_subgraphs = list()
        for gid, g in self.graphs.items():
            for vid, vertex in g.vertices.items():
                if vertex.vlb in vertex_labels:
                    frequent_size1_subgraphs.append((VACANT_VERTEX_ID,
                                                     vid,
                                                     (vertex.vlb,
                                                      VACANT_EDGE_LABEL,
                                                      VACANT_VERTEX_LABEL, 1),
                                                     vertex_labels[vertex.vlb]))
                for edge in vertex.edges.values():
                    if edge.elb in edge_labels:
                        frequent_size1_subgraphs.append((edge.frm,
                                                         edge.to,
                                                         (VACANT_VERTEX_LABEL,
                                                          edge.elb,
                                                          VACANT_VERTEX_LABEL,
                                                          edge.weight),
                                                         edge_labels[edge.elb]))
        frequent_size1_subgraphs = sorted(
            frequent_size1_subgraphs,
            key=lambda x: (x[3], x[0], x[1])
        )
        return frequent_size1_subgraphs

    def _subgraph_mining(self, dfscode, support):
        """Subgraph mining."""
        if dfscode.get_num_vertices() > self._max_num_vertices:
            return self
        if support < self._min_support:
            return self
        if dfscode.get_num_vertices() >= self._min_num_vertices:
            self._report(dfscode, support)
        if dfscode.get_num_vertices() == self._max_num_vertices:
            return self
        for vid, vertex_label_support in self._support_vertex_labels(dfscode).items():
            for edge in self._valid_edge_labels(dfscode, vid):
                new_dfscode = self._embed(dfscode, vid, vertex_label_support,
                                           edge)
                new_support = min(
                    vertex_label_support,
                    self._support_edge_labels(new_dfscode, vid, edge)
                )
                self._subgraph_mining(new_dfscode, new_support)
        return self

    def _embed(self, dfscode, vid, vertex_label_support, edge):
        """Embedding one edge into dfs code."""
        new_dfscode = copy.deepcopy(dfscode)
        old_vertex = new_dfscode[-1].to
        if self._where:
            old_fragment = dfscode
        new_dfscode.push_back(
            old_vertex,
            vid,
            (VACANT_VERTEX_LABEL,
             edge[1],
             VACANT_VERTEX_LABEL,
             edge[3])
        )
        if self._where:
            self._find_instances(new_dfscode, vertex_label_support)
        return new_dfscode

    def _support_vertex_labels(self, dfscode):
        """Calculate the support of each vertex label."""
        if not dfscode:
            return collections.defaultdict(int)
        prev_vertex = dfscode[-1].to
        vertex_labels_support = collections.defaultdict(int)
        for gid, g in self.graphs.items():
            for vertex_id, vertex in g.vertices.items():
                if vertex_id <= prev_vertex:
                    continue
                for edge in vertex.edges.values():
                    if edge.to == prev_vertex:
                        vertex_labels_support[vertex.vlb] += 1
        return vertex_labels_support

    def _valid_edge_labels(self, dfscode, vid):
        """Generate valid edge labels for vid."""
        valid_edge_labels = list()
        for gid, g in self.graphs.items():
            if gid < dfscode[0].frm or gid > dfscode[0].to:
                continue
            if vid not in g.vertices:
                continue
            for edge in g.vertices[vid].edges.values():
                if edge.to < dfscode[-1].to:
                    continue
                if self._where:
                    if not self._DFScodeMatchVertex(gid, dfscode):
                        continue
                valid_edge_labels.append((edge.elb, edge.to, edge.weight))
        return valid_edge_labels

    def _support_edge_labels(self, dfscode, vid, edge):
        """Calculate the support of edge labels."""
        if not dfscode:
            return float('inf')
        prev_vertex = dfscode[-1].to
        if self._where:
            prev_vertex = dfscode[-1].frm
        edge_labels_support = collections.defaultdict(int)
        for gid, g in self.graphs.items():
            if gid < dfscode[0].frm or gid > dfscode[0].to:
                continue
            if vid not in g.vertices:
                continue
            for edge in g.vertices[vid].edges.values():
                if edge.to < prev_vertex:
                    continue
                if self._where:
                    if not self._DFScodeMatchVertex(gid, dfscode):
                        continue
                edge_labels_support[edge.elb] += 1
        if self._where:
            return sum(edge_labels_support.values())
        return min(edge_labels_support.values())

    def _report(self, dfscode, support):
        """Report one frequent subgraph."""
        new_graph = dfscode.to_graph(is_undirected=self._is_undirected)
        if self._where:
            self._frequent_subgraphs.append(
                (new_graph,
                 dfscode,
                 support,
                 self._where_edges)
            )
            self._where_edges = None
            return self
        self._frequent_subgraphs.append((new_graph, dfscode, support))
        if not self._verbose:
            return self
        if len(self._frequent_subgraphs) % 100 == 0:
            print(len(self._frequent_subgraphs))
        return self

    def _visualize_patterns(self):
        """Visualize frequent patterns."""
        for frequent_subgraph, _, _ in self._frequent_size1_subgraphs:
            frequent_subgraph.plot()
        return self

    def _find_instances(self, dfscode, vertex_label_support):
        """Find instances of current frequent subgraph."""
        self._where_edges = collections.defaultdict(list)
        for gid, g in self.graphs.items():
            if gid < dfscode[0].frm or gid > dfscode[0].to:
                continue
            if not self._DFScodeMatchVertex(gid, dfscode):
                continue
            history = History(g, None)
            for vid, vertex in g.vertices.items():
                if vertex.vlb != dfscode[0].vevlb[0]:
                    continue
                self._DFScodeMatchVertexFrom(history,
                                             g,
                                             vid,
                                             dfscode,
                                             vertex_label_support,
                                             1)

    def _DFScodeMatchVertexFrom(self, history, g, vid, dfscode,
                                vertex_label_support, edge_idx):
        """Check if dfscode can be matched starting from vid."""
        if edge_idx == len(dfscode):
            return self
        old_vertex, new_vertex, (vlb1, elb, vlb2, weight) = dfscode[edge_idx]
        if self._where:
            if new_vertex < vid or not self._DFScodeMatchVertex(g.gid, dfscode):
                return self
        for edge in g.vertices[vid].edges.values():
            if edge.to < old_vertex:
                continue
            if (edge.elb == elb and
                    not history.has_edge(edge.eid) and
                    not history.has_vertex(edge.to) and
                    not self._DFScodeHasDuplication(dfscode, vid, edge)):
                self._where_edges[dfscode[edge_idx - 1].frm, vid].append(
                    (edge.frm, edge.to, edge.elb, edge.weight)
                )
                history = History(g, history)
                history.vertices_used[vid] = 1
                history.edges_used[edge.eid] = 1
                self._DFScodeMatchVertexFrom(history,
                                             g,
                                             edge.to,
                                             dfscode,
                                             vertex_label_support,
                                             edge_idx + 1)
        return self

    def _DFScodeHasDuplication(self, dfscode, vid, edge):
        """Check if there are duplications in DFS code."""
        frm, to, (vlb1, elb, vlb2, weight) = dfscode[-1]
        if self._is_undirected:
            if to < frm:
                return False
            if to == frm:
                return True
        if self._is_undirected:
            if (frm == vid and
                    (vid, edge.to, edge.elb, edge.weight) in dfscode):
                return True
            if (to == vid and
                    (edge.frm, vid, edge.elb, edge.weight) in dfscode):
                return True
        if (frm == vid and
                to == edge.to and
                elb == edge.elb):
            return True
        return False

    def _DFScodeMatchVertex(self, gid, dfscode):
        """Check if DFS code can be matched to gid."""
        # Check graph gid has enough vertices.
        min_vid = min(dfsedge.frm for dfsedge in dfscode)  # min vid in DFS code.
        if len(self.graphs[gid].vertices) < min_vid:
            return False
        # DFS code needs at least one edge.
        if len(dfscode) == 1:
            return True
        return self._DFScodeMatchEdge(gid, dfscode)

    def _DFScodeMatchEdge(self, gid, dfscode):
        """Check if DFS code can be matched to gid."""
        first_edge = dfscode[0]
        if self._is_undirected and first_edge.frm > first_edge.to:
            first_edge.frm, first_edge.to = first_edge.to, first_edge.frm
        for vid, vertex in self.graphs[gid].vertices.items():
            for edge in vertex.edges.values():
                if edge.to < first_edge.to:
                    continue
                if edge.elb != first_edge.vevlb[1]:
                    continue
                if edge.to > first_edge.to:
                    continue
                if edge.to == first_edge.to:
                    if (edge.elb, edge.weight) != (first_edge.vevlb[1], first_edge.vevlb[3]):
                        continue
                if self._DFScodeMatchEdgeFrom(
                        DFScode().push_back(first_edge.frm,
                                            vid,
                                            first_edge.vevlb),
                        gid, dfscode[1:]):
                    return True
        return False

    def _DFScodeMatchEdgeFrom(self, dfscode, gid, rest_dfscode):
        """Check if DFS code can be matched to gid."""
        if not rest_dfscode:
            return True
        prev_dfsedge, rest_dfscode = rest_dfscode[0], rest_dfscode[1:]
        old_vertex, new_vertex, (vlb1, elb, vlb2, weight) = prev_dfsedge
        if self._is_undirected:
            if old_vertex > new_vertex:
                old_vertex, new_vertex = new_vertex, old_vertex
        if self._is_undirected:
            if old_vertex == new_vertex:
                for vid, vertex in self.graphs[gid].vertices.items():
                    if vid < old_vertex:
                        continue
                    for edge in vertex.edges.values():
                        if edge.to < old_vertex:
                            continue
                        if edge.to == old_vertex:
                            if (edge.elb, edge.weight) != (elb, weight):
                                continue
                        if (self._DFScodeMatchEdgeFrom(
                                DFScode().push_back(
                                    old_vertex,
                                    vid,
                                    (VACANT_VERTEX_LABEL,
                                     elb,
                                     VACANT_VERTEX_LABEL,
                                     weight)
                                ),
                                gid,
                                rest_dfscode)):
                            return True
            for vid, vertex in self.graphs[gid].vertices.items():
                if vid < old_vertex:
                    continue
                for edge in vertex.edges.values():
                    if edge.to < new_vertex:
                        continue
                    if edge.elb != elb:
                        continue
                    if edge.to > new_vertex:
                        continue
                    if edge.to == new_vertex:
                        if (edge.elb, edge.weight) != (elb, weight):
                            continue
                    if (self._DFScodeMatchEdgeFrom(
                            DFScode().push_back(
                                old_vertex,
                                new_vertex,
                                (VACANT_VERTEX_LABEL,
                                 edge.elb,
                                 VACANT_VERTEX_LABEL,
                                 edge.weight)
                            ),
                            gid,
                            rest_dfscode)):
                        return True
        for vid, vertex in self.graphs[gid].vertices.items():
            if vid < old_vertex:
                continue
            for edge in vertex.edges.values():
                if edge.to < new_vertex:
                    continue
                if edge.elb != elb:
                    continue
                if edge.to > new_vertex:
                    continue
                if edge.to == new_vertex:
                    if (edge.elb, edge.weight) != (elb, weight):
                        continue
                if (self._DFScodeMatchEdgeFrom(
                        DFScode().push_back(
                            old_vertex,
                            new_vertex,
                            (VACANT_VERTEX_LABEL,
                             edge.elb,
                             VACANT_VERTEX_LABEL,
                             edge.weight)
                        ),
                        gid,
                        rest_dfscode)):
                    return True
        return False


def main(database_file_name,
         min_support=10,
         min_weighted_support=10,
         min_num_vertices=1,
         max_num_vertices=float('inf'),
         max_ngraphs=float('inf'),
         is_undirected=True,
         verbose=False,
         visualize=False,
         where=False):
    """Run gSpan."""
    gspan = gSpan(database_file_name,
                  min_support=min_support,
                  min_weighted_support=min_weighted_support,
                  min_num_vertices=min_num_vertices,
                  max_num_vertices=max_num_vertices,
                  max_ngraphs=max_ngraphs,
                  is_undirected=is_undirected,
                  verbose=verbose,
                  visualize=visualize,
                  where=where)
    return gspan.run()

if __name__ == '__main__':
    main()
