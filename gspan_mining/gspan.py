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
    def deco(*args, **kwargs):
        timestamps = args[0].timestamps
        timestamps[func.__name__ + '_in'] = time.time()
        func(*args, **kwargs)
        timestamps[func.__name__ + '_out'] = time.time()
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
            frm, to, (vlb1, elb, vlb2) = dfsedge.frm, dfsedge.to, dfsedge.vevlb
            if vlb1 != VACANT_VERTEX_LABEL:
                g.add_vertex(frm, vlb1)
            if vlb2 != VACANT_VERTEX_LABEL:
                g.add_vertex(to, vlb2)
            g.add_edge(AUTO_EDGE_ID, frm, to, elb)
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
                  'min number of that.\nMax number of vertices is '
                  'set as infinite.')
            self._max_num_vertices = float('inf')
        self._report_df = pd.DataFrame()

    def time_stats(self):
        """Print the time statistics."""
        timestamps = self.timestamps
        for k, v in timestamps.items():
            if k.endswith('_in'):
                print(k[:-3] + ':', v)
                print(k[:-3] + '_time:', timestamps[k[:-3] + '_out'] - v)

    @record_timestamp
    def run(self):
        """Run the gSpan algorithm."""
        self._read_graphs()
        self._frequent_size1_subgraphs = self._get_frequent_size1_subgraphs()
        self._frequent_subgraphs += self._frequent_size1_subgraphs
        self._report_df = pd.DataFrame()
        for gid in sorted(self.graphs.keys()):
            self._DFScode = DFScode()
            self._projected = Projected()
            self._report_df = self._report_df.append(
                self._subgraph_mining(Projected(), gid),
                ignore_index=True
            )
        self._report_df['support'] = self._report_df['support'].astype(int)
        if self._where:
            self._report_df['where'] = self._report_df['where'].astype(str)
        self._report_df.sort_values(by=['support', 'where'],
                                    ascending=[False, True],
                                    inplace=True)
        return self._report_df

    def _read_graphs(self):
        """Read graphs from the specified file."""
        with codecs.open(self._database_file_name, 'r', 'utf-8') as f:
            lines = f.readlines()

        lines.append('t # -1\n')

        lines = [x.strip() for x in lines]
        current_gid = None
        vertices = []
        edges = []
        for line in lines:
            if line.startswith('t #'):
                if current_gid is not None:
                    self.graphs[current_gid] = Graph(current_gid,
                                                     vertices,
                                                     edges,
                                                     is_undirected=self._is_undirected,
                                                     eid_auto_increment=True)
                    if len(self.graphs) >= self._max_ngraphs:
                        break
                    vertices = []
                    edges = []
                current_gid = int(line.split()[-1])
            elif line.startswith('v '):
                vertices.append(line)
            elif line.startswith('e '):
                edges.append(line)

    def _get_frequent_size1_subgraphs(self):
        """Return frequent subgraphs of size 1."""
        single_edge_labels = collections.defaultdict(int)
        for gid in self.graphs.keys():
            g = self.graphs[gid]
            for e in g.edges.values():
                single_edge_labels[e.elb] += 1

        frequent_subgraphs = []
        for elb, support in single_edge_labels.items():
            if support >= self._min_support:
                dfscode = DFScode()
                dfscode.push_back(0, 1, (VACANT_VERTEX_LABEL, elb, VACANT_VERTEX_LABEL))
                frequent_subgraphs.append((dfscode, support))

        return frequent_subgraphs

    @record_timestamp
    def _subgraph_mining(self, projected):
        """Return the report dataframe of the projected frequent subgraphs."""
        self._support = len(projected)
        if self._support >= self._min_support:
            if self._DFScode.get_num_vertices() < self._min_num_vertices:
                return pd.DataFrame()

            if self._verbose:
                print('frequent subgraph:', self._DFScode, 'support:', self._support)

            self._frequent_subgraphs.append(
                (copy.deepcopy(self._DFScode), self._support)
            )
        if self._DFScode.get_num_vertices() == self._max_num_vertices:
            return pd.DataFrame()

        if self._where:
            projected = sorted(projected,
                               key=lambda p: (self.graphs[p.gid].name,
                                              p.edge.frm if p.edge else -1))

        history = History(None, None)
        for i, pdfs in enumerate(projected):
            gid = pdfs.gid
            if self._where:
                graph_name = self.graphs[gid].name
            else:
                graph_name = None
            if pdfs.edge:
                if pdfs.edge.frm != VACANT_VERTEX_LABEL:
                    vlb_fr, vlb_to = self.graphs[gid].vertices[pdfs.edge.frm], \
                                     self.graphs[gid].vertices[pdfs.edge.to]
                    elb = self.graphs[gid].edges[pdfs.edge.eid].elb
                    history.vertices_used[vlb_fr] = 1
                    history.vertices_used[vlb_to] = 1
                    history.edges_used[pdfs.edge.eid] = 1
                    self._DFScode.push_back(pdfs.edge.frm,
                                            pdfs.edge.to,
                                            (vlb_fr, elb, vlb_to))

            if self._verbose:
                print('projected graph:', i, 'graph id:', gid, 'graph name:', graph_name)
            self._projected = self._projected[:0]
            self._projected = self._projected + projected[i + 1:]

            # `get_candidates` might be empty when it is called for the first time,
            # since it is calculated for the first time.
            if pdfs.edge is None:
                current_candidates = self._get_candidates(gid, None, None)
            else:
                current_candidates = self._get_candidates(
                    gid, pdfs.edge.to, history
                )
            if self._verbose:
                print('current candidates:', len(current_candidates))
            for edge in current_candidates:
                if self._verbose:
                    print('candidate edge:', edge)
                new_dfscode = copy.deepcopy(self._DFScode)
                new_pdfsg = self._projected_graph(gid, edge)
                if new_pdfsg:
                    new_projected = copy.deepcopy(self._projected)
                    new_projected = new_projected.push_back(gid, edge, pdfs)
                    self._subgraph_mining(new_projected)

    def _get_candidates(self, gid, last_vertex, history):
        """Return candidate edges of a graph."""
        if last_vertex is None:
            vertex_iterator = sorted(
                list(self.graphs[gid].vertices.keys()))
        else:
            vertex_iterator = sorted(
                list(self.graphs[gid].vertices.keys()))[last_vertex + 1:]

        candidates = set()
        for frm in vertex_iterator:
            edges = self.graphs[gid].vertex[frm]
            if not edges:
                continue
            for edge in edges:
                if not history.has_vertex(edge.to):
                    candidates.add(edge)
        return candidates

    def _projected_graph(self, gid, edge):
        """Return projected graph from DFS code."""
        graph = self.graphs[gid]
        edge_label = graph.edges[edge.eid].elb
        last_vid = edge.to

        projected_edges = []
        for e in graph.vertex[last_vid]:
            if e.elb == edge_label:
                projected_edges.append(e)

        if not projected_edges:
            return None

        projected_graph = Projected()
        for projected_edge in projected_edges:
            projected_graph.push_back(gid, projected_edge, None)

        return projected_graph


def main(FLAGS=None):
    """Run gSpan."""

    if FLAGS is None:
        FLAGS, _ = parser.parse_known_args()

    if not os.path.exists(FLAGS.database_file_name):
        print('{} does not exist.'.format(FLAGS.database_file_name))
        return

    gs = gSpan(
        database_file_name=FLAGS.database_file_name,
        min_support=FLAGS.min_support,
        min_num_vertices=FLAGS.lower_bound_of_num_vertices,
        max_num_vertices=FLAGS.upper_bound_of_num_vertices,
        max_ngraphs=FLAGS.num_graphs,
        is_undirected=(not FLAGS.directed),
        verbose=FLAGS.verbose,
        visualize=FLAGS.plot,
        where=FLAGS.where
    )

    gs.run()
    gs.time_stats()
    return gs


if __name__ == '__main__':
    main()
