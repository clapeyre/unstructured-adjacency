"""
Efficient implementation of unstructured mesh operations

Created Feb 8th, 2018 by C. Lapeyre (lapeyre@cerfacs.fr)
"""
from __future__ import print_function
import numpy as np
from scipy import sparse


class UnstructuredAdjacency(object):
    """Efficient scipy implementation of unstructured mesh adjacency ops

    The connectivity is stored in a sparse adjencency matrix A of shape
    (nnode, ncell). The gather operation on vector X (nnode) yields the
    scattered vector Y (ncell), and the scatter operation yields the
    filtered vector X' (nnode). This writes:
        Y  = 1/nvert . A . X
        X' = 1/bincount . A^t . Y
    where ^t is the transpose operation.

    The gatter-scatter operation resulting in filtering X can be performed
    efficiently by storing:
        F  = 1/bincount . A^t . 1/nvert . A
        X' = F . X
    """
    def __init__(self, connectivity):
        """Store connectivity as adjacency matrix, extract global parameters

        connectivity: a dictionary-like object with
            {number_of_vertex: connectivity}
          where:
            - number_of_vertex : an integer (number of vertex per element)
            - connectivity     : a list of node indices describing elements of
                                 length number_of_vertex * number_of_cells
          One entry is expected per element type. Ex:
              {3: [triangle_cells_node_list],
               4: [quad_cells_node_list]}
        """
        self._connectivity = connectivity
        concat = np.hstack(connectivity.values())
        self.nnode = np.unique(concat).size
        min_node_number = concat.min()
        # Ensure node numbering starts at 0
        self._connectivity = {nvert: conn - min_node_number
                              for nvert, conn in self._connectivity.items()}
        self.ncell_per_nvert = {nvert: conn.size / nvert
                                for nvert, conn in self._connectivity.items()}
        self._bincount = np.bincount(concat - min_node_number)
        self._adjacency_matrix = self._make_adjacency()

    @property
    def ncell(self):
        """Total number of cells"""
        return sum(self.ncell_per_nvert.values())

    def _make_adjacency(self, weights=None):
        """Create adjacency matrix by reading connectivity"""
        if weights is None:
            weights = {nvert: np.ones(self._connectivity[nvert].shape)
                       for nvert in self._connectivity}
        adjacency = []
        for nvert in sorted(self._connectivity):
            conn = self._connectivity[nvert]
            ncell = self.ncell_per_nvert[nvert]
            indptr = np.arange(ncell + 1) * nvert
            adjacency.append(sparse.csr_matrix((weights[nvert], conn, indptr),
                                               shape=(ncell, self.nnode)))
        return sparse.vstack(adjacency)

    @property
    def _node2cell_matrix(self):
        """The (ncell, nnode) matrix"""
        node_weights = {nvert:
                        np.ones(self.ncell_per_nvert[nvert] * nvert) / nvert
                        for nvert in sorted(self.ncell_per_nvert)}
        return self._make_adjacency(weights=node_weights)

    @property
    def _cell2node_matrix(self):
        """The (nnode, ncell) matrix"""
        mat = self._make_adjacency().T
        weights = sparse.diags(1. / self._bincount)
        return weights * mat

    def get_node2cell(self):
        """Return the node2cell function"""
        return lambda node_vector: self._node2cell_matrix * node_vector

    def get_cell2node(self):
        """Return the cell2node function"""
        return lambda cell_vector: self._cell2node_matrix * cell_vector

    def get_filter(self, times=1):
        """Return the full gather + scatter filter operation

        If you need to perform the operation N times, you can use the
        times attribute.

        """
        gatherscatter = self._cell2node_matrix * self._node2cell_matrix

        def filt(node_vector):
            """Function to return by get_filter method"""
            for _ in range(times):
                node_vector = gatherscatter * node_vector
            return node_vector
        return filt
