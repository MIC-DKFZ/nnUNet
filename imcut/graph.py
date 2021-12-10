#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

# import os.path as op

logger = logging.getLogger(__name__)
import numpy as nm
import numpy as np
import copy
import time
import cython

# from io import open

# TODO zpětná indexace původních pixelů (v add_nodes())
# TODO nastavení velikosti bloku (v sr_tab)
# TODO funguje to ve 3D?
# TODO možnost vypínání zápisu do VTK (mjirik)
# TODO možnost kontroly jmena souborů do VTK(mjirik)
#
# TODO Jeden nový uzel reprezentuje více voxelů v původním obrázku
# TODO indexy vytvářet ve split_voxel()?
# TODO co je igrp? split_voxel()
# TODO Uzly se jen přidávají, při odběru se nastaví flag na False? Dořešuje se to ve finish()
# TODO co je ndid ve split_voxel()
# TODO Přidat relabeling ve finish
# TODO zápis do VTK souboru? line 381, #  f = open(fname, 'w') # UnicodeDecodeError: 'ascii' codec can't decode byte 0xcd in position 0: ordinal not in range(128)
#
# data = nm.array([[0,0,0],
#                  [0,1,1],
#                  [0,1,1],
#                  [1,0,1]])


class Graph(object):

    def __init__(
            self,
            data,
            voxelsize,
            grid_function=None,
            nsplit=3,
            compute_msindex=True,
            edge_weight_table=None,
            compute_low_nodes_index=True,
    ):
        """

        :param data:
        :param voxelsize:
        :param grid_function: '2d' or 'nd'. Use '2d' for former implementation
        :param nsplit: size of low resolution block
        :param compute_msindex: compute indexes of nodes arranged in a ndarray with the same shape as higres image
        :param edge_weight_table: ndarray with size 2 * self.img.ndims. First axis describe whether is the edge
        between lowres(0) or highres(1) or hight-low (1) voxels. Second axis describe edge direction (edge axis).
        """
        # same dimension as data
        self.start_time = time.time()
        self.voxelsize = nm.asarray(voxelsize)
        # always 3D
        self.voxelsize3 = np.zeros([3])
        self.voxelsize3[: len(voxelsize)] = voxelsize

        self.data = np.asarray(data)
        if self.voxelsize.size != len(data.shape):
            logger.error("Datashape should be the same as voxelsize")
            raise ValueError("Datashape should be the same as voxelsize")
        self._edge_weight_table = edge_weight_table
        self.compute_low_node_inverse = compute_low_nodes_index
        # estimate maximum node number as number of lowres nodes + number of higres nodes + (nsplit - 1)^dim
        # 2d (nsplit, req) 2:3, 3:8, 4:12
        # 3D
        number_of_resized_nodes = np.count_nonzero(self.data)
        # self.ndmax_debug = data.size + number_of_resized_nodes* np.power(nsplit, self.data.ndim)
        self.ndmax = data.size + number_of_resized_nodes * np.power(
            nsplit, self.data.ndim
        )
        # init nodes
        self.nnodes = 0
        self.lastnode = 0
        self.nodes = nm.zeros((self.ndmax, 3), dtype=nm.float32)
        # node_flag: if true, this node is used in final output
        self.node_flag = nm.zeros((self.ndmax,), dtype=nm.bool)

        # init edges
        # estimate maximum number of dges as number of nodes multiplied by number of directions
        # the rest part is
        # if self.data.ndim == 2:
        #     edmax_rest = (9 * nsplit - 12)
        # elif self.data.ndim == 3:
        #     # edmax_rest = (9 * nsplit - 12) + 8 * nsplit**2 - 9 * nsplit - 35
        #     edmax_rest = 8 * nsplit**2 - 47
        # else:
        #     # this is not so efficient but should work
        #     edmax_rest = 9 * nsplit**(self.data.ndim - 1)
        # number of edges from every node
        edmax_from_node = self.data.ndim * self.ndmax
        # edges from lowres node to higres node from all directions
        edmax_into_node = (
                number_of_resized_nodes * self.data.ndim * nsplit ** (self.data.ndim - 1)
        )
        self.edmax = edmax_from_node + edmax_into_node
        edmax = self.edmax
        # self.edmax_debug = edmax_from_node + edmax_into_node
        self.nedges = 0
        self.lastedge = 0
        eddtype = get_efficient_signed_int_type(self.ndmax)
        self.edges = -nm.ones((edmax, 2), dtype=eddtype)
        # edge_flag: if true, this edge is used in final output
        self.edge_flag = nm.zeros((edmax,), dtype=nm.bool)
        # TODO Just trying new time reduction without where()
        # self.edge_flag_idx = []
        self.edge_dir = nm.zeros((edmax,), dtype=nm.int8)
        if self._edge_weight_table is not None:
            # dtype is given by graph-cut
            self.edges_weights = nm.zeros((edmax,), dtype=nm.int16)
        # list of edges on low resolution
        edgrdtype = get_efficient_signed_int_type(edmax)
        self.edge_group = -nm.ones((edmax,), dtype=edgrdtype)
        self.nsplit = nsplit
        self.compute_msindex = compute_msindex
        # indexes of nodes arranged in ndimage
        self.msinds = None
        if grid_function in (None, "nd", "ND"):
            self.gen_grid_fcn = gen_grid_nd
        elif grid_function in ("2d", "2D"):
            self.gen_grid_fcn = gen_grid_2d
        else:
            self.gen_grid_fcn = grid_function

        self._tile_shape = tuple(np.tile(nsplit, self.data.ndim))
        self.srt = SRTab()
        self.cache = {}
        self.stats = {}
        self.stats["t graph low"] = 0
        self.stats["t graph high"] = 0
        self.stats["t split 01"] = 0
        self.stats["t split 02"] = 0
        self.stats["t split 03"] = 0
        self.stats["t split 04"] = 0
        self.stats["t split 05"] = 0
        self.stats["t split 06"] = 0
        self.stats["t split 07"] = 0
        self.stats["t split 08"] = 0
        self.stats["t split 081"] = 0
        self.stats["t split 082"] = 0
        self.stats["t split 0821"] = 0
        self.stats["t split 09"] = 0
        self.stats["t split 10"] = 0
        self.stats["t graph 01"] = time.time() - self.start_time

    def add_nodes(self, coors, node_low_or_high=None):
        """
        Add new nodes at the end of the list.
        """
        last = self.lastnode
        if type(coors) is nm.ndarray:
            if len(coors.shape) == 1:
                coors = coors.reshape((1, coors.size))

            nadd = coors.shape[0]
            idx = slice(last, last + nadd)
        else:
            nadd = 1
            idx = self.lastnode
        right_dimension = coors.shape[1]
        self.nodes[idx, :right_dimension] = coors
        self.node_flag[idx] = True
        self.lastnode += nadd
        self.nnodes += nadd

    def add_edges(self, conn, edge_direction, edge_group=None, edge_low_or_high=None):
        """
        Add new edges at the end of the list.
        :param edge_direction: direction flag
        :param edge_group: describes group of edges from same low super node and same direction
        :param edge_low_or_high: zero for low to low resolution, one for high to high or high to low resolution.
        It is used to set weight from weight table.
        """
        last = self.lastedge
        if type(conn) is nm.ndarray:
            nadd = conn.shape[0]
            idx = slice(last, last + nadd)
            if edge_group is None:
                edge_group = nm.arange(nadd) + last
        else:
            nadd = 1
            idx = nm.array([last])
            conn = nm.array(conn).reshape((1, 2))
            if edge_group is None:
                edge_group = idx

        self.edges[idx, :] = conn
        self.edge_flag[idx] = True
        # t_start0 = time.time()
        # self.edge_flag_idx.extend(list(range(idx.start, idx.stop)))
        # self.stats["t split 082"] += time.time() - t_start0
        self.edge_dir[idx] = edge_direction
        self.edge_group[idx] = edge_group
        # TODO change this just to array of low_or_high_resolution
        if edge_low_or_high is not None and self._edge_weight_table is not None:
            self.edges_weights[idx] = self._edge_weight_table[
                edge_low_or_high, edge_direction
            ]
        self.lastedge += nadd
        self.nedges += nadd

    def finish(self):
        ndidxs = nm.where(self.node_flag)[0]
        aux = -nm.ones((self.nodes.shape[0],), dtype=int)
        aux[ndidxs] = nm.arange(ndidxs.shape[0])
        edges = aux[self.edges[self.edge_flag]]
        nodes = self.nodes[ndidxs]

        # if self.compute_low_node_inverse:
        #     self.low_node_inverse = np.nonzero(self.nodes[:self.data.size])

        if self.compute_msindex:
            # self.msindex = self.msi.msindex
            # relabel
            self.msinds = aux[self.msi.msinds]
            # import sed3
            # ed = sed3.sed3(self.msindex)
            # ed.show()
            # np.unique(self.msindex.ravel())

        if self._edge_weight_table is not None:
            # self.edges_weights = aux[self.edges_weights[self.edge_flag]]
            self.edges_weights = self.edges_weights[self.edge_flag]
            # del self.edges_weights
            # self.edges_weights = edges_weights

        del self.nodes
        del self.node_flag
        del self.edges
        del self.edge_flag
        del self.edge_dir
        del self.edge_group
        del self.msi

        self.nodes = nodes
        self.edges = edges
        self.node_flag = nm.ones((nodes.shape[0],), dtype=nm.bool)
        self.edge_flag = nm.ones((edges.shape[0],), dtype=nm.bool)

    def write_vtk(self, fname):
        write_grid_to_vtk(fname, self.nodes, self.edges, self.node_flag, self.edge_flag)

    def edges_by_group(self, idxs):
        """

        :param idxs: low resolution edge id
        :return: multiscale edges. If this part remain in low resolution the output is just one number
        """
        ed = self.edge_group[idxs]
        ugrps = nm.unique(ed)
        out = []
        for igrp in ugrps:
            out.append(idxs[nm.where(ed == igrp)[0]])

        return out

    def _edge_group_substitution(
        self, ndid, nsplit, idxs, sr_tab, ndoffset, ed_remove, into_or_from
    ):
        """
        Reconnect edges.
        :param ndid: id of low resolution edges
        :param nsplit: number of split
        :param idxs: indexes of low resolution
        :param sr_tab:
        :param ndoffset:
        :param ed_remove:
        :param into_or_from: if zero, connection of input edges is done. If one, connection of output edges
        is performed.
        :return:
        """
        # this is useful for type(idxs) == np.ndarray
        eidxs = idxs[nm.where(self.edges[idxs, 1 - into_or_from] == ndid)[0]]
        # selected_edges = self.edges[idxs, 1 - into_or_from]
        # selected_edges == ndid
        # whre = nm.where(self.edges[idxs, 1 - into_or_from] == ndid)
        # whre0 = (nm.where(self.edges[idxs, 1 - into_or_from] == ndid) == ndid)[0]
        # eidxs = [idxs[i] for i in idxs]
        for igrp in self.edges_by_group(eidxs):
            if igrp.shape[0] > 1:
                # high resolution block to high resolution block
                # all directions are the same
                directions = self.edge_dir[igrp[0]]
                edge_indexes = sr_tab[directions, :].T.flatten() + ndoffset
                # debug code
                # if len(igrp) != len(edge_indexes):
                #     print("Problem ")
                self.edges[igrp, 1] = edge_indexes
                if self._edge_weight_table is not None:
                    self.edges_weights[igrp] = self._edge_weight_table[1, directions]
            else:
                # low res block to hi res block, if into_or_from is set to 0
                # hig res block to low res block, if into_or_from is set to 1
                ed_remove.append(igrp[0])
                # number of new edges is equal to number of pixels on one side of the box (in 2D and D too)
                nnewed = np.power(nsplit, self.data.ndim - 1)
                muleidxs = nm.tile(igrp, nnewed)
                # copy the low-res edge multipletime
                newed = self.edges[muleidxs, :]
                neweddir = self.edge_dir[muleidxs]
                local_node_ids = sr_tab[
                    self.edge_dir[igrp] + self.data.ndim * into_or_from, :
                ].T.flatten()
                # first or second (the actual) node id is substitued by new node indexes
                newed[:, 1 - into_or_from] = local_node_ids + ndoffset
                if self._edge_weight_table is not None:
                    self.add_edges(
                        newed, neweddir, self.edge_group[igrp], edge_low_or_high=1
                    )
                else:
                    self.add_edges(
                        newed, neweddir, self.edge_group[igrp], edge_low_or_high=None
                    )
        return ed_remove

    def split_voxel(self, ndid):
        """

        :param ndid: int-like, low resolution voxel_id
        :param nsplit: int-like number
        :param tile_shape: this parameter will be used in future
        :return:
        """
        # TODO use tile_shape instead of nsplit
        # nsplit - was size of split square, tiles_shape = [nsplit, nsplit]
        # generate subgrid
        # tile_shape = tuple(tile_shape)
        # nsplit = tile_shape[0]
        # tile_shape = (nsplit, nsplit)
        t_start = time.time()
        self.stats["t split 01"] += time.time() - t_start
        nsplit = self.nsplit
        tile_shape = self._tile_shape
        if tile_shape in self.cache:
            nd, ed, ed_dir = self.cache[tile_shape]
        else:
            nd, ed, ed_dir = self.gen_grid_fcn(tile_shape, self.voxelsize / nsplit)
            # nd, ed, ed_dir = gen_base_graph(tile_shape, self.voxelsize / tile_shape)
            self.cache[tile_shape] = nd, ed, ed_dir
        self.stats["t split 02"] += time.time() - t_start

        ndoffset = self.lastnode
        # in new implementation nodes are 2D on 2D shape and 3D in 3D shape
        # in old implementation nodes are always 3D
        sr_tab = self.srt.get_sr_subtab(tile_shape)
        self.stats["t split 03"] += time.time() - t_start
        if self.compute_msindex:
            self.msi.set_block_higres(ndid, self.srt.inds + ndoffset)
        self.stats["t split 04"] += time.time() - t_start
        nd = make_nodes_3d(nd)
        self.stats["t split 05"] += time.time() - t_start
        self.add_nodes(nd + self.nodes[ndid, :] - (self.voxelsize3 / 2))
        self.stats["t split 06"] += time.time() - t_start
        if self._edge_weight_table is not None:
            # high resolution
            self.add_edges(ed + ndoffset, ed_dir, edge_low_or_high=1)
        else:
            self.add_edges(ed + ndoffset, ed_dir, edge_low_or_high=None)
        self.stats["t split 07"] += time.time() - t_start

        # connect subgrid
        ed_remove = []
        # sr_tab_old = self.sr_tab[nsplit]

        # TODO use just one variant
        # t_start0 = time.time()
        idxs = nm.where(self.edge_flag > 0)[0]
        # self.stats["t split 081"] += time.time() - t_start0
        # no np.where() variant
        # t_start0 = time.time()
        # idxs = np.array(self.edge_flag_idx)
        # self.stats["t split 082"] += time.time() - t_start0
        self.stats["t split 08"] += time.time() - t_start


        # edges "into" node?
        ed_remove = self._edge_group_substitution(
            ndid, nsplit, idxs, sr_tab, ndoffset, ed_remove, into_or_from=0
        )
        self.stats["t split 09"] += time.time() - t_start

        # edges "from" node?
        ed_remove = self._edge_group_substitution(
            ndid, nsplit, idxs, sr_tab, ndoffset, ed_remove, into_or_from=1
        )

        self.stats["t split 10"] += time.time() - t_start
        # remove node
        self.node_flag[ndid] = False
        # remove edges
        self.edge_flag[ed_remove] = False
        # TODO maybe remove
        # t_start0 = time.time()
        # self.edge_flag_idx = [one_flag_id for one_flag_id in self.edge_flag_idx if one_flag_id not in ed_remove]
        # self.stats["t split 0821"] += time.time() - t_start0

    def generate_base_grid(self, vtk_filename=None):
        """
        Run first step of algorithm. Next step is split_voxels
        :param vtk_filename:
        :return:
        """
        nd, ed, ed_dir = self.gen_grid_fcn(self.data.shape, self.voxelsize)
        self.add_nodes(nd)
        self.add_edges(ed, ed_dir, edge_low_or_high=0)

        if vtk_filename is not None:
            self.write_vtk(vtk_filename)

    def split_voxels(self, vtk_filename=None):
        """
        Second step of algorithm
        :return:()
        """
        self.cache = {}
        self.stats["t graph 10"] = time.time() - self.start_time
        self.msi = MultiscaleArray(self.data.shape, block_size=self.nsplit)

        # old implementation
        # idxs = nm.where(self.data)
        # nr, nc = self.data.shape
        # for k, (ir, ic) in enumerate(zip(*idxs)):
        #     ndid = ic + ir * nc
        #     self.split_voxel(ndid, self.nsplit)

        # new_implementation
        # for ndid in np.flatnonzero(self.data):
        #     self.split_voxel(ndid, self.nsplit)

        # even newer implementation
        self.stats["t graph 11"] = time.time() - self.start_time
        for ndid, val in enumerate(self.data.ravel()):
            t_split_start = time.time()
            if val == 0:
                if self.compute_msindex:
                    self.msi.set_block_lowres(ndid, ndid)
                self.stats["t graph low"] += time.time() - t_split_start
            else:
                self.split_voxel(ndid)
                self.stats["t graph high"] += time.time() - t_split_start

        self.stats["t graph 13"] = time.time() - self.start_time
        self.finish()
        if vtk_filename is not None:
            self.write_vtk(vtk_filename)
        self.stats["t graph 14"] = time.time() - self.start_time

    def run(self, base_grid_vtk_fn=None, final_grid_vtk_fn=None):
        # cache dict.
        self.cache = {}

        # generate base grid
        self.stats["t graph 02"] = time.time() - self.start_time
        self.generate_base_grid(base_grid_vtk_fn)
        self.stats["t graph 09"] = time.time() - self.start_time
        # self.generate_base_grid()
        # split voxels
        self.split_voxels(final_grid_vtk_fn)
        # self.split_voxels()


def get_efficient_signed_int_type(number):
    if number < np.iinfo(np.int16).max:
        eddtype = np.int16
    elif number < np.iinfo(np.int32).max:
        eddtype = np.int32
    elif number < np.iinfo(np.int64).max:
        eddtype = np.int64
    elif number < np.iinfo(np.int128).max:
        eddtype = np.int128
    else:
        logger.error("Edge number higher than int128.")

    return eddtype


class SRTab(object):
    """
    Table of connection on transition between low resolution and high resolution
    """

    def __init__(self):
        # spliting reconnection table
        # sr_tab = {
        #     2: nm.array([(0,2), (0,1), (1,3), (2,3)]),
        #     3: nm.array([(0,3,6), (0,1,2), (2,5,8), (6,7,8)]),
        #     4: nm.array([(0,4,8,12), (0,1,2,3), (3,7,11,15), (12,13,14,15)]),
        # }
        self.sr_tab = {}
        # self.set_new_shape(shape)

    def set_new_shape(self, shape):
        self.shape = shape
        if len(shape) not in (2, 3):
            logger.error("2D or 3D shape expected")

        self.inds = np.array(range(np.prod(self.shape)))
        # direction_order = [0, 1, 2, 3, 4, 5, 6]
        reshaped = self.inds.reshape(self.shape)

        tab = []
        for direction in range(len(self.shape) - 1, -1, -1):
            # direction = direction_order[i]
            tab.append(reshaped.take(0, direction).flatten())
        for direction in range(len(self.shape) - 1, -1, -1):
            # direction = direction_order[i]
            tab.append(reshaped.take(-1, direction).flatten())
        self.sr_tab[tuple(shape)] = np.array(tab)

    def get_sr_subtab(self, shape):
        shape = tuple(shape)
        if shape not in self.sr_tab:
            self.set_new_shape(shape)
        return self.sr_tab[shape]


def grid_edges(shape, inds=None, return_directions=True):
    """
    Get list of grid edges
    :param shape:
    :param inds:
    :param return_directions:
    :return:
    """
    if inds is None:
        inds = np.arange(np.prod(shape)).reshape(shape)
    # if not self.segparams['use_boundary_penalties'] and \
    #         boundary_penalties_fcn is None :
    if len(shape) == 2:
        edgx = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
        edgy = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]

        edges = [edgx, edgy]

        directions = [
            np.ones([edgx.shape[0]], dtype=np.int8) * 0,
            np.ones([edgy.shape[0]], dtype=np.int8) * 1,
        ]

    elif len(shape) == 3:
        # This is faster for some specific format
        edgx = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
        edgy = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
        edgz = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
        edges = [edgx, edgy, edgz]
    else:
        logger.error("Expected 2D or 3D data")

    # for all edges along first direction put 0, for second direction put 1, for third direction put 3
    if return_directions:
        directions = []
        for idirection in range(len(shape)):
            directions.append(
                np.ones([edges[idirection].shape[0]], dtype=np.int8) * idirection
            )
    edges = np.concatenate(edges)
    if return_directions:
        edge_dir = np.concatenate(directions)
        return edges, edge_dir
    else:
        return edges


def grid_nodes(shape, voxelsize=None):
    voxelsize = np.asarray(voxelsize)  # [:len(shape)]
    nodes = np.moveaxis(np.indices(shape), 0, -1).reshape(-1, len(shape))
    if voxelsize is not None:
        nodes = (nodes * voxelsize) + (0.5 * voxelsize)
    return nodes


def gen_grid_nd(shape, voxelsize=None, inds=None):
    edges, edge_dir = grid_edges(shape, inds, return_directions=True)
    # nodes coordinates
    nodes = grid_nodes(shape, voxelsize)
    return nodes, edges, edge_dir


def gen_grid_2d(shape, voxelsize):
    """
    Generate list of edges for a base grid.
    """
    nr, nc = shape
    nrm1, ncm1 = nr - 1, nc - 1
    # sh = nm.asarray(shape)
    # calculate number of edges, in 2D: (nrows * (ncols - 1)) + ((nrows - 1) * ncols)
    nedges = 0
    for direction in range(len(shape)):
        sh = copy.copy(list(shape))
        sh[direction] += -1
        nedges += nm.prod(sh)

    nedges_old = ncm1 * nr + nrm1 * nc
    edges = nm.zeros((nedges, 2), dtype=nm.int16)
    edge_dir = nm.zeros((ncm1 * nr + nrm1 * nc,), dtype=nm.bool)
    nodes = nm.zeros((nm.prod(shape), 3), dtype=nm.float32)

    # edges
    idx = 0
    row = nm.zeros((ncm1, 2), dtype=nm.int16)
    row[:, 0] = nm.arange(ncm1)
    row[:, 1] = nm.arange(ncm1) + 1
    for ii in range(nr):
        edges[slice(idx, idx + ncm1), :] = row + nc * ii
        idx += ncm1

    edge_dir[slice(0, idx)] = 0  # horizontal dir

    idx0 = idx
    col = nm.zeros((nrm1, 2), dtype=nm.int16)
    col[:, 0] = nm.arange(nrm1) * nc
    col[:, 1] = nm.arange(nrm1) * nc + nc
    for ii in range(nc):
        edges[slice(idx, idx + nrm1), :] = col + ii
        idx += nrm1

    edge_dir[slice(idx0, idx)] = 1  # vertical dir

    # nodes
    idx = 0
    row = nm.zeros((nc, 3), dtype=nm.float32)
    row[:, 0] = voxelsize[0] * (nm.arange(nc) + 0.5)
    row[:, 1] = voxelsize[1] * 0.5
    for ii in range(nr):
        nodes[slice(idx, idx + nc), :] = row
        row[:, 1] += voxelsize[1]
        idx += nc

    return nodes, edges, edge_dir


def make_nodes_3d(nodes):
    if nodes.shape[1] == 2:
        zeros = np.zeros([nodes.shape[0], 1], dtype=nodes.dtype)
        nodes = np.concatenate([nodes, zeros], axis=1)
    return nodes


def write_grid_to_vtk(fname, nodes, edges, node_flag=None, edge_flag=None):
    """
    Write nodes and edges to VTK file
    :param fname: VTK filename
    :param nodes:
    :param edges:
    :param node_flag: set if this node is really used in output
    :param edge_flag: set if this flag is used in output
    :return:
    """

    if node_flag is None:
        node_flag = np.ones([nodes.shape[0]], dtype=np.bool)
    if edge_flag is None:
        edge_flag = np.ones([edges.shape[0]], dtype=np.bool)
    nodes = make_nodes_3d(nodes)
    f = open(fname, "w")

    f.write("# vtk DataFile Version 2.6\n")
    f.write("output file\nASCII\nDATASET UNSTRUCTURED_GRID\n")

    idxs = nm.where(node_flag > 0)[0]
    nnd = len(idxs)
    aux = -nm.ones(node_flag.shape, dtype=nm.int32)
    aux[idxs] = nm.arange(nnd, dtype=nm.int32)
    f.write("\nPOINTS %d float\n" % nnd)
    for ndi in idxs:
        f.write("%.6f %.6f %.6f\n" % tuple(nodes[ndi, :]))

    idxs = nm.where(edge_flag > 0)[0]
    ned = len(idxs)
    f.write("\nCELLS %d %d\n" % (ned, ned * 3))
    for edi in idxs:
        f.write("2 %d %d\n" % tuple(aux[edges[edi, :]]))

    f.write("\nCELL_TYPES %d\n" % ned)
    for edi in idxs:
        f.write("3\n")


class MultiscaleArray(object):
    def __init__(self, shape, block_size, arr=None):
        self.shape = np.asarray(shape)
        if arr is None:
            self.msinds = np.zeros(self.shape * block_size, dtype=int)
        else:
            self.msinds = arr
        self.block_size = block_size
        self.block_shape = [block_size] * self.msinds.ndim
        self.cache_slice = [None] * self.msinds.ndim

    def _prepare_cache_slice(self, index):
        coords = np.unravel_index(index, self.shape)

        for ax, single_ax_coord in enumerate(coords):
            coord_higres_start = single_ax_coord * self.block_size
            self.cache_slice[ax] = slice(
                coord_higres_start, coord_higres_start + self.block_size
            )

    def set_block_lowres(self, index, val):
        self._prepare_cache_slice(index)
        self.msinds[tuple(self.cache_slice)] = val

    def set_block_higres(self, index, val):
        self._prepare_cache_slice(index)
        self.msinds[tuple(self.cache_slice)] = np.asarray(val).reshape(self.block_shape)

    def mul_block(self, index, val):
        """Multiply values in block"""
        self._prepare_cache_slice(index)
        self.msinds[tuple(self.cache_slice)] *= val


# def relabel(arr, forward_indexes=None):
#     # for x in np.nditer(arr, op_flags=["readwrite"]):
#     #     x[...] = forward_indexes[x]
#     pass
