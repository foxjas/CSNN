import sys
import numpy as np
import argparse
import os
import time
import scipy
from pprint import pprint
from scipy.spatial.distance import pdist
from pymatgen.io.vasp.outputs import Poscar
from pymatgen.transformations.standard_transformations import RotationTransformation
from mesh.mesh import *


class ConfigurationFingerprint(object):
    """
    Maps configuration file to icosahedral spherical mesh signals
    Output dim: [N x R x V]
    """

    def __init__(self, r_atom, r_dim, l, map_type, scaling_type, T=0.01):
        self.r_atom = r_atom
        self.r_dim = r_dim
        self.mesh_lvl = l
        self.map_type = map_type
        self.scaling_type = scaling_type
        self.smooth_cutoff = T
        self.mesh = icosphere(self.mesh_lvl) # TODO: this should be generated once and saved to file
        self.mesh_knn = scipy.spatial.cKDTree(self.mesh.vertices)
        
        self.scaling_fn = None
        if scaling_type == "none":
            # identity
            self.scaling_fn = lambda x, r: x
        elif scaling_type == "inverse":
            self.scaling_fn = lambda x, r: x/r
        elif scaling_type == "inverse_sq":
            self.scaling_fn = lambda x, r: x/(r**2)

        self.smoothing_fn = None
        if map_type == "linear":
            self.mapping = self.map_linear
        elif map_type == "sqrt":
            self.mapping = self.map_sqrt
        elif map_type == "log":
            self.mapping = self.map_log
            
    def __call__(self, config_data, max_atoms, rotate=False):
        sites, environments, neighbor_dists = self.spherical_neighbors(config_data, self.r_atom, rotate)
        features = self.mapping(sites, environments, neighbor_dists, self.r_dim, self.r_atom)
        non_null_atoms = np.full(max_atoms, False)
        non_null_atoms[:len(sites)] = True
        X = np.zeros((max_atoms, *features.shape[1:]))
        X[non_null_atoms, ...] = features

        return X, non_null_atoms


    def spherical_neighbors(self, atoms_data, r, rotate):
        """
        Finds coordinates and distances of neighbors, centered at reference point

        :param r: radius cut-off
        """
        supercell = atoms_data.structure
        if rotate:
            axis, angle = random_axis_angle()
            rot = RotationTransformation(axis, angle, angle_in_radians=True)
            supercell = rot.apply_transformation(supercell)

        atom_sites = supercell.sites
        s0 = atom_sites[0]

        atom_coords = np.array([s.coords for s in atom_sites])
        center_idx, point_idx, offsets, dists = supercell.get_neighbor_list(r, atom_sites)
        neighbor_coords = atom_coords[point_idx] + offsets.dot(s0.lattice.matrix)
        
        neighbor_counts = np.zeros(len(atom_sites), dtype=int)
        for c_idx in center_idx:
            neighbor_counts[c_idx] += 1

        atom_neighbors = []
        atom_neighbor_dists = []
        neighbor_offsets = np.zeros(len(atom_sites)+1, dtype=int)
        for i, count in enumerate(neighbor_counts):
            neighbor_offsets[i+1] = neighbor_offsets[i] + count
       
        for i in range(len(neighbor_counts)):
            off1, off2 = neighbor_offsets[i], neighbor_offsets[i+1]
            atom_neighbors.append(neighbor_coords[off1:off2])
            atom_neighbor_dists.append(dists[off1:off2])

        neighb_coords_centered = []
        neighb_dists_scaled = []
        for (coords, neighb_coords, dists) in zip(atom_coords, atom_neighbors, atom_neighbor_dists):
            if len(neighb_coords):
                neighbors_c = neighb_coords-coords
                neighb_coords_centered.append(neighbors_c)

        return atom_coords, neighb_coords_centered, atom_neighbor_dists


    def map_linear(self, sites, environments, neighbor_dists, R, r_max, r_min=1):
        """
        Assumes r=1 (outermost) radius
        Input: [N x neighbors x 3]
        Output: [N x 1 x V x F] array
        """

        n_atoms = len(sites)
        r_range = r_max - r_min
        scale = np.array([i+1 for i in range(R)])
        #print("scale: {}".format(scale))
        s_max = scale[-1]
        rcuts = (scale/s_max)*r_range
        #print("rcuts: {}".format(rcuts))
        
        spherical_signal = np.zeros((n_atoms, R, self.mesh.vertices.shape[0]))
        for site_idx in range(len(sites)):
            pts = environments[site_idx]
            pts_unit = normalize_to_radius(pts, 1)
            _, mesh_idx = self.mesh_knn.query(pts_unit, k=1, n_jobs=1) # nearest vertex
            p_dists = neighbor_dists[site_idx]
            p_dists_scaled = p_dists - r_min
            r_idx = (np.ceil(p_dists_scaled*s_max / r_range)-1).astype(int)

            Z = self.scaling_fn(np.ones_like(mesh_idx), p_dists)
            for i, (r_idx, v_idx) in enumerate(zip(r_idx, mesh_idx)):
                spherical_signal[site_idx, r_idx, v_idx] += Z[i]

        spherical_signal = spherical_signal.transpose((0, 2, 1)) # [n, V, F]
        spherical_signal = spherical_signal[:, np.newaxis, :, :]

        return spherical_signal


def random_axis_angle():
    """
    Does not necessarily correspond to uniformly random rotation because
    angle should be *non-uniformly* sampled
    """
    v = np.random.normal(size=3)
    v_unit = v/np.linalg.norm(v)
    angle = np.random.uniform(0, 2*np.pi)
    return v_unit, angle

        
def rbf(s, gamma=1):
    """
    https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    Input: scalar
    """
    return np.exp(-gamma*(s**2))


def normalize_to_radius(points, radius):
    '''
    Reproject points to specified radius(es)
    - points: array
    - radius: scalar or array
    '''
    scalar = (points**2).sum(axis=-1, keepdims=True)**.5
    unit = points / scalar
    offset = radius - scalar
    points_new = points + unit*offset
    return points_new
