import sys
import numpy as np
import argparse
import os
import scipy
import torch
from scipy.spatial.distance import pdist
from mesh.mesh import *
from .dataset import ModelNet


class IcosphereSignalConstant(object):
    """
    Map point cloud to icosahedral (multi-)spherical mesh feature map.
    Each point is assigned as indicator value to neighborhood of vertices.
    """

    def __init__(self, l, radius_lvls):
        self.mesh_lvl = l
        self.mesh = icosphere(self.mesh_lvl)
        self.radius_lvls = radius_lvls

    def __call__(self, pts):
        dists = np.linalg.norm(pts, axis=-1)
        max_dist = np.max(dists)
        pts /= max_dist
        #max_dist_after = np.max(np.linalg.norm(pts, axis=-1))
        #assert max_dist_after <= 1 + 1e-6, "Exceeding distance: {}".format(max_dist_after)
        features = self.map_data_to_mesh(pts, self.radius_lvls)
        return torch.FloatTensor(features)


    def map_data_to_mesh(self, pts, R):
        """
        R: radial bandwidth
        Input: [N x 3] array
        Returns [R x V x F] array
        """
        if not self.mesh.knn:
            self.mesh.knn = scipy.spatial.cKDTree(self.mesh.vertices)

        K = 3
        r_inc = 1/R
        spherical_signal = np.zeros((R, self.mesh.vertices.shape[0]))
        unit_data_pts = normalize_to_radius(pts, 1)

        p_dists = np.linalg.norm(pts, axis=-1) # original radial distance
        _, mesh_idx = self.mesh.knn.query(unit_data_pts, k=K, n_jobs=1)
        r_idx_upper = (p_dists/r_inc).astype(int)
        r_idx_upper -= (r_idx_upper==R).astype(int) # drop 1 level if pt is on unit boundary

        for i, (r_idx, v_idx) in enumerate(zip(r_idx_upper, mesh_idx)):
            spherical_signal[r_idx, v_idx] += 1

        feat_dim = 1
        mesh_signals = np.array(spherical_signal).reshape(R, -1, feat_dim)

        return mesh_signals


class IcosphereSignalKernel(object):

    """
    Map point cloud to icosahedral (multi-)spherical mesh feature map. 
    Each point is assigned to neighborhood of vertices using radial basis function.
    """

    def __init__(self, l, r_dim, radial_factor, T=0.01):
        """
        Radial_factor needs to be power of 2
        """
        self.mesh_lvl = l
        self.r_dim = r_dim
        self.r_factor = radial_factor
        self.mesh_lvl = l
        self.mesh = icosphere(self.mesh_lvl)
        self.mesh_knn = scipy.spatial.cKDTree(self.mesh.vertices)
        self.gamma = self.compute_gamma(self.mesh, self.r_dim, T)
        #print("self.gamma: {}".format(self.gamma))

    def __call__(self, pts):
        dists = np.linalg.norm(pts, axis=-1)
        max_dist = np.max(dists)
        pts /= max_dist
        #max_dist_after = np.max(np.linalg.norm(pts, axis=-1))
        #assert max_dist_after <= 1 + 1e-6, "Exceeding distance: {}".format(max_dist_after)
        features = self.smooth_data_over_mesh(pts, rbf, self.r_dim, self.r_factor)
        return torch.FloatTensor(features)


    def smooth_data_over_mesh(self, pts, smoothing_fn, R, expand_factor):
        """
        Assumes r=1 (outermost) radius
        Input: [N x 3] array
        Output: [R x V x F] array
        """

        K = 3
        r_inc = 1/R
        R_expand = R*expand_factor
        r_inc_expand = 1/R_expand
        spherical_signal_expand = np.zeros((R_expand, self.mesh.vertices.shape[0]))
        unit_data_pts = normalize_to_radius(pts, 1)

        p_dists = np.linalg.norm(pts, axis=-1) # original radial distance
        _, mesh_idx = self.mesh_knn.query(unit_data_pts, k=K, n_jobs=1)
        r_idx_orig = (p_dists/r_inc).astype(int) # [N, 1]
        r_idx_orig -= (r_idx_orig==R).astype(int) # drop 1 level if pt is on unit boundary
        
        # expanded levels and radii
        r_idx_scaled = (r_idx_orig+1)*expand_factor-1
        r_idx_offsets = np.arange(expand_factor)
        r_idx_expand = r_idx_scaled[...,np.newaxis] - r_idx_offsets # [N, expand_factor]
        r_expand = (r_idx_expand+1)*r_inc_expand
        r_expand = r_expand[..., np.newaxis, np.newaxis]

        # expanded vertex positions
        rep_vertices = self.mesh.vertices[mesh_idx] # [N, K, 3]
        rep_vertices = rep_vertices[:, np.newaxis, :, :]
        vertices = normalize_to_radius(rep_vertices, r_expand) # [N, expand_factor, K, 3]

        # map data pts to expanded vertices
        pts = pts[:, np.newaxis, np.newaxis, :]
        X = vertices - pts
        d = np.linalg.norm(X, axis=-1)
        gamma_r = self.gamma[r_idx_orig]
        gamma_r = gamma_r[:, np.newaxis, np.newaxis]
        Z = smoothing_fn(d, gamma=gamma_r) # [N, expand_factor, K]

        # update spherical signal
        for i, (r_idx, v_idx) in enumerate(zip(r_idx_expand, mesh_idx)):
            spherical_signal_expand[r_idx[...,np.newaxis], v_idx] += Z[i]

        spherical_signal_expand = spherical_signal_expand.reshape((R, expand_factor, -1))
        spherical_signal_expand = spherical_signal_expand.transpose((0, 2, 1)) # [R, V, expand_factor]
        spherical_signal = spherical_signal_expand

        return spherical_signal


    def compute_gamma(self, ico, R, T, face_idx=0):
        """
        Assumes 3 vertices per level
        """
        rep_vert_ids = ico.faces[face_idx]
        rep_vertices = ico.vertices[rep_vert_ids]
        r_inc = 1/R
        dist_bounds = np.zeros(R)
   
        # handle radial dimensions R > 1
        for i in range(R, 1, -1):
            r_upper = i*r_inc
            r_lower = (i-1)*r_inc
            vertices_upper = normalize_to_radius(rep_vertices, r_upper)
            vertices_lower = normalize_to_radius(rep_vertices, r_lower)
            vertices_all = np.concatenate((vertices_upper, vertices_lower))
            d_max = np.max(pdist(vertices_all, metric='euclidean'))
            dist_bounds[i-1] = d_max

        # handle nearest radial level to origin
        r = r_inc
        vertices = normalize_to_radius(rep_vertices, r)
        vertices_all = np.append(vertices, np.zeros((1, 3)), axis=0)
        d_max = np.max(pdist(vertices_all, metric='euclidean'))
        dist_bounds[0] = d_max

        gammas = -np.log(T)/(dist_bounds**2)

        return gammas


def rbf(s, gamma=1):
    """
    Gaussian radial basis function
    """
    return np.exp(-gamma*(s**2))


def normalize_to_radius(points, radius):
    """
    Reproject points to specified radii
    - points: array
    - radius: scalar or array
    """
    scalar = (points**2).sum(axis=-1, keepdims=True)**.5
    unit = points / scalar
    offset = radius - scalar
    points_new = points + unit*offset

    return points_new
