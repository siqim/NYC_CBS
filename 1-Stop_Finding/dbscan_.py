# -*- coding: utf-8 -*-
"""
DBSCAN: Density-Based Spatial Clustering of Applications with Noise
"""

# Author: Robert Layton <robertlayton@gmail.com>
#         Joel Nothman <joel.nothman@gmail.com>
#         Lars Buitinck
#
# License: BSD 3 clause

# Modified by Siqi Miao - Mar 19 2019

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster._dbscan_inner import dbscan_inner


def filter_low_quality_cluster(labels, thres=20):
    uniq_labels, counts = np.unique(labels, return_counts=True)
    low_quality_cluster = uniq_labels[np.argwhere(counts<20)]

    labels[np.isin(labels, low_quality_cluster)] = -1
    return labels


def dbscan(coords, timestamps, eps_d=500, eps_t=10/1.66667e-11, min_samples=5, metric_d='l1', metric_t='l1', algorithm='auto', leaf_size=30, n_jobs=1):

    neighbors_model_O = NearestNeighbors(radius=eps_d, leaf_size=leaf_size, metric=metric_d, n_jobs=n_jobs, algorithm=algorithm)
    neighbors_model_O.fit(coords[:, :2])
    neighborhoods_O = neighbors_model_O.radius_neighbors(coords[:, :2], eps_d, return_distance=False)


    neighbors_model_D = NearestNeighbors(radius=eps_d, leaf_size=leaf_size, metric=metric_d, n_jobs=n_jobs, algorithm=algorithm)
    neighbors_model_D.fit(coords[:, 2:])
    neighborhoods_D = neighbors_model_D.radius_neighbors(coords[:, 2:], eps_d, return_distance=False)


    neighbors_model_t_O = NearestNeighbors(radius=eps_t, leaf_size=leaf_size, metric=metric_t, n_jobs=n_jobs, algorithm=algorithm)
    neighbors_model_t_O.fit(timestamps[:, [0]])
    neighborhoods_t_O = neighbors_model_t_O.radius_neighbors(timestamps[:, [0]], eps_t, return_distance=False)

    neighbors_model_t_D = NearestNeighbors(radius=eps_t, leaf_size=leaf_size, metric=metric_t, n_jobs=n_jobs, algorithm=algorithm)
    neighbors_model_t_D.fit(timestamps[:, [1]])
    neighborhoods_t_D = neighbors_model_t_D.radius_neighbors(timestamps[:, [1]], eps_t, return_distance=False)


    n_neighbors = np.zeros(coords.shape[0], dtype=np.int16)
    neighborhoods = np.empty(coords.shape[0], dtype=object)
    for i in range(coords.shape[0]):
        neighbor_i = np.array(list(set(neighborhoods_O[i]).intersection(set(neighborhoods_D[i]), set(neighborhoods_t_O[i]), set(neighborhoods_t_D[i]))))
        neighborhoods[i] = neighbor_i

        n_neighbors[i] = neighbor_i.shape[0]

    # Initially, all samples are noise.
    labels = -np.ones(coords.shape[0], dtype=np.intp)

    # A list of all core samples found.
    core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
    dbscan_inner(core_samples, neighborhoods, labels)
    return np.where(core_samples)[0], labels


class DBSCAN(BaseEstimator, ClusterMixin):

    def __init__(self, eps_d=500, eps_t=10/1.66667e-11, min_samples=5, metric_d='l1', metric_t='l1', algorithm='auto', leaf_size=30, n_jobs=1):
        self.eps_d = eps_d
        self.eps_t = eps_t
        self.min_samples = min_samples
        self.metric_d = metric_d
        self.metric_t = metric_t
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs

    def fit(self, coords, timestamps):

        clust = dbscan(coords, timestamps, **self.get_params())
        self.core_sample_indices_, self.labels_ = clust
        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = coords[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, coords.shape[1]))
        return self
