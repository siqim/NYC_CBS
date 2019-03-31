# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:49:08 2019

@author: msq96
"""


import numpy as np
from sklearn.metrics import pairwise_distances


def canopy(coords, timestamps, D1, D2, t1, t2, distance_metric=['l2', 'l1']):

    dist_Os = pairwise_distances(coords[:, :2], metric=distance_metric[0]) # in km
    dist_Ds = pairwise_distances(coords[:, 2:], metric=distance_metric[0]) # in km
    diff_Ts = pairwise_distances(timestamps, metric=distance_metric[1]) * 1.66667e-11 # ns to min

    canopies = {}
    canopy_points = set(range(coords.shape[0]))
    point_removed = []
    while canopy_points:

        point = canopy_points.pop()
        i = len(canopies)

        points_meet_loose_conds = np.where( (dist_Os[point] <= D1) & (dist_Ds[point] <= D1) & (diff_Ts[point] <= t1) ) [0]
        points_meet_loose_conds = list(set(points_meet_loose_conds) - set(point_removed))
        assert(points_meet_loose_conds != [])

        canopies[i] = {"c":point, "points": points_meet_loose_conds}

        points_meet_tight_conds = np.where( (dist_Os[point] <= D2) & (dist_Ds[point] <= D2) & (diff_Ts[point] <= t2) ) [0]
        points_meet_tight_conds = list(set(points_meet_tight_conds) - set(point_removed))
        assert(points_meet_tight_conds != [])

        canopy_points = canopy_points.difference(set(points_meet_tight_conds))

        point_removed.extend(points_meet_tight_conds)

    return canopies
