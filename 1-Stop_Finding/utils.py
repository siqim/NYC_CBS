# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:50:46 2019

@author: msq96
"""

import utm
import math
from itertools import repeat

import numpy as np
from scipy.spatial import ConvexHull

from sklearn.metrics import pairwise_distances

from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric

import folium
from color_brewer import color_brewer
import matplotlib.pyplot as plt
import seaborn as sns

def visz_merged_stops(coords, labels, potential_stops, thres=100):

    palette = color_brewer('YlGnBu', len(set(labels)))
    uniq_labels = np.unique(labels)

    m = folium.Map(
        location=[40.686892, -73.9876514],
        tiles='cartodbdark_matter',
        zoom_start=12
    )

    for cluster in set(labels):
        if cluster != -1:
            temp_coords = coords[np.argwhere(labels == cluster)].squeeze()
            coord_Os = np.array([utm.to_latlon(each[0], each[1], 18, 'T')\
                                 for each in temp_coords[:, :2]])
            coord_Ds = np.array([utm.to_latlon(each[0], each[1], 18, 'T')\
                                 for each in temp_coords[:, 2:]])

            O_hull = coord_Os[ConvexHull(coord_Os).vertices]
            O_hull = np.concatenate((O_hull, O_hull[[0]]), axis=0)

            D_hull = coord_Ds[ConvexHull(coord_Ds).vertices]
            D_hull = np.concatenate((D_hull, D_hull[[0]]), axis=0)

            cluster_idx = np.argwhere(cluster==uniq_labels).squeeze()
            folium.PolyLine(
                locations=O_hull,
                color=palette[cluster_idx],
                popup='Origin Convex Hull ' + str(cluster),
                weight=1
                ).add_to(m)

            folium.PolyLine(
                locations=D_hull,
                color=palette[cluster_idx],
                popup='Destination Convex Hull ' + str(cluster),
                weight=1
                ).add_to(m)

    for stop_id, stop_loc in potential_stops.items():

        if stop_id != 0:
            stop_latlon = utm.to_latlon(stop_loc[:, 0], stop_loc[:, 1], 18, 'T')

            folium.Circle(
                radius=thres,
                location=stop_latlon,
                color='lightblue',
                fill=True,
                fill_color='lightblue',
                popup='Stop id %d'%stop_id
            ).add_to(m)

    m.save('merged_stops.html')

def merge_stops(potential_stops_with_id, thres=100, metric='l1'):
    merged_potential_stops_with_id = {}
    merged_potential_stops_with_id[0] = 'DUMMY_STOP'

    potential_stops = np.stack(list(potential_stops_with_id.values())).squeeze()

    dist_mat = pairwise_distances(potential_stops, metric=metric)
    dist_mat[dist_mat==0] = np.inf


    while np.min(dist_mat) < thres:

        mask = np.ones(potential_stops.shape[0])
        for idx_i, loc_i in enumerate(potential_stops):
            for idx_j, loc_j in enumerate(potential_stops):

                if idx_j > idx_i:
                    dist = pairwise_distances(loc_i.reshape(1,-1), loc_j.reshape(1,-1), metric=metric).reshape(-1)

#                    print(dist)
                    if dist < thres:

                        merged_loc = ((loc_i + loc_j) / 2).reshape(1, -1)
                        mask[idx_i] = 0
                        mask[idx_j] = 0
                        potential_stops = np.concatenate((potential_stops[mask==1], merged_loc), axis=0)
                        break
            if dist < thres:
                break

        dist_mat = pairwise_distances(potential_stops, metric=metric)
        dist_mat[dist_mat==0] = np.inf

    merged_potential_stops_with_id.update({i:potential_stops[i-1].reshape(1, -1) for i in range(1, len(potential_stops)+1)})

    return merged_potential_stops_with_id


def visz_stops_and_convexhull(coords, labels, potential_stops):
    all_potential_stops = []

    palette = color_brewer('YlGnBu', len(set(labels)))
    uniq_labels = np.unique(labels)

    m = folium.Map(
        location=[40.686892, -73.9876514],
        tiles='cartodbdark_matter',
        zoom_start=12
    )

    for cluster in set(labels):
        if cluster != -1:
            temp_coords = coords[np.argwhere(labels == cluster)].squeeze()
            coord_Os = np.array([utm.to_latlon(each[0], each[1], 18, 'T')\
                                 for each in temp_coords[:, :2]])
            coord_Ds = np.array([utm.to_latlon(each[0], each[1], 18, 'T')\
                                 for each in temp_coords[:, 2:]])

            O_hull = coord_Os[ConvexHull(coord_Os).vertices]
            O_hull = np.concatenate((O_hull, O_hull[[0]]), axis=0)

            D_hull = coord_Ds[ConvexHull(coord_Ds).vertices]
            D_hull = np.concatenate((D_hull, D_hull[[0]]), axis=0)

            cluster_idx = np.argwhere(cluster==uniq_labels).squeeze()
            folium.PolyLine(
                locations=O_hull,
                color=palette[cluster_idx],
                popup='Origin Convex Hull ' + str(cluster),
                weight=1
                ).add_to(m)

            folium.PolyLine(
                locations=D_hull,
                color=palette[cluster_idx],
                popup='Destination Convex Hull ' + str(cluster),
                weight=1
                ).add_to(m)

            OD_centers = potential_stops[cluster]
            O_centers = OD_centers['O_centers']
            D_centers = OD_centers['D_centers']

            for k in O_centers:
                stops = O_centers[k]
                all_potential_stops.extend(stops)

                stop_latlons = [utm.to_latlon(each[0], each[1], 18, 'T') for each in stops]

                for stop in stop_latlons:
                    folium.Circle(
                        radius=100,
                        location=stop,
                        color=palette[cluster_idx],
                        fill=True,
                        fill_color=palette[cluster_idx],
                        popup='Origin ' + str(cluster) + ' k=' + str(k)
                    ).add_to(m)

            for k in D_centers:
                stops = D_centers[k]
                all_potential_stops.extend(stops)

                stop_latlons = [utm.to_latlon(each[0], each[1], 18, 'T') for each in stops]

                for stop in stop_latlons:
                    folium.Circle(
                        radius=100,
                        location=stop,
                        color=palette[cluster_idx],
                        fill=True,
                        fill_color=palette[cluster_idx],
                        popup='Destination ' + str(cluster) + ' k=' + str(k)
                    ).add_to(m)

    m.save('stops_and_convexhull.html')

    potential_stops = {i:all_potential_stops[i].reshape(1, -1) for i in range(0, len(all_potential_stops))}

#    potential_stops[0] = 'DUMMY_STOPS'

    return potential_stops


def generate_potential_stops(coords, labels, area_covered_by_one_stop):

    def k_get_range(coords_2d, area_covered_by_one_stop=(500)**2):
        hull_area = ConvexHull(coords_2d).volume # in 2-D case .volume returns real area.

        max_k = math.ceil(hull_area / area_covered_by_one_stop)

        return np.arange(start=1, stop=max_k+1)

    def multi_k_means(coords, area_covered_by_one_stop):
        manhattan_metric = distance_metric(type_metric.MANHATTAN)
        k_range = k_get_range(coords, area_covered_by_one_stop)

        centers = {}
        for k in k_range:
            initial_centers = kmeans_plusplus_initializer(coords, k).initialize() # k-means ++
            kmeans_instance = kmeans(coords, initial_centers, metric=manhattan_metric) # L1

            kmeans_instance.process()
            final_centers = kmeans_instance.get_centers()
            centers[k] = np.stack(final_centers)
        return centers

    res = {}
    for cluster in set(labels):
        if cluster != -1:
            temp_coords = coords[np.argwhere(labels == cluster)].squeeze()
            coord_Os = temp_coords[:, :2]
            coord_Ds = temp_coords[:, 2:]

            O_centers = multi_k_means(coord_Os, area_covered_by_one_stop)
            D_centers = multi_k_means(coord_Ds, area_covered_by_one_stop)

            res[cluster] = {'O_centers': O_centers, 'D_centers': D_centers}
    return res


def calc_in_cluster_stats(labels, coords, timestamps, skip_critical=True):
    uniq_labels = np.unique(labels)
    stats = {}
    for label in uniq_labels:
        if label != -1:
            if skip_critical and label==0:
                continue
            indices = np.argwhere(labels==label)
            c_coords = coords[indices].squeeze()
            c_timestamps = timestamps[indices].squeeze().reshape(-1, 2)

            O_hull_area = ConvexHull(c_coords[:, :2]).volume # in 2-D case .volume returns real area.
            D_hull_area = ConvexHull(c_coords[:, 2:]).volume

            dist = pairwise_distances(c_coords[:, :2], metric='l2') \
                    + pairwise_distances(c_coords[:, 2:], metric='l2') # in m
            dist_mean, dist_std = np.mean(dist), np.std(dist)
            dist_percentile = list(map(np.percentile, repeat(dist), [10, 25, 50, 75, 100]))
            del dist

            diff_Ts = pairwise_distances(c_timestamps[:, [0]], metric='l1') * 1.66667e-11 \
                        + pairwise_distances(c_timestamps[:, [1]], metric='l1') * 1.66667e-11 # ns to mi
            diff_Ts_mean, diff_Ts_std = np.mean(diff_Ts), np.std(diff_Ts)
            diff_Ts_percentile = list(map(np.percentile, repeat(diff_Ts), [10, 25, 50, 75, 100]))
            del diff_Ts

            trip_length = np.linalg.norm(c_coords[:, :2] - c_coords[:, 2:], 2, axis=1)
            trip_length_mean, trip_length_std = np.mean(trip_length), np.std(trip_length)
            trip_length_percentile = list(map(np.percentile, repeat(trip_length), [10, 25, 50, 75, 100]))
            del trip_length

            counts = indices.shape[0]

            stats[label] = {'Dist_mean_std': [dist_mean, dist_std],
                            'Dist_percentile': dist_percentile,
                            'T_mean_std': [diff_Ts_mean, diff_Ts_std],
                            'T_percentile': diff_Ts_percentile,
                            'TripLen_mean_std': [trip_length_mean, trip_length_std],
                            'TripLen_percentile': trip_length_percentile,
                            'O_hull_area': O_hull_area,
                            'D_hull_area': D_hull_area,
                            'counts': counts}
    return stats


def cluster_map_visz(coords, labels, show_noise=False, show_od=False, only_o=False, skip_critical=False):
    if only_o:
        palette = color_brewer('YlGnBu', len(set(labels)))
        uniq_labels = np.unique(labels)
    else:
        palette = color_brewer('Dark2', 3)


    O = coords[:, :2]
    D = coords[:, 2:]
    O_latlon = [utm.to_latlon(each[0], each[1], 18, 'T') for each in O]
    D_latlon = [utm.to_latlon(each[0], each[1], 18, 'T') for each in D]


    m = folium.Map(
        location=[40.686892, -73.9876514],
        tiles='cartodbdark_matter',
        zoom_start=12
    )

    for loc, label in zip(O_latlon, labels):
        if show_noise:
            if label != -1:
                folium.Circle(
                    radius=50,
                    location=loc,
                    color=palette[1],
                    fill=True,
                    fill_color=palette[1],
                    popup='Origin ' + str(label)
                    ).add_to(m)

            else:
                folium.Circle(
                    radius=10,
                    location=loc,
                    color=palette[0],
                    fill=True,
                    fill_color=palette[0]
                    ).add_to(m)

        else:
            if label != -1:
                if skip_critical and label == 0:
                    continue
                if not only_o:
                    folium.Circle(
                        radius=50,
                        location=loc,
                        color=palette[1],
                        fill=True,
                        fill_color=palette[1],
                        popup='Origin ' + str(label)
                    ).add_to(m)
                else:
                    label_idx = np.argwhere(label==uniq_labels).squeeze()
                    folium.Circle(
                        radius=50,
                        location=loc,
                        color=palette[label_idx],
                        fill=True,
                        fill_color=palette[label_idx],
                        popup='Origin ' + str(label)
                    ).add_to(m)


    if not only_o:
        for loc, label in zip(D_latlon, labels):
            if show_noise:
                if label != -1:
                    folium.Circle(
                        radius=50,
                        location=loc,
                        color=palette[2],
                        fill=True,
                        fill_color=palette[2],
                        popup='Destination ' + str(label)
                        ).add_to(m)

                else:
                    folium.Circle(
                        radius=10,
                        location=loc,
                        color=palette[0],
                        fill=True,
                        fill_color=palette[0]
                        ).add_to(m)

            else:
                if label != -1:
                    if skip_critical and label == 0:
                        continue
                    folium.Circle(
                        radius=50,
                        location=loc,
                        color=palette[2],
                        fill=True,
                        fill_color=palette[2],
                        popup='Destination ' + str(label)
                    ).add_to(m)

    if show_od:
        for o, d, label, coord in zip(O_latlon, D_latlon, labels, coords):
            if label != -1:
                if skip_critical and label == 0:
                    continue

                dist = pairwise_distances(coord[:2].reshape(1, -1), coord[2:].reshape(1, -1), metric='l2')
                folium.PolyLine(
                    locations=[o, d],
                    color='#f5f5f5',
                    weight=1,
                    popup='Distance %.2f m' % dist
                ).add_to(m)

    m.save('cluster_map.html')


def cluster_visz(clusters, coords):
    """For visualizing canopy clustering results. Deprecated since now uses DBSCAN.
        Params:
            clusters: list[{'c': idx, 'points': list[idx]}]

        Returns:

    """

    idx_with_label = {}

    # -1 = centroid, 0 = overlap, >1 = unique clusters.
    for label, cluster in enumerate(clusters, 1):

        for point in cluster['points']:
            if point in idx_with_label:
                if idx_with_label[point] != -1:
                    idx_with_label[point] = 0

            else:
                idx_with_label[point] = label
        idx_with_label[cluster['c']] = -1


    fig, ax = plt.subplots()
    palette = sns.color_palette("hls", len(set(idx_with_label.values())))
    palette[0] = (0, 0, 0)
    sns.scatterplot(x=coords[list(idx_with_label.keys()), 0], y=coords[list(idx_with_label.keys()), 1],
                    hue=list(idx_with_label.values()), ax=ax, palette=palette)
    ax.set_title('Clusters on Origin')
    plt.show()


    fig, ax = plt.subplots()

    palette = sns.color_palette("Set1", n_colors=len(set(idx_with_label.values())), desat=.5)
    palette[0] = (0, 0, 0)
    sns.scatterplot(x=coords[list(idx_with_label.keys()), 2], y=coords[list(idx_with_label.keys()), 3],
                    hue=list(idx_with_label.values()), ax=ax, palette=palette)
    ax.set_title('Clusters on Destination')
    plt.show()

    return idx_with_label, np.unique(list(idx_with_label.values()), return_counts=True)[1]
