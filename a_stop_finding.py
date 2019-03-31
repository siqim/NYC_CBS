# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:56:15 2019

@author: msq96
"""


import pickle
import numpy as np

from a_dbscan_ import DBSCAN, filter_low_quality_cluster
from a_utils import cluster_map_visz, calc_in_cluster_stats, generate_potential_stops, visz_stops_and_convexhull


print('Reading file...')
with open('../data/yg_tripdata_2016-05.pickle', 'rb') as f:
    nyc_taxi = pickle.load(f)

subset = nyc_taxi[(nyc_taxi['pick_day']==11) & (nyc_taxi['pick_hour']>=8) & (nyc_taxi['pick_hour']<=9) & \
                  (nyc_taxi['drop_day']==11) & (nyc_taxi['drop_hour']>=8) & (nyc_taxi['drop_hour']<=9)] # & (nyc_taxi['type']=='green')]

all_coords = np.stack(subset['coords'])
all_timestamp = (subset['dropoff_datetime'].values).reshape(-1, 1)


print('Clustering...')
db = DBSCAN(eps_d=300, eps_t=10/1.66667e-11, min_samples=10, metric_d='l1', metric_t='l1').fit(all_coords, all_timestamp)
labels = filter_low_quality_cluster(db.labels_, thres=20)
print(list(zip(*np.unique(labels, return_counts=True))))

print('Visualizing clusters...')
cluster_map_visz(all_coords, labels, show_noise=False, show_od=True, only_o=False, skip_critical=False)

print('Computing statistics...')
stats = calc_in_cluster_stats(labels, all_coords, all_timestamp, skip_critical=False)

print('Generating potential stops...')
potential_stops = generate_potential_stops(all_coords, labels, area_covered_by_one_stop=(500)**2) # Assume one stop can cover users within 500 m (L1 distance).

print('Visualizing potential stops...')
potential_stops_with_id = visz_stops_and_convexhull(all_coords, labels, potential_stops)

print('Saving data...')
potential_users = subset[labels!=-1]
with open('../data/potential_users.pickle', 'wb') as f:
    pickle.dump(potential_users, f)
with open('../data/potential_stops_with_id.pickle', 'wb') as f:
    pickle.dump(potential_stops_with_id, f)

print('Phase 1 Finshed!')
