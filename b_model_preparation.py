# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:49:12 2019

@author: msq96
"""


import pickle
import numpy as np
from b_utils import generate_candidate_stop_id_for_each_user, get_time_units,\
                    generate_drop_off_space_time_window, infer_pick_up_space_time_window,\
                    get_feasible_time_unit, Data

print('Loading data...')
with open('../data/potential_users.pickle', 'rb') as f:
    potential_users = pickle.load(f)
with open('../data/potential_stops_with_id.pickle', 'rb') as f:
    potential_stops_with_id = pickle.load(f)
with open('../data/potential_stop_dist_mat.pickle', 'rb') as f:
    potential_stop_dist_mat = pickle.load(f)


potential_users['candidate_pick_up_loc'] = potential_users['coords'].apply(lambda coords:\
               generate_candidate_stop_id_for_each_user(np.array(coords[:2]).reshape(1, -1), potential_stops_with_id, 500))
potential_users['candidate_drop_off_loc'] = potential_users['coords'].apply(lambda coords:\
               generate_candidate_stop_id_for_each_user(np.array(coords[2:]).reshape(1, -1), potential_stops_with_id, 500))


time_units = get_time_units(7, 10)

potential_users['drop_off_space_time_window'] = potential_users[['candidate_drop_off_loc', 'dropoff_datetime']].apply(lambda x:\
               generate_drop_off_space_time_window(x[0], x[1], time_units, thres=5), axis=1)


potential_users['trip_time'] = potential_users[['pickup_datetime', 'dropoff_datetime']].apply(lambda x: abs(x[0]-x[1]).total_seconds()//60, axis=1)
potential_users['pick_up_space_time_window'] = potential_users[['candidate_pick_up_loc', 'trip_time', 'drop_off_space_time_window']].apply(lambda x:\
               infer_pick_up_space_time_window(x[0], x[1], x[2]), axis=1)

stop_pickup_time_unit, stop_dropoff_time_unit = get_feasible_time_unit(potential_users, potential_stops_with_id)

data = Data(potential_stops_with_id, stop_pickup_time_unit, stop_dropoff_time_unit, potential_stop_dist_mat, potential_users)

with open('../data/data.pickle', 'wb') as fout:
    pickle.dump(data, fout)
