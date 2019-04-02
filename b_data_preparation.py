# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:49:12 2019

@author: msq96
"""


import pickle
import numpy as np
from b_utils import generate_candidate_stop_id_for_each_user, get_time_units,\
                    generate_space_time_window, get_feasible_routes,\
                    get_uniq_feasible_routes, get_preferred_time_window,\
                    Data

print('Loading data...')
with open('../data/potential_users.pickle', 'rb') as f:
    potential_users = pickle.load(f)
with open('../data/potential_stops_with_id.pickle', 'rb') as f:
    potential_stops_with_id = pickle.load(f)
with open('../data/potential_stop_dist_time_mat.pickle', 'rb') as f:
    potential_stop_dist_mat, potential_stop_time_mat = pickle.load(f)

print('Processing data...')
potential_users['trip_time'] = potential_users[['pickup_datetime', 'dropoff_datetime']].apply(lambda x:\
                               int(np.round(abs(x[1]-x[0]).total_seconds()/60)), axis=1)

potential_users['candidate_pick_up_loc'] = potential_users['coords'].apply(lambda coords:\
               generate_candidate_stop_id_for_each_user(np.array(coords[:2]).reshape(1, -1), potential_stops_with_id, thres=500))
potential_users['candidate_drop_off_loc'] = potential_users['coords'].apply(lambda coords:\
               generate_candidate_stop_id_for_each_user(np.array(coords[2:]).reshape(1, -1), potential_stops_with_id, thres=500))

# although we consider people in 8-9 am, we need to enlarge time window allowed
# such that we can serve people who might want to get on the bus before 8 am or after 10 am.
time_units = get_time_units(7, 10)

potential_users['drop_off_space_time_window'] = potential_users[['candidate_drop_off_loc', 'dropoff_datetime']].apply(lambda x:\
                                                generate_space_time_window(x[0], x[1], time_units, thres=5), axis=1)

potential_users['pick_up_space_time_window'] = potential_users[['candidate_pick_up_loc', 'pickup_datetime']].apply(lambda x:\
                                               generate_space_time_window(x[0], x[1], time_units, thres=5), axis=1)


potential_users['feasible_routes'] = potential_users[['pick_up_space_time_window', 'drop_off_space_time_window']].apply(lambda x:\
                                     get_feasible_routes(x[0], x[1], potential_stop_time_mat), axis=1)
uniq_feasible_routes = get_uniq_feasible_routes(potential_users['feasible_routes'])


potential_users['preferred_time_window'] = potential_users[['pickup_datetime', 'dropoff_datetime']].apply(lambda x:\
                                           get_preferred_time_window(x[0], x[1], time_units), axis=1)


data = Data(potential_stops_with_id, time_units, uniq_feasible_routes, potential_stop_dist_mat, potential_stop_time_mat, potential_users)

print('Saving data...')
with open('../data/data.pickle', 'wb') as fout:
    pickle.dump(data, fout)
