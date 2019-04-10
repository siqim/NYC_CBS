# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:49:12 2019

@author: msq96
"""


import pickle
import numpy as np
from utils import generate_candidate_stop_id_for_each_user, get_time_units,\
                    generate_space_time_window, get_preferred_time_window,\
                    get_uniq_dummy_routes, Data, get_worst_time_window, get_A_O, get_A_D

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
               generate_candidate_stop_id_for_each_user(np.array(coords[:2]).reshape(1, -1), potential_stops_with_id, thres=700))
potential_users['candidate_drop_off_loc'] = potential_users['coords'].apply(lambda coords:\
               generate_candidate_stop_id_for_each_user(np.array(coords[2:]).reshape(1, -1), potential_stops_with_id, thres=700))

# although we consider people in 8-9 am, we need to enlarge time window allowed
# such that we can serve people who might want to get on the bus before 8 am or after 10 am.
time_units = get_time_units(7, 10)

potential_users['pick_up_space_time_window'] = potential_users[['candidate_pick_up_loc', 'pickup_datetime']].apply(lambda x:\
                                               generate_space_time_window(x[0], x[1], time_units, thres=10), axis=1)
potential_users['drop_off_space_time_window'] = potential_users[['candidate_drop_off_loc', 'dropoff_datetime']].apply(lambda x:\
                                                generate_space_time_window(x[0], x[1], time_units, thres=10), axis=1)

potential_users['preferred_time_window'] = potential_users[['pickup_datetime', 'dropoff_datetime']].apply(lambda x:\
                                           get_preferred_time_window(x[0], x[1], time_units), axis=1)
potential_users['worst_time_window'] = potential_users[['pick_up_space_time_window', 'drop_off_space_time_window']].apply(lambda x:\
                                       get_worst_time_window(x[0], x[1]), axis=1)


potential_users['A_O'] = potential_users[['pick_up_space_time_window', 'worst_time_window', 'candidate_drop_off_loc']].apply(lambda x:\
                         get_A_O(x[0], x[1], potential_stop_time_mat, x[2]), axis=1)

potential_users['A_D'] = potential_users[['drop_off_space_time_window', 'worst_time_window', 'candidate_pick_up_loc']].apply(lambda x:\
                         get_A_D(x[0], x[1], potential_stop_time_mat, x[2]), axis=1)

all_t = [ each_window[1] for each_users_window in potential_users['pick_up_space_time_window'] for each_window in each_users_window ]
all_s = [ each_window[1] for each_users_window in potential_users['drop_off_space_time_window'] for each_window in each_users_window ]

t_min = np.min(all_t)
t_max = np.max(all_t)
s_min = np.min(all_s)
s_max = np.max(all_s)
T_min = np.min([t_min, s_min])
T_max = np.max([t_max, s_max])

num_stops = potential_stop_dist_mat.shape[0]

#all_routes = [(i,j,t,s) for i in range(1, num_stops) for j in range(1, num_stops) for t in range(T_min, T_max+1) for s in range(T_min, T_max+1)\
#                       if potential_stop_time_mat[i,j]==s-t and t!=s]
#dummy_routes_O = get_uniq_dummy_routes([(0, route[1], T_min, route[3]) for route in all_routes])
#dummy_routes_D = get_uniq_dummy_routes([(route[0], 0, route[2], T_max) for route in all_routes])

#routes = all_routes + dummy_routes_O + dummy_routes_D
#assert len(routes) == len(dummy_routes_D) + len(dummy_routes_O) + len(all_routes)


all_routes = [(i,j,t,s) for i in range(1, num_stops) for j in range(1, num_stops) for t in range(T_min, T_max+1) for s in range(T_min, T_max+1)\
              if (i!=j and i!=0 and j!=0 and potential_stop_time_mat[i,j]==s-t) or (i==j and s>t)]

V = [(i,t) for i in range(potential_stop_dist_mat.shape[0]) for t in range(T_min, T_max+1)]

dummy_routes_O = get_uniq_dummy_routes([(0, i, T_min, t) for i,t in V if (i!=0 and t>=T_min) or (i==0 and t==T_max)])
dummy_routes_D = get_uniq_dummy_routes([(j, 0, s, T_max) for j,s in V if (j!=0 and T_max>=s) or (j==0 and s==T_min)])
routes = all_routes + dummy_routes_O + dummy_routes_D
routes.remove(((0, 0, T_min, T_max)))

potential_users['coords'] = [np.array(each).reshape(1, -1) for each in potential_users['coords']]

data = Data(potential_stops_with_id, time_units, routes, potential_stop_dist_mat, potential_stop_time_mat, potential_users, T_min, T_max, dummy_stop_id=0)

print('Saving data...')
with open('../data/data.pickle', 'wb') as fout:
    pickle.dump(data, fout)
