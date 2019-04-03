# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:49:31 2019

@author: msq96
"""


import utm
from copy import deepcopy
import pickle
import datetime
import googlemaps
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import trange


def generate_candidate_stop_id_for_each_user(user_loc, potential_stops_with_id, thres=500):
    flags = []
    for stop_id, stop_loc in potential_stops_with_id.items():
        if stop_id != 0:
            if pairwise_distances(stop_loc, user_loc, metric='l1') <= thres:
                flags.append(stop_id)
    return flags

def get_time_units(start=8, end=9, year=2016, month=5, day=11):

    time_span = end - start + 1
    num_time_units = 60 * time_span
    start_time_unit = datetime.datetime(year, month, day, start)

    unit_ids = np.arange(0, num_time_units)
    return {unit_id:start_time_unit + datetime.timedelta(minutes=idx) for idx, unit_id in enumerate(unit_ids)}

def generate_space_time_window(candidate_loc, preferred_time, time_units, thres=5):
    space_time_window = []
    for time_unit_id, time_unit in time_units.items():
        if int(np.round(abs(time_unit-preferred_time).total_seconds()/60)) <= thres:
            space_time_window.extend([[each_loc, time_unit_id] for each_loc in candidate_loc])
    return space_time_window

def dist_mat_potential_stops(dist_time_mat_filename, potential_stops_with_id, key, year=2019, month=5, day=11, hour=9, minute=0):
    latlon = np.empty((len(potential_stops_with_id)), dtype=object)
    for idx, loc_latlon in potential_stops_with_id.items():
        if idx != 0:
            projected_latlon = loc_latlon.reshape(-1)
            latlon[idx] = utm.to_latlon(projected_latlon[0], projected_latlon[1], 18, 'T')
        else:
            latlon[idx] = loc_latlon

    num_stops = len(latlon)
    gmaps = googlemaps.Client(key=key)
    departure_time = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)
    dist_mat = np.zeros((num_stops, num_stops))
    time_mat = np.zeros((num_stops, num_stops))
    for i in trange(num_stops, desc='1st loop'):
        for j in trange(num_stops, desc='2nd loop', leave=False):
            if j > i and j != 0 and i != 0:
                time_dist_dict = gmaps.distance_matrix(origins=latlon[i],
                                                 destinations=latlon[j],
                                                 mode="driving",
                                                 departure_time=departure_time,
                                                 traffic_model='best_guess')
                time_needed = int(np.round(time_dist_dict['rows'][0]['elements'][0]['duration_in_traffic']['value'] / 60))
                dist = time_dist_dict['rows'][0]['elements'][0]['distance']['value']
                time_mat[i][j] = time_needed
                dist_mat[i][j] = dist

    dist_mat = np.transpose(dist_mat) + dist_mat
    dist_mat = dist_mat.astype(np.int32)

    time_mat = np.transpose(time_mat) + time_mat
    time_mat = time_mat.astype(np.int32)

    with open(dist_time_mat_filename, 'wb') as fout:
        pickle.dump([dist_mat, time_mat], fout)

#def get_feasible_routes(pick_up_space_time_window, drop_off_space_time_window, potential_stop_time_mat):
#    all_routes = [[each_p[0], each_d[0], each_p[1], each_d[1]] for each_p in pick_up_space_time_window for each_d in drop_off_space_time_window]
#    feasible_routes = []
#    for route in all_routes:
#        if route[2] != route[3]:
#            time_needed = potential_stop_time_mat[route[0], route[1]]
#            if time_needed == route[-1] - route[-2]:
#                feasible_routes.append(route)
#    return feasible_routes

#def get_uniq_feasible_routes(all_feasible_routes):
#    uniq_feasible_routes = []
#    for each_user_routes in tqdm(all_feasible_routes):
#        for each_route in each_user_routes:
#            if each_route not in uniq_feasible_routes: # slow
#                uniq_feasible_routes.append(each_route)
#    return uniq_feasible_routes

#def get_uniq_feasible_routes(all_feasible_routes):
#    uniq_feasible_routes = {}
#    for each_user_routes in all_feasible_routes:
#        for each_route in each_user_routes:
#            if uniq_feasible_routes.get(tuple(each_route), 0) == 0:
#                uniq_feasible_routes[tuple(each_route)] = 1
#    return list(uniq_feasible_routes.keys())

def get_uniq_dummy_routes(dummy_routes):
    uniq_dummy_routes = {}
    for each_route in dummy_routes:
        if uniq_dummy_routes.get(each_route, 0) == 0:
            uniq_dummy_routes[each_route] = 1
    return list(uniq_dummy_routes.keys())


# TODO round to nearest minute.
def get_preferred_time_window(pickup_time, dropoff_time, time_units):
    for time_unit_id, time_unit in time_units.items():
        if time_unit == pickup_time.replace(second=0):
            pickup_time_unit = deepcopy(time_unit_id)
        elif time_unit == dropoff_time.replace(second=0):
            dropoff_time_unit = deepcopy(time_unit_id)
    return [pickup_time_unit, dropoff_time_unit]

def get_worst_time_window(pick_up_space_time_window, drop_off_space_time_window):
    p = []
    for each_window in pick_up_space_time_window:
        p.append(each_window[1])
    if p == []:
        return -1
    else:
        min_p = min(p)

    d = []
    for each_window in drop_off_space_time_window:
        d.append(each_window[1])
    if d == []:
        return -1
    else:
        max_d = max(d)

    return [min_p, max_d]

def get_A_O(pick_up_space_time_window, worst_time_window, stop_time_mat, candidate_drop_off_loc):
    if worst_time_window == -1:
        return []

    A_O = []
    latest = worst_time_window[1]
    for i, t in pick_up_space_time_window:
        for j in range(1, stop_time_mat.shape[0]):
            s = t + stop_time_mat[i, j]
            delta = np.min([stop_time_mat[j, each_dropoff] for each_dropoff in candidate_drop_off_loc])
            if s + delta <= latest and t!=s:
                A_O.append([i,j,t,s])
    return A_O

def get_A_D(drop_off_space_time_window, worst_time_window, stop_time_mat, candidate_pick_up_loc):
    if worst_time_window == -1:
        return []

    A_D = []
    earliest = worst_time_window[0]
    for j, s in drop_off_space_time_window:
        for i in range(1, stop_time_mat.shape[0]):
            t = s - stop_time_mat[i, j]
            delta = np.min([stop_time_mat[each_pickup, i] for each_pickup in candidate_pick_up_loc])
            if t - delta >= earliest and t!=s:
                A_D.append([i,j,t,s])
    return A_D


class Data(object):

    def __init__(self, stop_locs, time_units, routes, stop_dist_mat, stop_time_mat, potential_users_df, T_min, T_max, dummy_stop_id):
        self.stop_locs = stop_locs
        self.time_units = time_units
        self.routes = routes
        self.stop_dist_mat = stop_dist_mat
        self.stop_time_mat = stop_time_mat
        self.users_df = potential_users_df
        self.dummy_stop_id = dummy_stop_id
        self.T_min = T_min
        self.T_max = T_max

def nested_list_to_tuple(nested_list):
    return [tuple(each_list) for each_list in nested_list]

if __name__ == '__main__':
    key = '***'
    year=2019
    month=5
    day=11
    hour=9
    minute=0
    with open('../data/potential_stops_with_id.pickle', 'rb') as f:
        potential_stops_with_id = pickle.load(f)

    dist_time_mat_filename = '../data/potential_stop_dist_time_mat.pickle'
    dist_mat_potential_stops(dist_time_mat_filename, potential_stops_with_id, key, year, month, day, hour, minute)
