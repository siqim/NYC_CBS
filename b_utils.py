# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:49:31 2019

@author: msq96
"""


import utm
import pickle
import datetime
import googlemaps
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import trange


def generate_candidate_stop_id_for_each_user(user_loc, potential_stops_with_id, thres=500):
    flags = []
    for stop_id, stop_loc in potential_stops_with_id.items():
        if pairwise_distances(stop_loc, user_loc, metric='l1') <= thres:
            flags.append(stop_id)
    return flags

def get_time_units(start=8, end=9, year=2016, month=5, day=11):

    time_span = end - start + 1
    num_time_units = 60 * time_span
    start_time_unit = datetime.datetime(year, month, day, start)

    unit_ids = np.arange(0, num_time_units)
    return {unit_id:start_time_unit + datetime.timedelta(minutes=idx) for idx, unit_id in enumerate(unit_ids)}

def generate_drop_off_space_time_window(candidate_loc, time, time_units, thres=5):
    time_window = []
    for time_unit_id, time_unit in time_units.items():
        if abs(time_unit-time).total_seconds()//60 <= thres:
            time_window.extend([[each_loc, time_unit_id] for each_loc in candidate_loc])
    return time_window


# TODO: allow users to get on the bus earlier
def infer_pick_up_space_time_window(candidate_loc, trip_time, drop_off_space_time_window):
    drop_off_time_window = (np.array(list(set([each[1] for each in drop_off_space_time_window]))) - trip_time).astype(np.int).tolist()
    time_window = [[each_loc, each_time_window] for each_loc in candidate_loc for each_time_window in drop_off_time_window]
    return time_window

def dist_mat_potential_stops(dist_mat_file, potential_stops_with_id, year, month, day, hour, minute):
    latlon = []
    for value in potential_stops_with_id.values():
        projected_latlon = value.reshape(-1)
        latlon.append(utm.to_latlon(projected_latlon[0], projected_latlon[1], 18, 'T'))

    num_stops = len(latlon)
    gmaps = googlemaps.Client(key='***')
    departure_time = datetime.datetime(year=2019, month=5, day=11, hour=9, minute=0)
    dist_mat = np.zeros((num_stops, num_stops))
    for i in trange(num_stops, desc='1st loop'):
        for j in trange(num_stops, desc='2nd loop', leave=False):
            if j > i:
                dist_dict = gmaps.distance_matrix(origins=latlon[i],
                                                 destinations=latlon[j],
                                                 mode="driving",
                                                 departure_time=departure_time,
                                                 traffic_model='best_guess')
                dist = dist_dict['rows'][0]['elements'][0]['duration_in_traffic']['value'] // 60
                dist_mat[i][j] = dist

    dist_mat = np.transpose(dist_mat) + dist_mat

    dist_mat = dist_mat.astype(np.int)

    with open(dist_mat_file, 'wb') as fout:
        pickle.dump(dist_mat, fout)
    return dist_mat


class Data(object):

    def __init__(self, stop_locs, stop_pickup_time_unit, stop_dropoff_time_unit, stop_dist_mat, potential_users_df):
        self.stop_locs = stop_locs
        self.stop_pickup_time_unit = stop_pickup_time_unit
        self.stop_dropoff_time_unit = stop_dropoff_time_unit
        self.stop_dist_mat = stop_dist_mat
        self.users_df = potential_users_df


def get_unit_space_time_window(space_time_window):
    space_time_window = space_time_window.tolist()
    space_time_window = [each_window for each_user in space_time_window for each_window in each_user]
    uniq_space_time_window = []
    for each in space_time_window:
        if each not in uniq_space_time_window:
            uniq_space_time_window.append(each)
    uniq_space_time_window = np.stack(uniq_space_time_window)
    return uniq_space_time_window


def get_feasible_time_unit(potential_users, potential_stops_with_id):
    uniq_pick_up_space_time_window = get_unit_space_time_window(potential_users['pick_up_space_time_window'])
    uniq_drop_off_space_time_window = get_unit_space_time_window(potential_users['drop_off_space_time_window'])

    stop_pickup_time_unit = {}
    stop_dropoff_time_unit = {}
    for stop_id in potential_stops_with_id.keys():
        feasible_pick_up_time_unit = uniq_pick_up_space_time_window[np.argwhere(uniq_pick_up_space_time_window[:, 0]==stop_id)].squeeze()[:, 1].tolist()
        feasible_drop_off_time_unit = uniq_drop_off_space_time_window[np.argwhere(uniq_drop_off_space_time_window[:, 0]==stop_id)].squeeze()[:, 1].tolist()

        stop_pickup_time_unit[stop_id] = feasible_pick_up_time_unit
        stop_dropoff_time_unit[stop_id] = feasible_drop_off_time_unit

    return stop_pickup_time_unit, stop_dropoff_time_unit



