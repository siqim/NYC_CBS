# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:49:31 2019

@author: msq96
"""


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

def get_uniq_dummy_routes(dummy_routes):
    uniq_dummy_routes = {}
    for each_route in dummy_routes:
        if uniq_dummy_routes.get(each_route, 0) == 0:
            uniq_dummy_routes[each_route] = 1
    return list(uniq_dummy_routes.keys())