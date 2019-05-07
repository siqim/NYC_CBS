# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:49:31 2019

@author: msq96
"""


import utm
import pickle
import folium
import numpy as np

import os
import re
from copy import  deepcopy
from tqdm import tqdm
import xml.etree.ElementTree as ET


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


def join_route(y):
    route = []

    for i in range(y.shape[0]):
        if i == 0:
            temp = y[i]

        else:
            if y[i][0] == y[i][1] == temp[0] == temp[1]:
                temp[3] = y[i][3]
            else:
                route.append(temp)
                temp = y[i]
    route.append(temp)

    if len(route) >= 2:

        if route[0][1] == route[1][0] == route[1][1]:
            route[0][3] = route[1][3]
            route.pop(1)

        if route[-1][0] == route[-2][0] == route[-2][1]:
            route[-1][2] = route[-2][2]
            route.pop(-2)

    return route


def visz(routes, stop_locs, show_idx=None):

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']


    m = folium.Map(
        location=[40.686892, -73.9876514],
        tiles='cartodbdark_matter',
        zoom_start=12
    )

    for idx, route in enumerate(routes):
        if show_idx is not None:
            if idx != show_idx:
                continue
        for stop in route:
            if stop[0]==0 or stop[1]==0:
                continue
            else:
                O = stop_locs[stop[0]].squeeze()
                D = stop_locs[stop[1]].squeeze()

                O = utm.to_latlon(O[0], O[1], 18, 'T')
                D = utm.to_latlon(D[0], D[1], 18, 'T')

                end_time = stop[-1]
                start_time = stop[-2]
                travel_time = end_time - start_time

                folium.PolyLine(
                    locations=[O, D],
                    color=colors[idx],
                    weight=2,
                    popup='Start Time %d, End Time %d, Travel Time %.2f min' % (start_time, end_time,travel_time)
                ).add_to(m)

                folium.Circle(
                    radius=30,
                    location=O,
                    color='lightblue',
                    fill=True,
                    fill_color='lightblue',
                    popup='Stop id %d, Time %d'%(stop[0], start_time)
                ).add_to(m)

                folium.Circle(
                    radius=30,
                    location=D,
                    color='lightblue',
                    fill=True,
                    fill_color='lightblue',
                    popup='Stop id %d, Time %d'%(stop[0], end_time)
                ).add_to(m)

    m.save('./solution_%d.html'%show_idx)



if __name__ == '__main__':

    with open('./solutions/solution_5.pk', 'rb') as f:
        sol = pickle.load(f)

    data = pickle.load(open('../data/data.pickle', 'rb'))


    sol = [k for k,v in sol['y'].items() if v==1]

    sol = np.stack(sol)

    route_ids = np.unique(sol[:, -1])

    routes = []
    for route_id in route_ids:
        route = sol[np.argwhere(sol[:,-1]==route_id)].squeeze()[:,:-1]

        route = route[route[:,2].argsort()]

        route = join_route(route)

        routes.append(route)



    dir_sol = '../solutions/'
    solnames = os.listdir(dir_sol)
    solnames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    sols = []
    objs = []
    for solname in tqdm(solnames):
        if '.pk' in solname:
            sol = pickle.load(open(dir_sol + solname, 'rb'))
            y = [k for k, v in sol['y'].items() if v==1 and k[-1]!=0]

            y = np.stack(y) if y!=[] else y

            temp = deepcopy(y[0])
            init_idx = np.argwhere(y[:, 0]==0)
            y[0] = deepcopy(y[init_idx])
            y[init_idx] = deepcopy(temp)


            y = y[y[:, 2].argsort()]
            y = join_route(y)

            sols.append({'y':y})

        else:
            tree = ET.parse(dir_sol+solname)
            root = tree.getroot()
            objs.append(float(root[0].attrib['objectiveValue']))

    for i in range(len(objs)):
        sols[i]['obj'] = objs[i]

    sols = [sol for sol in sols if sol['obj'] != 0]

    potential_routes = {}
    for sol in sols:
        O = deepcopy(sol['y'][1])
        D = deepcopy(sol['y'][-2])

        O[1] = deepcopy(D[1])
        O[3] = deepcopy(D[3])

        potential_routes[tuple(O[:-1])] = np.stack(sol['y'])[1:-1, :-1]

    for i, route in enumerate(routes):
        for j, line in enumerate(route):
            fill_line = potential_routes.get(tuple(line), None)
            if fill_line is not None:
                routes[i][j] = fill_line

    res_routes = []
    for route in routes:
        res = []
        for line in route:
            temp = line.tolist()
            try:
                if np.array(temp).shape[1] > 1:
                    res = res + temp
            except IndexError:
                res = res + [temp]

        res = np.stack(res)
        res = res[res[:,2].argsort()]
        res_routes.append(join_route(res))



    [visz(res_routes, data.stop_locs, show_idx=i) for i in range(0, 5)]

