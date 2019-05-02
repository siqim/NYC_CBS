# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:51:15 2019

@author: msq96
"""


import re
import os
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import xml.etree.ElementTree as ET
from docplex.mp.model import Model

from utils import get_uniq_dummy_routes


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


def get_feasible_links(potential_routes, stop_time_mat):
    feasible_links = []

    for i in range(len(potential_routes) - 1):
        for j in range(i+1, len(potential_routes)):
            O = potential_routes[i][1]
            D = potential_routes[j][0]

            t = potential_routes[i][3]
            s = potential_routes[j][2]

            if stop_time_mat[O][D] + t <= s:
                feasible_links.append((O, D, t, s))
    return list(set(feasible_links))

def get_next_feasible(query_route, pool, stop_time_mat):

    for idx, pool_route in enumerate(pool):
        if pool_route[2] >= query_route[3] + stop_time_mat[query_route[1]][pool_route[0]]:
            return True, idx
    else:
        return False, None



data = pickle.load(open('../data/data.pickle', 'rb'))


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

potential_routes_and_rev = []
for sol in sols:
    O = sol['y'][1]
    D = sol['y'][-2]

    O[1] = D[1]
    O[3] = D[3]

    O[-1] = sol['obj']
    potential_routes_and_rev.append(O)

potential_routes_and_rev = np.stack(potential_routes_and_rev)

potential_routes = potential_routes_and_rev[potential_routes_and_rev[:, 2].argsort()][:, :-1]
rev_mat = {tuple(each[:-1]):each[-1] for each in potential_routes_and_rev}
feasible_links = get_feasible_links(potential_routes, data.stop_time_mat)


num_buses = 12
routes = {i:[] for i in range(num_buses)}

init_route_idx = list(range(num_buses))
[routes[i].append(potential_routes[init_route_idx[i]]) for i in range(num_buses)]

return_order = np.argsort(potential_routes[init_route_idx, 3])


potential_routes = np.delete(potential_routes, init_route_idx, axis=0)

while 1:

    temp_pool = []
    status_recorder = []
    for bus_id in return_order:
        status, fesible_route_idx = get_next_feasible(routes[bus_id][-1], potential_routes, data.stop_time_mat)
        status_recorder.append(status)

        if status:
            routes[bus_id].append(potential_routes[fesible_route_idx])
            temp_pool.append(potential_routes[fesible_route_idx])

            potential_routes = np.delete(potential_routes, fesible_route_idx, axis=0)

            if potential_routes.shape[0] == 0:
                break
        else:
            temp_pool.append(routes[bus_id][-1])


    if all(each==False for each in status_recorder) and potential_routes.shape[0] != 0:
        print('Fail!', num_buses)
        break

    elif potential_routes.shape[0] == 0:
        print('Success!', num_buses)
        break

    else:
        return_order = np.argsort(np.stack(temp_pool)[:, 3])





