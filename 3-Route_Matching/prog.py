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

potential_routes = potential_routes_and_rev[potential_routes_and_rev[:, 3].argsort()][:, :-1]
potential_routes = [tuple(each) for each in potential_routes]
rev_mat = {tuple(each[:-1]):each[-1] for each in potential_routes_and_rev}
feasible_links = get_feasible_links(potential_routes, data.stop_time_mat)


all_links = potential_routes + feasible_links
V = list(set([(i,t) for i,_,t,_ in all_links] + [(j,s) for _,j,_,s in all_links]))
all_loc = list(set([each[0] for each in V]))


all_routes = [(i,j,t,s) for i in all_loc for j in all_loc for t in range(data.T_min, data.T_max+1) for s in range(data.T_min, data.T_max+1)\
              if (i!=j and i!=0 and j!=0 and data.stop_dist_mat[i,j]==s-t) or (i==j and s>t)]

all_routes = list(set(all_routes + all_links))

all_loc = all_loc + [0]
V = [(i,t) for i in all_loc for t in range(data.T_min, data.T_max+1)]


dummy_routes_O = get_uniq_dummy_routes([(0, i, data.T_min, t) for i,t in V if (i!=0 and t>=data.T_min) or (i==0 and t==data.T_max)])
dummy_routes_D = get_uniq_dummy_routes([(j, 0, s, data.T_max) for j,s in V if (j!=0 and data.T_max>=s) or (j==0 and s==data.T_min)])
all_routes = list(set(all_routes + dummy_routes_O + dummy_routes_D))



#########################################
#########################################


gamma = 9 * 1e-3 # operational cost per meter
mdl = Model(name='CB-Planning_Route_Matching')
mdl.parameters.threads = 8
mdl.parameters.timelimit = 60 * 60
mdl.parameters.emphasis.mip = 0

num_available_buses = 5
K = [k for k in range(num_available_buses)]
all_y = [(i,j,t,s,k) for k in K for i,j,t,s in all_routes]
y = mdl.binary_var_dict(all_y, name='y')

mdl.maximize( mdl.sum(y[i,j,t,s,k] * rev_mat[i,j,t,s] for k in K for i,j,t,s in potential_routes)
              - mdl.sum(y[i,j,t,s,k] * data.stop_dist_mat[i,j] * gamma for k in K for i,j,t,s in feasible_links) )



mdl.add_constraints(mdl.sum( y[0,j,data.T_min,s,k] for j,s in V if (j!=0 and s>= data.T_min) or (j==0 and s==data.T_max) ) \
                        -mdl.sum( y[j,0,s,data.T_min,k] for j,s in V if data.T_min == data.T_max)  == 1 for k in K)

mdl.add_constraints(mdl.sum( y[0,j,data.T_max,s,k] for j,s in V if data.T_max==data.T_min) \
                        -mdl.sum( y[j,0,s,data.T_max,k] for j,s in V if (j!=0 and data.T_max>= s) or (j==0 and s==data.T_min) )  == -1 for k in K)


mdl.add_constraints(mdl.sum( y.get((i,j,t,s,k),0) for j,s in V \
                            if  (
                                    (i!=0 and j!=0 and i!=j and s-t>=data.stop_time_mat[i,j]) \
                                    or (i==0 and j!=0 and s>=t and t==data.T_min) \
                                    or (j==0 and i!=0 and s>=t and s==data.T_max) \
                                    or (i==j and i!=0 and s>t)\
                                    or (i==j and i==0 and t==data.T_min and s==data.T_max)
                                )
                            )
                        -mdl.sum( y.get((j,i,s,t,k),0) for j,s in V \
                            if  (
                                    (i!=0 and j!=0 and i!=j and t-s>=data.stop_time_mat[j,i]) \
                                    or (i==0 and j!=0 and t>=s and t==data.T_max) \
                                    or (j==0 and i!=0 and t>=s and s==data.T_min) \
                                    or (i==j and i!=0 and t>s) \
                                    or (i==j and i==0 and s==data.T_min and t==data.T_max)
                                )
                            ) == 0 \
                   for k in K for i,t in V if (not (i==0 and t==data.T_min)) and (not (i==0 and t==data.T_max))  )

mdl.add_constraints(mdl.sum(y[i,j,t,s,k] for k in K) <= 1 for i,j,t,s in potential_routes)

solution = mdl.solve(log_output=True)


with open('./solution.xml', 'w') as f:
    f.write(solution.export_as_mst_string())


t = solution.get_value_dict(y)
t = [k for k,v in t.items() if v==1]

a1 = np.stack([each for each in t if each[-1]==1])

a1 = join_route(a1[a1[:,2].argsort()])
