# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:02:27 2019

@author: msq96
"""


import time
import pickle
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances

from docplex.mp.model import Model

from b_utils import get_uniq_dummy_routes


with open('../data/data.pickle', 'rb') as f:
    data = pickle.load(f)

num_potential_users = len(data.users_df)
num_available_buses = 3
rho = 3 * 1e-3 # ticket price per meter
gamma = 9 * 1e-3 # operational cost per meter
phi = 20 # vehicle capacity

beta_0c, beta_1c, beta_2c, beta_3c, beta_4c = 0.969, -0.007, -0.14, -0.241, -0.196
beta_0t, beta_1t, beta_2t, beta_3t, beta_4t = 0, 0, 0, -0.327, -0.145


P = [p for p in range(num_potential_users)]
K = [k for k in range(num_available_buses+1)] # k=0 is the virtual bus.
A = deepcopy(data.routes)
V_O = get_uniq_dummy_routes([(i,t) for i,_,t,_ in A])
V_D = get_uniq_dummy_routes([(j,s) for _,j,_,s in A])
V = list(set(V_O).intersection(set(V_D)))


all_x = [(p, k) for p in P for k in K]
all_y = [(i,j,t,s,k) for k in K for i,j,t,s in A]



mdl = Model(name='CB-Planning')

x = mdl.binary_var_dict(all_x, name='x')
y = mdl.binary_var_dict(all_y, name='y')

def mu(O, D, t_o, t_d, p):

    OD_coords = data.users_df.iloc[p]['coords']
    walk_dist = 1/2 * (pairwise_distances(OD_coords[:, :2], data.stop_locs[O], metric='l1') + pairwise_distances(OD_coords[:, 2:], data.stop_locs[D], metric='l1'))

    preferred_time_window = data.users_df.iloc[p]['preferred_time_window']
    time_adj = abs(preferred_time_window[0] - t_o) + abs(preferred_time_window[1] - t_d)

    travel_time = t_d - t_o

    fare = rho * data.stop_dist_mat[O, D]

    exp_mu_c = np.exp(beta_0c + beta_1c*walk_dist + beta_2c*time_adj + beta_3c*travel_time + beta_4c*fare)
    exp_mu_t = np.exp(beta_0t + beta_1t*walk_dist + beta_2t*time_adj + beta_3t*travel_time + beta_4t*fare)

    score = exp_mu_c / (exp_mu_c + exp_mu_t)

    return score.item()

def dist(p, k):
    if k == 0:
        return 0
    else:
        possible_Os = data.users_df.iloc[p]['candidate_pick_up_loc']
        possible_Ds = data.users_df.iloc[p]['candidate_drop_off_loc']
        O = 0
        D = 0
        t_o = data.T_min
        t_d = data.T_max
        for i, j, s, t in A:
            if y[i,j,s,t,k]:
                if i in possible_Os:
                    O = i
                    t_o = s
                if j in possible_Ds:
                    D = j
                    t_d = t
        if O == 0 or D == 0:
            return 0
        else:
            return data.stop_dist_mat[O, D] * mu(O, D, t_o, t_d, p)


print('Creating models...')
start_0 = time.time()
mdl.maximize( mdl.sum(x[p,k] * dist(p,k) * rho for p in P for k in K )
                - mdl.sum(y[i,j,s,t,k] * data.stop_dist_mat[i,j] * gamma for k in K for i,j,s,t in A) )
end_0 = time.time()
print('[%.2f] min used till now.' % (end_0-start_0)/60)


print('Adding the first four constrations...')
start_1 = time.time()
mdl.add_constraints( mdl.sum(x[p,k] for k in K)==1 for p in P)

mdl.add_constraints( mdl.sum(x[p,k] for p in P) <= phi  for k in K if k!=0)

mdl.add_constraints( x[p,k] <= mdl.sum( y[i,j,t,s,k] \
                    for i,j,t,s in data.users_df.iloc[p]['A_O']) for p in P for k in K if k != 0 )

mdl.add_constraints( x[p,k] <= mdl.sum( y[i,j,t,s,k] \
                    for i,j,t,s in data.users_df.iloc[p]['A_D']) for p in P for k in K if k != 0 )
end_1 = time.time()
print('[%.2f] min used till now.' % (end_1-start_1)/60)


print('Adding the last three constrations...')
start_2 = time.time()
mdl.add_constraints(mdl.sum( y[data.dummy_stop_id,j,data.T_min,s,k] for j,s in V_D if j!=data.dummy_stop_id and s>data.T_min) \
                   - mdl.sum( y[j,data.dummy_stop_id,s,data.T_min,k] for j,s in V_O if j!=data.dummy_stop_id and s<data.T_min)  == 1 for k in K)

mdl.add_constraints(mdl.sum( y[data.dummy_stop_id,j,data.T_max,s,k] for j,s in V_D if j!=data.dummy_stop_id and s>data.T_max) \
                   - mdl.sum( y[j,data.dummy_stop_id,s,data.T_max,k] for j,s in V_O if j!=data.dummy_stop_id and s<data.T_max)  == -1 for k in K)

mdl.add_constraints(mdl.sum( y[i,j,t,s,k] for j,s in V_D if s>t and i!=j and s-t==data.stop_time_mat[i,j]) \
                   - mdl.sum( y[j,i,s,t,k] for j,s in V_O if t>s and i!=j and t-s==data.stop_time_mat[j,i])  == 0 for k in K for i,t in V \
                       if i!=data.dummy_stop_id and t!=data.T_min and t!=data.T_max)
end_2 = time.time()
print('[%.2f] min used till now.' % (end_2-start_2)/60)

print('Solving...')
mdl.parameters.timelimit = 1000
solution = mdl.solve(log_output=True)
print(solution.solve_status)
print('[%.2f] min used in total.' % (time.time()-start_0)/60)









#rnd = np.random
#rnd.seed(0)
#
#n = 10 # number of clients
#Q = 15 # vehicle capacity
#N = [i for i in range(1, n+1)] # set of clients
#V = [0] + N # set of nodes
#q = {i: rnd.randint(1, 10) for i in N} # the amount that has to be delivered to customer i \in N
#
#loc_x = rnd.rand(len(V))*200
#loc_y = rnd.rand(len(V))*100
#
#plt.scatter(loc_x[1:], loc_y[1:], c='b')
#for i in N:
#    plt.annotate('$q_{%d}=%d$'%(i, q[i]), (loc_x[i]+2, loc_y[i]))
#
#plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
#plt.axis('equal')
#
#A = [(i, j) for i in V for j in V if i!=j] # set of arcs
#c = {(i, j): ((loc_x[i]-loc_x[j])**2 + (loc_y[i]-loc_y[j])**2)**0.5 for i, j in A} # cost of travel over arc (i, j) \in A
#
#
#mdl = Model('CVRP')
#x = mdl.binary_var_dict(A, name='x')
#u = mdl.continuous_var_dict(N, ub=Q, name='u')
#
#mdl.minimize(mdl.sum(c[i, j] * x[i, j] for i, j in A))
#mdl.add_constraints( mdl.sum(x[i, j] for j in V if j!=i)==1 for i in N )
#mdl.add_constraints( mdl.sum(x[i, j] for i in V if j!=i)==1 for j in N )
#mdl.add_indicator_constraints(mdl.indicator_constraint(x[i, j], u[j] == u[i]+q[j]) for i, j in A if i!=0 and j!=0)
#mdl.add_constraints(u[i]>=q[i] for i in N)
#
#mdl.parameters.timelimit = 5
#solution = mdl.solve(log_output=True)
#print(solution.solve_status)





