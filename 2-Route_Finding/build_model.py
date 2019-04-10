# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:02:27 2019

@author: msq96
"""


import time
import pickle
from copy import deepcopy
import numpy as np

from sklearn.metrics import pairwise_distances
from docplex.mp.model import Model


with open('../data/data.pickle', 'rb') as f:
    data = pickle.load(f)

num_potential_users = len(data.users_df)
num_available_buses = 1
rho = 3 * 1e-3 # ticket price per meter
gamma = 9 * 1e-3 # operational cost per meter
phi = 20 # vehicle capacity

beta_0c, beta_1c, beta_2c, beta_3c, beta_4c = 0.969, -0.007, -0.14, -0.241, -0.196
beta_0t, beta_1t, beta_2t, beta_3t, beta_4t = 0, 0, 0, -0.327, -0.145


P = [p for p in range(num_potential_users)]
K = [k for k in range(num_available_buses+1)] # k=0 is the virtual bus.
A = deepcopy(data.routes)
V = [(i,t) for i in range(data.stop_dist_mat.shape[0]) for t in range(data.T_min, data.T_max+1)]



all_x = [(p, k) for p in P for k in K]
all_y = [(i,j,t,s,k) for k in K for i,j,t,s in A]


mdl = Model(name='CB-Planning')
mdl.parameters.threads = 2
mdl.parameters.timelimit = 2500
mdl.parameters.emphasis.mip = 0

x = mdl.binary_var_dict(all_x, name='x')
y = mdl.binary_var_dict(all_y, name='y')

def mu(O, D, t_o, t_d, p):

    OD_coords = data.users_df.iloc[p]['coords']
    walk_dist = 1/2 * (pairwise_distances(OD_coords[:, :2], data.stop_locs[O], metric='l1') + pairwise_distances(OD_coords[:, 2:], data.stop_locs[D], metric='l1'))

    preferred_time_window = data.users_df.iloc[p]['preferred_time_window']
    time_adj = 1/2 * (abs(preferred_time_window[0] - t_o) + abs(preferred_time_window[1] - t_d))

    travel_time_c = t_d - t_o
    travel_time_t = data.users_df.iloc[p]['trip_time']

    fare_c = rho * data.stop_dist_mat[O, D]
    fare_t = data.users_df.iloc[p]['total_amount']

    exp_mu_c = np.exp(beta_0c + beta_1c*walk_dist + beta_2c*time_adj + beta_3c*travel_time_c + beta_4c*fare_c)
    exp_mu_t = np.exp(beta_0t + beta_1t*walk_dist + beta_2t*time_adj + beta_3t*travel_time_t + beta_4t*fare_t)

    score = exp_mu_c / (exp_mu_c + exp_mu_t)
    print(score)

    return score.item()

def dist(p, k):

    possible_Os = data.users_df.iloc[p]['candidate_pick_up_loc']
    possible_Ds = data.users_df.iloc[p]['candidate_drop_off_loc']
    O = 0
    D = 0
    t_o = data.T_min
    t_d = data.T_max
    for i, j, s, t in A:
#        if y[i,j,s,t,k].solution_value:
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
        return data.stop_dist_mat[O, D] #* mu(O, D, t_o, t_d, p)


print('Creating the model...')
start_0 = time.time()
mdl.maximize( mdl.sum(x[p,k] * dist(p,k) * rho for p in P for k in K if k!=0)
                - mdl.sum(y[i,j,s,t,k] * data.stop_dist_mat[i,j] * gamma for k in K for i,j,s,t in A if k!=0) )
end_0 = time.time()
print('[%.2f] min used for creating the model.' % ((end_0-start_0)/60))


print('Adding the first four constrations...')
start_1 = time.time()
mdl.add_constraints( mdl.sum(x[p,k] for k in K)==1 for p in P)

mdl.add_constraints( mdl.sum(x[p,k] for p in P) <= phi  for k in K if k!=0)

mdl.add_constraints( x[p,k] <= mdl.sum( y[i,j,t,s,k] \
                    for i,j,t,s in data.users_df.iloc[p]['A_O']) for p in P for k in K if k != 0 )

mdl.add_constraints( x[p,k] <= mdl.sum( y[i,j,t,s,k] \
                    for i,j,t,s in data.users_df.iloc[p]['A_D']) for p in P for k in K if k != 0 )
end_1 = time.time()
print('[%.2f] min used for adding the first four constrations.' % ((end_1-start_1)/60))


print('Adding the last four constrations...')
start_2 = time.time()
mdl.add_constraints(mdl.sum( y[0,j,data.T_min,s,k] for j,s in V if (j!=0 and s>= data.T_min) or (j==0 and s==data.T_max) ) \
                   -mdl.sum( y[j,0,s,data.T_min,k] for j,s in V if data.T_min == data.T_max)  == 1 for k in K)

mdl.add_constraints(mdl.sum( y[0,j,data.T_max,s,k] for j,s in V if data.T_max==data.T_min) \
                   -mdl.sum( y[j,0,s,data.T_max,k] for j,s in V if (j!=0 and data.T_max>= s) or (j==0 and s==data.T_min) )  == -1 for k in K)


mdl.add_constraints(mdl.sum( y[i,j,t,s,k] for j,s in V \
                            if  (
                                    (i!=0 and j!=0 and i!=j and s-t==data.stop_time_mat[i,j]) \
                                    or (i==0 and j!=0 and s>=t and t==data.T_min) \
                                    or (j==0 and i!=0 and s>=t and s==data.T_max) \
                                    or (i==j and i!=0 and s>t)\
                                    or (i==j and i==0 and t==data.T_min and s==data.T_max)
                                )
                            )
                   -mdl.sum( y[j,i,s,t,k] for j,s in V \
                            if  (
                                    (i!=0 and j!=0 and i!=j and t-s==data.stop_time_mat[j,i]) \
                                    or (i==0 and j!=0 and t>=s and t==data.T_max) \
                                    or (j==0 and i!=0 and t>=s and s==data.T_min) \
                                    or (i==j and i!=0 and t>s) \
                                    or (i==j and i==0 and s==data.T_min and t==data.T_max)
                                )
                            ) == 0 \
                   for k in K for i,t in V if (not (i==0 and t==data.T_min)) and (not (i==0 and t==data.T_max))  )

#add=mdl.add_constraints(mdl.sum(y[i,j,t,s,k] for i,j,t,s in A if i==j) == 1 for k in K if k!=0)


end_2 = time.time()
print('[%.2f] min used for adding the last three constrations.' % ((end_2-start_2)/60))

print('Solving...')
solution = mdl.solve(log_output=True)
print(solution.solve_status)
print('[%.2f] min used in total.' % ((time.time()-start_0)/60))


with open('solution_3.xml','w') as f:
    f.write(solution.export_as_mst_string())



res = solution.get_value_dict(y)
route = []
for k, v in res.items():
    if v == 1 and k[-1] == 1:
        route.append(k)

route = np.stack([list(each) for each in route])

tt=route[route[:,2].argsort()]


p_id = [p for p in P for k in K if (x[p, k].solution_value and k==0)]
data.users_df = data.users_df.iloc[p_id]


