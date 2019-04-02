# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:02:27 2019

@author: msq96
"""


import pickle
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from docplex.mp.model import Model

from b_utils import nested_list_to_tuple


with open('../data/data.pickle', 'rb') as f:
    data = pickle.load(f)

num_potential_users = len(data.users_df)
num_available_buses = 10
rho = 3 * 1e-3 # ticket price per meter
gamma = 9 * 1e-3 # operational cost per meter
phi = 20 # vehicle capacity

beta_0c, beta_1c, beta_2c, beta_3c, beta_4c = 0.969, -0.007, -0.14, -0.241, -0.196
beta_0t, beta_1t, beta_2t, beta_3t, beta_4t = 0, 0, 0, -0.327, -0.145


N = [n for n in range(num_potential_users)]
M = [m for m in range(num_available_buses)]
A = deepcopy(data.uniq_feasible_routes)

all_x = [route + [m] for m in M for route in A]
all_a = [each_feasible_route + [m] + [n] for n, each_users_feasible_routes in enumerate(data.users_df['feasible_routes'])\
         for each_feasible_route in each_users_feasible_routes for m in M]


mdl = Model(name='CB-Planning')
x = mdl.binary_var_dict(nested_list_to_tuple(all_x), name='x')
a = mdl.binary_var_dict(nested_list_to_tuple(all_a), name='a')


def Len():
    pass
def Adj():
    pass
def p(n, m):
    walk_dist = 1/2 * mdl.sum( a.get((i,j,s,t,m,n), 0) * (Len(n, 'o', i) + Len(n, 'd', j)) for i,j,s,t in A )

    time_adj =  mdl.sum( a.get((i,j,s,t,m,n), 0) * (Adj(n, 'o', s) + Adj(n, 'd', t)) for i,j,s,t in A )

    travel_time = mdl.sum( a.get((i,j,s,t,m,n), 0) * (t - s) for i,j,s,t in A )

    fare = rho * mdl.sum( a.get((i,j,s,t,m,n), 0) * data.stop_dist_mat[i,j] for i,j,s,t in A )

    exp_mu_c = np.exp(beta_0c + beta_1c*walk_dist + beta_2c*time_adj + beta_3c*travel_time + beta_4c*fare)
    exp_mu_t = np.exp(beta_0t + beta_1t*walk_dist + beta_2t*time_adj + beta_3t*travel_time + beta_4t*fare)

    return exp_mu_c / (exp_mu_c + exp_mu_t)


mdl.maximize( mdl.sum( p(n,m) * rho * mdl.sum( a.get((i,j,s,t,m,n), 0)*data.stop_dist_mat[i,j] for i,j,s,t in A ) for m in M for n in N )
              - mdl.sum( gamma * mdl.sum( x.get((i,j,s,t,m), 0)*data.stop_dist_mat[i,j] for i,j,s,t in A ) for m in M ) )

# TODO flow balance constraint #8 #12

mdl.add_constraints( mdl.sum( x.get((i,j,s,t,m), 0) for _,j,_,t in A ) <= 1 for i,_,s,_,m in all_x ) # 9

mdl.add_constraints( a.get((i,j,s,t,m,n), 0) <= x.get((i,j,s,t,m), 0) for i,j,s,t,m,n in all_a ) # 10

mdl.add_constraints( mdl.sum( a.get((i,j,s,t,m,n), 0) for m in M ) <=1 for n in N for i,j,s,t in A  ) # 11

# do not have constraint #13 14 15

mdl.add_constraints( mdl.sum( a.get((i,j,s,t,m,n), 0) for n in N ) <= phi*x.get((i,j,s,t,m), 0)  for i,j,s,t,m in all_x )


mdl.parameters.timelimit = 100
solution = mdl.solve(log_output=True)
print(solution.solve_status)







rnd = np.random
rnd.seed(0)

n = 10 # number of clients
Q = 15 # vehicle capacity
N = [i for i in range(1, n+1)] # set of clients
V = [0] + N # set of nodes
q = {i: rnd.randint(1, 10) for i in N} # the amount that has to be delivered to customer i \in N

loc_x = rnd.rand(len(V))*200
loc_y = rnd.rand(len(V))*100

plt.scatter(loc_x[1:], loc_y[1:], c='b')
for i in N:
    plt.annotate('$q_{%d}=%d$'%(i, q[i]), (loc_x[i]+2, loc_y[i]))

plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
plt.axis('equal')

A = [(i, j) for i in V for j in V if i!=j] # set of arcs
c = {(i, j): ((loc_x[i]-loc_x[j])**2 + (loc_y[i]-loc_y[j])**2)**0.5 for i, j in A} # cost of travel over arc (i, j) \in A


mdl = Model('CVRP')
x = mdl.binary_var_dict(A, name='x')
u = mdl.continuous_var_dict(N, ub=Q, name='u')

mdl.minimize(mdl.sum(c[i, j] * x[i, j] for i, j in A))
mdl.add_constraints( mdl.sum(x[i, j] for j in V if j!=i)==1 for i in N )
mdl.add_constraints( mdl.sum(x[i, j] for i in V if j!=i)==1 for j in N )
mdl.add_indicator_constraints(mdl.indicator_constraint(x[i, j], u[j] == u[i]+q[j]) for i, j in A if i!=0 and j!=0)
mdl.add_constraints(u[i]>=q[i] for i in N)

mdl.parameters.timelimit = 5
solution = mdl.solve(log_output=True)
print(solution.solve_status)





