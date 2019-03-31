# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:02:27 2019

@author: msq96
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt

from docplex.mp.model import Model


with open('../data/data.pickle', 'rb') as f:
    data = pickle.load(f)


def build_Y(data, K):
    Y = []
    for i in data.stop_locs.keys():
        for j in data.stop_locs.keys():
            if i != j:
                for t in data.stop_pickup_time_unit[i]:
                    for s in data.stop_dropoff_time_unit[j]:
                        time_diff = data.stop_dist_mat[i][j]
                        if t - s == time_diff:
                            for k in range(K):
                                Y.append((i, j, t, s, k))
    return Y


P = len(data.users_df)
K = 10
X = [(p, k) for p in range(P) for k in range(K)]
Y = build_Y(data, K)

model = Model(name='CB-Planning')
x = model.binary_var_dict(X, name='x')
y = model.binary_var_dict(Y, name='y')








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





