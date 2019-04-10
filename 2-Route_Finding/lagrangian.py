# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:02:27 2019

@author: msq96
"""


import time
import pickle
from copy import deepcopy
import numpy as np


from docplex.mp.model import Model


with open('./data.pickle', 'rb') as f:
    data = pickle.load(f)

num_potential_users = len(data.users_df)
num_available_buses = 3
rho = 3 * 1e-3 # ticket price per meter
gamma = 9 * 1e-3 # operational cost per meter
phi = 30 # vehicle capacity


print('Building data...')
P = [p for p in range(num_potential_users)]
K = [k for k in range(num_available_buses+1)] # k=0 is the virtual bus.
A = deepcopy(data.routes)
V = [(i,t) for i in range(data.stop_dist_mat.shape[0]) for t in range(data.T_min, data.T_max+1)]


all_x = [(p, k) for p in P for k in K]
all_y = [(i,j,t,s,k) for k in K for i,j,t,s in A]
all_pi_pen = [(p, k) for p in P for k in K if k!=0]
all_la_pen = [(p, k) for p in P for k in K if k!=0]


mdl = Model(name='CB-Planning')
mdl.parameters.emphasis.mip = 1


x = mdl.binary_var_dict(all_x, name='x')
y = mdl.binary_var_dict(all_y, name='y')
pi_pen = mdl.continuous_var_dict(all_pi_pen, name='pi_pen')
la_pen = mdl.continuous_var_dict(all_la_pen, name='la_pen')


def dist(p, k):

    possible_Os = data.users_df.iloc[p]['candidate_pick_up_loc']
    possible_Ds = data.users_df.iloc[p]['candidate_drop_off_loc']
    O = 0
    D = 0
    for i, j, s, t in A:
        if y[i,j,s,t,k]:
            if i in possible_Os:
                O = i
            if j in possible_Ds:
                D = j
    if O == 0 or D == 0:
        return 0
    else:
        return data.stop_dist_mat[O, D] #* mu(O, D, t_o, t_d, p)

print('Adding constraints...')
mdl.add_constraints( mdl.sum(x[p,k] for k in K)==1 for p in P) #2

mdl.add_constraints( mdl.sum(x[p,k] for p in P) <= phi  for k in K if k!=0) #3

mdl.add_constraints(mdl.sum( y[0,j,data.T_min,s,k] for j,s in V if (j!=0 and s>= data.T_min) or (j==0 and s==data.T_max) ) \
                   -mdl.sum( y[j,0,s,data.T_min,k] for j,s in V if data.T_min == data.T_max)  == 1 for k in K) #7

mdl.add_constraints(mdl.sum( y[0,j,data.T_max,s,k] for j,s in V if data.T_max==data.T_min) \
                   -mdl.sum( y[j,0,s,data.T_max,k] for j,s in V if (j!=0 and data.T_max>= s) or (j==0 and s==data.T_min) )  == -1 for k in K) #7


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
                   for k in K for i,t in V if (not (i==0 and t==data.T_min)) and (not (i==0 and t==data.T_max))  ) #7



mdl.add_constraints( mdl.sum( y[i,j,t,s,k] for i,j,t,s in data.users_df.iloc[p]['A_O']) - x[p,k] == pi_pen[p,k] for p in P for k in K if k != 0 ) #5
mdl.add_constraints( mdl.sum( y[i,j,t,s,k] for i,j,t,s in data.users_df.iloc[p]['A_D']) - x[p,k] == la_pen[p,k] for p in P for k in K if k != 0 ) #6

max_iters = 20
eps = 1e-6
loop_count = 0
best = 0
initial_multiplier = 1

pi_mul = {(p,k):initial_multiplier for p in P for k in K if k!=0}
la_mul = {(p,k):initial_multiplier for p in P for k in K if k!=0}


print('Adding kpi...')
total_profit = mdl.sum(x[p,k] * dist(p,k) * rho for p in P for k in K if k!=0) \
                - mdl.sum(y[i,j,s,t,k] * data.stop_dist_mat[i,j] * gamma for k in K for i,j,s,t in A if k!=0)
mdl.add_kpi(total_profit, "Total profit")


print("starting the loop")
while loop_count <= max_iters:
    loop_count += 1
    # Rebuilt at each loop iteration
    total_penalty = mdl.sum(pi_mul[p,k]*pi_pen[p,k] + la_mul[p,k]*la_pen[p,k] for p in P for k in K if k!=0)
    mdl.parameters.timelimit = 600

    mdl.maximize(total_profit + total_penalty)
    s = mdl.solve(log_output=True)
    if not s:
        print("*** solve fails, stopping at iteration: %d" % loop_count)
        break
    best = s.objective_value
    pi_penalties = {(p,k): pi_pen[p,k].solution_value for p in P for k in K if k!=0}
    la_penalties = {(p,k): la_pen[p,k].solution_value for p in P for k in K if k!=0}
    print('%d> new lagrangian iteration:\n\t obj=%g.' % (loop_count, best))

    do_stop = True
    justifier = 0
    temp = []
    for p in P:
        for k in K:
            if k!= 0:
                penalized_violation = pi_penalties[p,k] * pi_mul[p,k] + la_penalties[p,k] * la_mul[p,k]
                temp.append(penalized_violation)
                if penalized_violation >= eps:
                    do_stop = False
                    justifier = penalized_violation
                    break

    if do_stop:
        print("* Lagrangian relaxation succeeds, best={:g}, penalty={:g}, #iterations={}"
                .format(best, total_penalty.solution_value, loop_count))
        break
    else:
        # Update multipliers and start the loop again.
        scale_factor = 1.0 / float(loop_count)

        pi_mul = {(p,k):max(pi_mul[p,k] - scale_factor * pi_penalties[p,k], 0.) for p in P for k in K if k!=0}
        la_mul = {(p,k):max(la_mul[p,k] - scale_factor * la_penalties[p,k], 0.) for p in P for k in K if k!=0}

        print('{0}> -- loop continues, justifier={1:g}'.format(loop_count, justifier))





print(best)