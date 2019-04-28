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


class Optim(object):

    beta_0c, beta_1c, beta_2c, beta_3c, beta_4c = 0.969, -0.007, -0.14, -0.241, -0.196
    beta_0t, beta_1t, beta_2t, beta_3t, beta_4t = 0, 0, 0, -0.327, -0.145
    rho = 3 * 1e-3 # ticket price per meter
    gamma = 9 * 1e-3 # operational cost per meter
    phi = 20 # vehicle capacity

    def __init__(self, model_id, data, num_available_buses=1):
        self.data = data
        self.model_id = model_id
        self.num_potential_users = len(self.data.users_df)

        self.P = [p for p in range(self.num_potential_users)]
        self.K = [k for k in range(num_available_buses+1)] # k=0 is the virtual bus.
        self.A = deepcopy(self.data.routes)
        self.V = [(i, t) for i in range(self.data.stop_dist_mat.shape[0]) for t in range(self.data.T_min, self.data.T_max+1)]

        self.all_x = [(p, k) for p in self.P for k in self.K]
        self.all_y = [(i,j,t,s,k) for k in self.K for i,j,t,s in self.A]

    def build_model(self, threads=4, timelimit=10*3600, mip=0):
        self.start_0 = time.time()
        print('Creating model and variables...')
        self.mdl = Model(name='CB-Planning_%d'%self.model_id)
        self.mdl.parameters.threads = threads
        self.mdl.parameters.timelimit = timelimit
        self.mdl.parameters.emphasis.mip = 0

        self.x = self.mdl.binary_var_dict(self.all_x, name='x')
        self.y = self.mdl.binary_var_dict(self.all_y, name='y')

        print('Creating objective...')
        self.mdl.maximize( self.mdl.sum(self.x[p,k] * self._dist(p,k) * self.rho for p in self.P for k in self.K if k!=0)
                         - self.mdl.sum(self.y[i,j,s,t,k] * self.data.stop_dist_mat[i,j] * self.gamma for k in self.K for i,j,s,t in self.A if k!=0) )
        end_0 = time.time()
        print('[%.2f] min used for building the model.' % ((end_0-self.start_0)/60))

    def add_contsr(self):

        print('Adding the first four constrations...')
        start_1 = time.time()
        self.mdl.add_constraints( self.mdl.sum(self.x[p,k] for k in self.K)==1 for p in self.P)

        self.mdl.add_constraints( self.mdl.sum(self.x[p,k] for p in self.P) <= self.phi  for k in self.K if k!=0)

        self.mdl.add_constraints( self.x[p,k] <= self.mdl.sum( self.y[i,j,t,s,k] \
                                  for i,j,t,s in self.data.users_df.iloc[p]['A_O']) for p in self.P for k in self.K if k != 0 )

        self.mdl.add_constraints( self.x[p,k] <= self.mdl.sum( self.y[i,j,t,s,k] \
                                  for i,j,t,s in self.data.users_df.iloc[p]['A_D']) for p in self.P for k in self.K if k != 0 )
        end_1 = time.time()
        print('[%.2f] min used for adding the first four constrations.' % ((end_1-start_1)/60))


        print('Adding the last four constrations...')
        start_2 = time.time()
        self.mdl.add_constraints(self.mdl.sum( self.y[0,j,self.data.T_min,s,k] for j,s in self.V if (j!=0 and s>= self.data.T_min) or (j==0 and s==self.data.T_max) ) \
                                -self.mdl.sum( self.y[j,0,s,self.data.T_min,k] for j,s in self.V if self.data.T_min == self.data.T_max)  == 1 for k in self.K)

        self.mdl.add_constraints(self.mdl.sum( self.y[0,j,self.data.T_max,s,k] for j,s in self.V if self.data.T_max==self.data.T_min) \
                                -self.mdl.sum( self.y[j,0,s,self.data.T_max,k] for j,s in self.V if (j!=0 and self.data.T_max>= s) or (j==0 and s==self.data.T_min) )  == -1 for k in self.K)


        self.mdl.add_constraints(self.mdl.sum( self.y[i,j,t,s,k] for j,s in self.V \
                                    if  (
                                            (i!=0 and j!=0 and i!=j and s-t==self.data.stop_time_mat[i,j]) \
                                            or (i==0 and j!=0 and s>=t and t==self.data.T_min) \
                                            or (j==0 and i!=0 and s>=t and s==self.data.T_max) \
                                            or (i==j and i!=0 and s>t)\
                                            or (i==j and i==0 and t==self.data.T_min and s==self.data.T_max)
                                        )
                                    )
                                -self.mdl.sum( self.y[j,i,s,t,k] for j,s in self.V \
                                    if  (
                                            (i!=0 and j!=0 and i!=j and t-s==self.data.stop_time_mat[j,i]) \
                                            or (i==0 and j!=0 and t>=s and t==self.data.T_max) \
                                            or (j==0 and i!=0 and t>=s and s==self.data.T_min) \
                                            or (i==j and i!=0 and t>s) \
                                            or (i==j and i==0 and s==self.data.T_min and t==self.data.T_max)
                                        )
                                    ) == 0 \
                           for k in self.K for i,t in self.V if (not (i==0 and t==self.data.T_min)) and (not (i==0 and t==self.data.T_max))  )

        #add=mdl.add_constraints(mdl.sum(y[i,j,t,s,k] for i,j,t,s in A if i==j) == 1 for k in K if k!=0)

        end_2 = time.time()
        print('[%.2f] min used for adding the last three constrations.' % ((end_2-start_2)/60))

    def solve(self):
        print('Solving the model...')
        self.solution = self.mdl.solve(log_output=True)
        print('Solution status:', self.solution.solve_status)
        print('[%.2f] min used in total.' % ((time.time()-self.start_0)/60))

        return self.solution

    def save_sol(self):
        with open('../solutions/solution_%d.xml'%self.model_id,'w') as f:
            f.write(self.solution.export_as_mst_string())

        with open('../solutions/solution_%d.pk'%self.model_id, 'wb') as f:
            sol = {'x': self.solution.get_value_dict(self.x), 'y': self.solution.get_value_dict(self.y)}
            pickle.dump(sol, f)

    def get_routes(self):
        res = self.solution.get_value_dict(self.y)
        route = []
        for k, v in res.items():
            if v == 1 and k[-1] == 1:
                route.append(k)

        route = np.stack([list(each) for each in route])

        tt = route[route[:,2].argsort()]
        return tt

    def get_pax_status(self):
        unserved_p_id = [p for p in self.P for k in self.K if (self.x[p, k].solution_value and k==0)]
        served_p_id = [p for p in self.P for k in self.K if (self.x[p, k].solution_value and k!=0)]
        return served_p_id, unserved_p_id

    def _mu(self, O, D, t_o, t_d, p):

        OD_coords = self.data.users_df.iloc[p]['coords']
        walk_dist = 1/2 * (pairwise_distances(OD_coords[:, :2], self.data.stop_locs[O], metric='l1') + pairwise_distances(OD_coords[:, 2:], self.data.stop_locs[D], metric='l1'))

        preferred_time_window = self.data.users_df.iloc[p]['preferred_time_window']
        time_adj = 1/2 * (abs(preferred_time_window[0] - t_o) + abs(preferred_time_window[1] - t_d))

        travel_time_c = t_d - t_o
        travel_time_t = self.data.users_df.iloc[p]['trip_time']

        fare_c = self.rho * self.data.stop_dist_mat[O, D]
        fare_t = self.data.users_df.iloc[p]['total_amount']

        exp_mu_c = np.exp(self.beta_0c + self.beta_1c*walk_dist + self.beta_2c*time_adj + self.beta_3c*travel_time_c + self.beta_4c*fare_c)
        exp_mu_t = np.exp(self.beta_0t + self.beta_1t*walk_dist + self.beta_2t*time_adj + self.beta_3t*travel_time_t + self.beta_4t*fare_t)

        score = exp_mu_c / (exp_mu_c + exp_mu_t)

        return score.item()

    def _dist(self, p, k):

        possible_Os = self.data.users_df.iloc[p]['candidate_pick_up_loc']
        possible_Ds = self.data.users_df.iloc[p]['candidate_drop_off_loc']
        O = 0
        D = 0
        t_o = self.data.T_min
        t_d = self.data.T_max
        for i, j, s, t in self.A:
#            if y[i,j,s,t,k].solution_value:
            if self.y[i,j,s,t,k]:
                if i in possible_Os:
                    O = i
                    t_o = s
                if j in possible_Ds:
                    D = j
                    t_d = t
        if O == 0 or D == 0:
            return 0
        else:
            return self.data.stop_dist_mat[O, D] #* self._mu(O, D, t_o, t_d, p)


if __name__ == '__main__':

    iters = 30
    data = pickle.load(open('../data/data.pickle', 'rb'))
    data.users_df['user_id'] = data.users_df.index

    for iter in range(iters):
        optim = Optim(model_id=iter, data=data, num_available_buses=1)

        optim.build_model(threads=8, timelimit=40*60, mip=0)
        optim.add_contsr()
        optim.solve()
        optim.save_sol()

        served_pax_iloc, unserved_pax_iloc = optim.get_pax_status()

        if served_pax_iloc == []:
            print('Solution has no changes! Quit optimization!')
            print('-------------------------------------------------------------------------------')
        else:
            print('Next round optimization starts...')
            print('-------------------------------------------------------------------------------')

        data.users_df = data.users_df.iloc[unserved_pax_iloc]
