import gurobipy as gp 
import numpy as np 
import networkx as nx


import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
import sys
import datetime
import math
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import logging 
import time
from collections import defaultdict
from sklearn.metrics import mean_squared_error as mse
from scipy.special import expit, logit
import copy

true=True
rand=np.random.binomial


''' Helper functions to generate opt problem 
'''
def zeros(d1,d2):
    return np.zeros([d1,d2])
def vcat(a1,a2):
    return np.vstack([a1,a2])
def ones(*args): 
    return np.ones(list(args))

def length(x): 
    return len(x)
def convert_grid_to_list(dim1, dim2):
    g = nx.grid_2d_graph(dim1, dim2)
    sources = []; destinations = []
    nodelist=[node for node in g.nodes]
    n_edges = len([e for e in g.edges])
    scalar_nodes = dict( zip(nodelist,range(len(nodelist))))
    tuple_nodes = dict( zip(range(len(nodelist)), nodelist) )
    print(scalar_nodes)
    for e in g.edges:
        sources += [scalar_nodes[e[0]]]
        destinations += [scalar_nodes[e[1]]]
    # unit test
#     print([(tuple_nodes[sources[i]], tuple_nodes[destinations[i]]) for i in range(len(sources))])
#     print(g.edges)
    return sources, destinations, scalar_nodes, tuple_nodes


###########
# Generate data 

"""
generate_poly_kernel_data(B_true, n, degree; inner_constant=1, outer_constant = 1, kernel_damp_normalize=true,
    kernel_damp_factor=1, noise=true, noise_half_width=0, normalize_c=true)

Generate (X, c) from the polynomial kernel model X_{ji} ~ N(0, 1) and
c_i(j) = ( (alpha_j * B_true[j,:] * X[:,i]  + inner_constant)^degree + outer_constant ) * epsilon_{ij} where
alpha_j is a damping term and epsilon_{ij} is a noise term.

# Arguments
- `kernel_damp_normalize`: if true, then set
alpha_j = kernel_damp_factor/norm(B_true[j,:]). This results in
(alpha_j * B_true[j,:] * X[:,i]  + inner_constant) being normally distributed with
mean inner_constant and standard deviation kernel_damp_factor.
- `noise`:  if true, generate epsilon_{ij} ~  Uniform[1 - noise_half_width, 1 + noise_half_width]
- `normalize_c`:  if true, normalize c at the end of everything
"""
def generate_poly_kernel_data_simple(B_true, n, degree, inner_constant=1, outer_constant = 1, kernel_damp_normalize=true,
kernel_damp_factor=1, noise=true, noise_half_width=0, normalize_c=true, normalize_small_threshold = 0.0001):

    (d, p) = B_true.shape
    X_observed = np.random.randn(p, n)
    dot_prods = B_true@X_observed
    # first generate c_observed without noise
    c_observed = zeros(d, n)
    for j in range(d):
        cur_kernel_damp_factor = kernel_damp_factor
        for i in range(n): 
            c_observed[j, i] = (cur_kernel_damp_factor*dot_prods[j, i] + inner_constant)**degree + outer_constant
            if noise:
                epsilon = (1 - noise_half_width) + 2*noise_half_width*np.random.random()
                c_observed[j, i] = c_observed[j, i]*epsilon
    return X_observed, c_observed

def get_weighted_predictors(regressor, c_train, X_train,weights=None, random_regr=False):
    [d,n_train] = c_train.shape
    predictors = {}
    for d_ in range(d): 
    	# If random: fix random state for refitting
        if random_regr: 
        	regr = regressor(random_state=1)
        else: 
        	regr = regressor()
        if weights is not None: 
        	regr.fit(X_train.T, c_train[d_,:], sample_weight=weights[d_,:])
        else: 
        	regr.fit(X_train.T, c_train[d_,:])
        predictors[d_] = regr
    return predictors

def get_regret(predictors,X_train, c_train, X_test, c_test, sp_oracle, quiet=False):
    [d,n_train] = c_train.shape
    c_pred = np.asarray([ predictors[d_].predict(X_train.T) for d_ in range(d)])
    regrets = np.zeros(n_train);x_star_regr = np.zeros(c_pred.shape)
    for i in range(n_train): 
        if not quiet: 
            if i%500==0: print(i)
        [r_star,x_star] = sp_oracle(c_train[:,i])
        [r_pred,x_pred] = sp_oracle(c_pred[:,i])
        regrets[i] = r_star-r_pred; x_star_regr[:,i]=x_star-x_pred
    return [regrets, x_star_regr]

def eval_regr(c_hat,c_star, oracle): 
    # oracle doesn't serialize for parallelization: need to chanage 
    [r_star,x_star] = oracle(c_star)
    [r_pred,x_pred] = oracle(c_hat)
    return r_star-r_pred,x_star-x_pred

def generate_data(n_train, n_test, n_holdout, polykernel_degree,polykernel_noise_half_width,B_true,gen_test=True):
    ''' return X, which is p x N 
    '''
    (X_train, c_train) = generate_poly_kernel_data_simple(B_true, n_train, polykernel_degree, polykernel_noise_half_width)
    (X_validation, c_validation) = generate_poly_kernel_data_simple(B_true, n_holdout, polykernel_degree, polykernel_noise_half_width)
    if gen_test: 
        (X_test, c_test) = generate_poly_kernel_data_simple(B_true, n_test, polykernel_degree, polykernel_noise_half_width)
        X_test = vcat(ones(1,n_test), X_test)
    # Add intercept in the first row of X
    # X is p x N 
    X_train = vcat(ones(1,n_train), X_train); X_validation = vcat(ones(1,n_holdout), X_validation); 
    if gen_test: 
    	return [X_train, c_train,X_validation, c_validation,X_test, c_test]
    else:
    	return [X_train, c_train,X_validation, c_validation]



# Adapted from https://github.com/JayMan91/NeurIPSIntopt/blob/034682a818220a7a0a72d478c045fbe532428710/shortespath/shortespath.py
class SPO:
	def __init__(self,A,num_features, num_layers, intermediate_size,num_instance= 1,
		activation_fn = nn.ReLU, epochs=10,optimizer=optim.Adam,
		validation=False,**hyperparams):
		self.A = A
		self.num_features = num_features
		self.num_layers = num_layers
		self.activation_fn = activation_fn
		self.intermediate_size = intermediate_size
		
		self.epochs = epochs
		self.num_instance = num_instance
		self.validation = validation
		
	
		self.net = make_fc(num_layers=num_layers, num_features=num_features, 
			activation_fn= activation_fn,
			intermediate_size= intermediate_size)
		self.optimizer = optimizer(self.net.parameters(), **hyperparams)
		
	def fit(self,X,y,instances):
		#A_torch = torch.from_numpy(self.A).float()	
		test_instances =  instances['test']
		validation_instances =  instances['validation']
		train_instances = instances['train']	
		time_  = 0
		self.model_time = 0		
		n_train = X.shape[0]

		if self.validation:
			validation_list = []

		X_torch = torch.from_numpy(X).float()
		y_torch = torch.from_numpy(y).float()

		true_solution ={}
		logging.info("training started")
		for e in range(self.epochs):
			for i in range(self.num_instance):
				start_time = time.time()
				self.optimizer.zero_grad()
				source, dest = train_instances[i]
				b = np.zeros(len(self.A))
				b[source] =1
				b[dest ]=-1
				if i not in true_solution:
					model = gp.Model()
					model.setParam('OutputFlag', 0)
					x = model.addMVar(shape= self.A.shape[1], lb=0.0, vtype=gp.GRB.CONTINUOUS, name="x")
					model.addConstr(self.A @ x == b, name="eq")
					model.setObjective((y_torch.detach().numpy())@x, gp.GRB.MINIMIZE)
					model.optimize()
					x_true = x.X

					true_solution[i] = np.copy(x_true)
				x_true = true_solution[i]

				c_pred = self.net(X_torch).squeeze()
				c_spo = (2*c_pred - y_torch)
				
				model = gp.Model()
				model.setParam('OutputFlag', 0)
				x = model.addMVar(shape= self.A.shape[1], lb=0.0, ub=1.0,vtype=gp.GRB.CONTINUOUS, name="x")
				model.addConstr(self.A @ x == b, name="eq")
				model.setObjective((c_spo.detach().numpy())@x, gp.GRB.MINIMIZE)
				model.optimize()
				#print(model.status)
				x_spo = x.X
				grad = torch.from_numpy( x_true - x_spo).float()
				loss = self.net(X_torch).squeeze()
				loss.backward(gradient=grad)
				self.optimizer.step()
				logging.info("bkwd done")

				end_time = time.time()
				time_ += end_time - start_time
				if self.validation:
					if ((i+1)%20==0):
						validation_list.append( validation_module(self.net,self.A, 
					X,y,train_instances,validation_instances, 
					test_instances,time_,e,i))

			print("Epoch {} Loss:{} Time: {:%Y-%m-%d %H:%M:%S}".format(e+1,loss.sum().item(),
				datetime.datetime.now()))
		if self.validation :
	
			dd = defaultdict(list)
			for d in validation_list:
				for key, value in d.items():
					dd[key].append(value)
			df = pd.DataFrame.from_dict(dd)
			# print(validation_module(self.net,self.A, 
			# 			X,y,train_instances,validation_instances, 
			# 			test_instances,time_,e,i))
			# pred = self.predict(X)
			# print(mse(pred,y))
			logging.info('Completion Time %s \n' %str(datetime.datetime.now()) )
			return df
	def validation_result(self,X,y, instances):
		validation_rslt = get_loss(self.net, self.A, X,y,instances)
		return  validation_rslt[0], validation_rslt[1]
    

	def predict(self,X):
		X_torch = torch.from_numpy(X).float()
		self.net.eval()
		pred= self.net(X_torch)
		self.net.train()
		return pred.detach().detach().numpy().squeeze()