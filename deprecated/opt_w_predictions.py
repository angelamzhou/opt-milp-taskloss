import gurobipy as gp 
import numpy as np 
import networkx as nx


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

def get_regret_instanceDict(predictors, instances, sp_oracle, quiet=False):
    [d,n_train] = c_train.shape
    c_pred = np.asarray([ predictors[d_].predict(X_train.T) for d_ in range(d)])
    regrets = np.zeros(n_train);x_star_regr = np.zeros(c_pred.shape)
    for inst in instances: 
        if not quiet: 
            if i%500==0: print(i)
        #[r_star,x_star] = sp_oracle(c_train[:,i])
        [r_pred,x_pred] = sp_oracle(c_pred[:,i])
        regrets[i] = instance['opt_val']- instance['c'] @ x_pred; x_star_regr[:,i]=instance['opt_sol']-x_pred
    return [regrets, x_star_regr]

def get_regret2(predictors,X_train, c_train, X_test, c_test, sp_oracle, quiet=False):
    [d,n_train] = c_train.shape
    c_pred = np.asarray([ predictors[d_].predict(X_train.T) for d_ in range(d)])
    regrets = np.zeros(n_train);x_star_regr = np.zeros(c_pred.shape)
    for i in range(n_train): 
        if not quiet: 
            if i%500==0: print(i)
        [r_star,x_star] = sp_oracle(c_train[:,i])
        [r_pred,x_pred] = sp_oracle(c_pred[:,i])
        regrets[i] = r_star-r_pred; x_star_regr[:,i]=abs(r_star-r_pred)*np.ones(x_star.shape)
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



