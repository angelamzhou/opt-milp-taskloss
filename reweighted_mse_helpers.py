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
    return sources, destinations, scalar_nodes, tuple_nodes

'''DG Helpers'''

def generate_poly_kernel_data_simple(B_true, n, degree, inner_constant=1, outer_constant = 1, kernel_damp_normalize=true,
kernel_damp_factor=1, noise=true, noise_half_width=0, normalize_c=true, normalize_small_threshold = 0.0001):
    '''
    Generates feature x true cost vector combinations using a polynomial kernel
    '''
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

def generateInstanceDict(X, c, oracle):
    '''
    Takes X, C values and pre-computes optimal values/optimal solutions for easier training/regret
    '''
    instances = []
    
    for i in range(c.shape[1]):
        opt_val, opt_sol = oracle.solve(c[:,i])
        instances.append({
            'objective':'min',
            'c': c[:,i],
            'features': X[:,i],
            'opt_val': opt_val,
            'opt_sol': opt_sol
        })
    
    return instances

def generate_data(n_train, n_test, n_holdout, polykernel_degree,polykernel_noise_half_width,B_true,gen_test=True):
    ''' return X, which is p x N 
    '''
    (X_train, c_train) = generate_poly_kernel_data_simple(B_true, n_train, polykernel_degree, 
                                                          polykernel_noise_half_width)
    (X_validation, c_validation) = generate_poly_kernel_data_simple(B_true, n_holdout, polykernel_degree, 
                                                                    polykernel_noise_half_width)
    
    if gen_test: 
        (X_test, c_test) = generate_poly_kernel_data_simple(B_true, n_test, polykernel_degree, 
                                                            polykernel_noise_half_width)
        X_test = vcat(ones(1,n_test), X_test)
    
    # Add intercept in the first row of X
    # X is p x N 
    X_train = vcat(ones(1,n_train), X_train)
    X_validation = vcat(ones(1,n_holdout), X_validation);
    
    if gen_test: 
        return [X_train, c_train,X_validation, c_validation,X_test, c_test]
    else:
        return [X_train, c_train,X_validation, c_validation]

    
''' ML Training Helpers'''

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

def get_regret(predictors,X_train, c_train, trainDict, 
               X_test, c_test, testDict, 
               oracle, quiet=False):
    
    train_reg,train_x_reg = computeDataSetRegret(predictors, X_train, c_train, trainDict, oracle )
    test_reg, test_x_reg = computeDataSetRegret(predictors, X_test, c_test, testDict, oracle )

    return [train_reg,train_x_reg, test_reg, test_x_reg]

def computeDataSetRegret(predictors, X, c, dataDict, oracle):
    [d,n] = c.shape
    c_preds = np.asarray([ predictors[d_].predict(X.T) for d_ in range(d)])
    
    regrets = np.zeros(n);
    x_star_regr = np.zeros(c_preds.shape)

    for i, instance in enumerate(dataDict): 
        [r_pred,x_pred] = oracle.solve(c_preds[:,i])
        
        regrets[i] = instance['opt_val']-instance['c'] @ x_pred; 
        x_star_regr[:,i]=instance['opt_sol'] -x_pred
        
    return regrets, x_star_regr

def feasible_least_squares(regressor, x_star_regr, mu, c_train, X_train, trainDict,
                           c_test, X_test, testDict, oracle, random_regr=False):
    '''
    One-step plug in 
    regressor: type of Scikit-Learn regressor
    x_star_regr: x^* - \hat x^* 
    c_train: training data, cost vector realizations
    X_train: training data 
    '''
    # Mix decision weights with uniform 
    weights = mu*np.abs(x_star_regr) + (1-mu)*np.ones(c_train.shape)
    weighted_predictors = get_weighted_predictors(regressor, c_train, X_train, weights, random_regr=random_regr)
    [train_reg,train_x_reg, test_reg, test_x_reg] = get_regret(weighted_predictors,
            X_train, c_train, trainDict, X_test, c_test, testDict, oracle, quiet=False)
    return [weighted_predictors, train_reg,train_x_reg, test_reg, test_x_reg]
    
def runSPO(oracle, trainDict, testDict):
    #warm start with LS
    learning_params = {
        'gradient': 'LS',
        'step_size_fn': 'long_dynamic'
    }
    oracle = ShortestPathOracle(graph_params)

    learner = SGDLearner(learning_params)
    LSmodel = LinearModel(6, 40)
    learner.learn(oracle, LSmodel, 
                  trainDict,testDict,
                  batch_size = len(trainDict), epochs=10000)
    
    #Run SPO plus 
    learning_params = {
        'gradient': 'spo_plus',
        'step_size_fn': 'long_dynamic'
    }
    learner = SGDLearner(learning_params)
    tr_reg, test_reg = learner.learn(oracle, LSmodel, 
                  trainDict,testDict,
                  batch_size = 20, epochs=2000)
    
    return tr_reg, test_reg    

def run_replication_over_weights(data_params, X_test, c_test, testDict, 
                                 mixture_weights, regressor, graph_params, 
                                 num_reweights = 1, random_regr=False): 
    '''
    Run a replication under fixed data parameters 
    Assume pre-generated test dataset (to reduce noise in test evaluation)
    '''
    
    #Set-up Oracle
    oracle = ShortestPathOracle(graph_params)
    
    #Extract experiment set-up
    [n_train, n_test, n_holdout, polykernel_degree,polykernel_noise_half_width,B_true] = data_params
    
    #Results helpers
    results = []
    
    #Generate new training/validation data
    [X_train, c_train,X_validation, c_validation] = generate_data(n_train, 
        n_test, n_holdout, polykernel_degree,polykernel_noise_half_width,B_true,gen_test=False)
    
    trainDict = generateInstanceDict(X_train, c_train, oracle)

    # Learn Initial (LS) predictor 
    start_time = time.time()
    predictors = get_weighted_predictors(regressor, c_train, X_train, random_regr=random_regr )
    [regrets, x_star_regr,regrets_tst, x_star_regr_tst] = get_regret(predictors, X_train, c_train, trainDict,
                                                                      X_test, c_test, testDict, oracle)
    ls_time = time.time() - start_time
    res = {'n_train':n_train,
        'polykernel_degree': polykernel_degree,
        'n_test': n_test,
           'time': ls_time,
        'algo': 'LS',
        'tr_regret': np.mean(np.sqrt(np.square(regrets))),
        'tst_regret': np.mean(np.sqrt(np.square(regrets_tst)))
    }
    results.append(res)
    
    # Run re-weight for all the mixture weights
    for k in range(len(mixture_weights)):
        
        start_time = time.time()
        for r in range(num_reweights):
            [weighted_predictors, train_reg,train_x_reg, test_reg, test_x_reg] = feasible_least_squares(
                regressor, x_star_regr, mixture_weights[k], c_train, X_train,trainDict,
                c_test, X_test,  testDict, oracle,random_regr=random_regr)

            res = {'n_train':n_train,
                        'polykernel_degree': polykernel_degree,
                        'n_test': n_test,
                   'time': time.time() - start_time + ls_time,
                'algo': 'reweight_LS',
                'reweight': r + 1,
                'mixture_weight': mixture_weights[k],
                'tr_regret': np.mean(np.sqrt(np.square(train_reg))),
                'tst_regret': np.mean(np.sqrt(np.square(test_reg)))
            }
            results.append(res)
            
            #For multiple reweights use the new weights
            x_star_regr = train_x_reg
    
    # run SPO
    start_time = time.time()
    tr_reg, tst_reg = runSPO(oracle, trainDict, testDict)
    
    res = {'n_train':n_train,
           'polykernel_degree': polykernel_degree,
           'n_test': n_test,
           'time': time.time() - start_time,
           'algo': 'SPO',
           'tr_regret': np.mean(np.sqrt(np.square(tr_reg))),
           'tst_regret': np.mean(np.sqrt(np.square(tst_reg)))
          }
    results.append(res)

    
    return results

