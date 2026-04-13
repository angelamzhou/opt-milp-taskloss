import gurobipy as gp 
import numpy as np 
import networkx as nx


import pickle
import sys
import datetime
import math
from sklearn import preprocessing
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
import logging 
import time
from collections import defaultdict
from sklearn.metrics import mean_squared_error as mse
from scipy.sparse import coo_matrix,csr_matrix
from scipy.special import expit, logit
import copy

from models import LinearModel
from sgd import SGDLearner

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
    for e in g.edges:
        sources += [scalar_nodes[e[0]]]
        destinations += [scalar_nodes[e[1]]]
    return sources, destinations, scalar_nodes, tuple_nodes


class ShortestPathOracle(object):
    def __init__(self, graph_params, quiet=True):
        nodes = graph_params['nodes']
        sources = graph_params['sources']
        destinations = graph_params['destinations']
        start_node = graph_params['start_node']
        end_node = graph_params['end_node']

        n_nodes = len(nodes)
        n_edges = len(sources)

        i_vec = np.hstack([sources, destinations])
        j_vec = np.hstack([range(n_edges), range(n_edges)])
        v_vec = np.hstack([-np.ones(n_edges), np.ones(n_edges)])

        a_mat = coo_matrix((v_vec, (i_vec, j_vec)))
        a_mat = csr_matrix(a_mat)
        bvec = np.zeros(n_nodes)
        bvec[start_node] = -1
        bvec[end_node] = 1

        self.m = gp.Model()
        if quiet:
            self.m.setParam("OutputFlag", 0)
        self.w = self.m.addMVar(n_edges, lb=0, ub=1)
        self.m.addConstrs(a_mat[i, :] @ self.w == bvec[i] for i in range(n_nodes))

    def init_model(self, params):
        pass

    def solve(self, c):
        self.m.setObjective(c @ self.w, gp.GRB.MINIMIZE)
        self.m.optimize()
        z_ast = self.m.objVal
        w_ast = np.asarray([self.w[i].X for i in range(len(c))]).flatten()
        return [z_ast, w_ast]

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

def _build_estimator(regressor, random_regr=False):
    if hasattr(regressor, 'fit') and not isinstance(regressor, type):
        estimator = clone(regressor)
    else:
        if random_regr:
            try:
                estimator = regressor(random_state=1)
            except TypeError:
                estimator = regressor()
        else:
            estimator = regressor()

    if random_regr and hasattr(estimator, 'get_params'):
        params = estimator.get_params()
        if 'random_state' in params and params['random_state'] is None:
            estimator.set_params(random_state=1)
    return estimator


def _extract_sample_weight(weights, index):
    if weights is None:
        return None
    if np.ndim(weights) == 1:
        return weights
    return weights[index, :]


def get_weighted_predictors(regressor, c_train, X_train,weights=None, random_regr=False):
    [d,n_train] = c_train.shape
    predictors = {}
    for d_ in range(d): 
        regr = _build_estimator(regressor, random_regr=random_regr)
        sample_weight = _extract_sample_weight(weights, d_)
        if sample_weight is not None:
            regr.fit(X_train.T, c_train[d_,:], sample_weight=sample_weight)
        else: 
            regr.fit(X_train.T, c_train[d_,:])
        predictors[d_] = regr
    return predictors


def predict_costs(predictors, X):
    return np.asarray([predictors[d_].predict(X.T) for d_ in sorted(predictors.keys())])

def get_regret(predictors,X_train, c_train, trainDict, 
               X_test, c_test, testDict, 
               oracle, quiet=False):
    
    train_reg,train_x_reg = computeDataSetRegret(predictors, X_train, c_train, trainDict, oracle )
    test_reg, test_x_reg = computeDataSetRegret(predictors, X_test, c_test, testDict, oracle )

    return [train_reg,train_x_reg, test_reg, test_x_reg]

def computeDataSetRegret(predictors, X, c, dataDict, oracle):
    [d,n] = c.shape
    c_preds = predict_costs(predictors, X)
    
    regrets = np.zeros(n);
    x_star_regr = np.zeros(c_preds.shape)

    for i, instance in enumerate(dataDict): 
        [r_pred,x_pred] = oracle.solve(c_preds[:,i])
        
        regrets[i] = instance['opt_val']-instance['c'] @ x_pred; 
        x_star_regr[:,i]=instance['opt_sol'] -x_pred
        
    return regrets, x_star_regr


def _mean_abs(values):
    return np.mean(np.abs(values))


def compute_prediction_mse(predictors, X, c, sample_weight=None):
    c_preds = predict_costs(predictors, X)
    sample_errors = np.mean(np.square(c_preds - c), axis=0)
    if sample_weight is None:
        return float(np.mean(sample_errors))
    sample_weight = np.asarray(sample_weight).reshape((-1,))
    return float(np.average(sample_errors, weights=sample_weight))


def _signed_regret_to_loss(regrets):
    return np.abs(np.asarray(regrets).reshape((-1,)))


def _normalize_regret_weights(regret_loss, cap_quantile=0.9):
    regret_loss = np.asarray(regret_loss).reshape((-1,))
    scale = np.quantile(regret_loss, cap_quantile)
    if scale <= 0:
        scale = np.max(regret_loss)
    if scale <= 0:
        return np.zeros_like(regret_loss)
    return np.clip(regret_loss/scale, 0, 1)


def get_raw_context_weights(regrets, mu, cap_quantile=0.9, max_weight=1.0):
    '''
    Convert per-example realized regrets into bounded scalar sample weights.
    '''
    regret_loss = _signed_regret_to_loss(regrets)
    normalized_loss = _normalize_regret_weights(regret_loss, cap_quantile=cap_quantile)
    return (1-mu) + mu*max_weight*normalized_loss


def fit_reweight_predictors(regressor, x_star_regr, mu, c_train, X_train, random_regr=False):
    weights = mu*np.abs(x_star_regr) + (1-mu)*np.ones(c_train.shape)
    predictors = get_weighted_predictors(regressor, c_train, X_train, weights,
                                         random_regr=random_regr)
    return predictors, weights


def _build_weight_estimator(weight_regressor=None, weight_param_grid=None,
                            weight_cv=3, random_regr=False):
    if weight_param_grid is not None:
        if weight_regressor is None:
            raise ValueError('weight_regressor must be provided when weight_param_grid is set')
        estimator = _build_estimator(weight_regressor, random_regr=random_regr)
        if weight_cv < 2:
            return estimator
        return GridSearchCV(estimator, weight_param_grid, cv=weight_cv,
                            scoring='neg_mean_squared_error')

    if weight_regressor is None:
        return RidgeCV(alphas=np.logspace(-4, 4, 9), cv=weight_cv)

    return _build_estimator(weight_regressor, random_regr=random_regr)


def estimate_context_weights_cross_fitted(X_train, raw_weights, mu, n_folds=5,
                                          weight_regressor=None, weight_param_grid=None,
                                          weight_cv=3, max_weight=1.0, random_regr=False):
    '''
    Estimate context-only alpha weights by predicting raw decision weights
    out-of-fold from context features.
    '''
    raw_weights = np.asarray(raw_weights).reshape((-1,))
    n_train = X_train.shape[1]
    lower = 1 - mu
    upper = 1 - mu + mu*max_weight

    if n_train < 2:
        return np.clip(raw_weights, lower, upper)

    n_splits = min(n_folds, n_train)
    if n_splits < 2:
        return np.clip(raw_weights, lower, upper)

    fitted_weights = np.zeros(n_train)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    for train_index, test_index in splitter.split(np.arange(n_train)):
        estimator = _build_weight_estimator(weight_regressor=weight_regressor,
                                            weight_param_grid=weight_param_grid,
                                            weight_cv=min(weight_cv, len(train_index)),
                                            random_regr=random_regr)
        estimator.fit(X_train[:, train_index].T, raw_weights[train_index])
        fitted_weights[test_index] = estimator.predict(X_train[:, test_index].T)

    return np.clip(fitted_weights, lower, upper)


def fit_context_reweight_predictors(regressor, regrets, mu, c_train, X_train,
                                    weight_regressor=None, weight_param_grid=None,
                                    n_folds=5, weight_cv=3, cap_quantile=0.9,
                                    max_weight=1.0, random_regr=False):
    raw_weights = get_raw_context_weights(regrets, mu, cap_quantile=cap_quantile,
                                          max_weight=max_weight)
    fitted_weights = estimate_context_weights_cross_fitted(
        X_train, raw_weights, mu, n_folds=n_folds,
        weight_regressor=weight_regressor,
        weight_param_grid=weight_param_grid,
        weight_cv=weight_cv,
        max_weight=max_weight,
        random_regr=random_regr
    )
    predictors = get_weighted_predictors(regressor, c_train, X_train, fitted_weights,
                                         random_regr=random_regr)
    return predictors, raw_weights, fitted_weights


def evaluate_mu_diagnostics(predictors, mu, X_validation, c_validation, validationDict,
                            pilot_validation_regrets, regressor, algo_name, X_train,
                            c_train, graph_params, weight_regressor=None,
                            weight_param_grid=None, context_n_folds=5,
                            context_weight_cv=3, cap_quantile=0.9,
                            max_weight=1.0, diagnostic_cv_folds=5,
                            random_regr=False):
    oracle = ShortestPathOracle(graph_params)
    holdout_regrets, _ = computeDataSetRegret(predictors, X_validation, c_validation,
                                              validationDict, oracle)
    holdout_weights = get_raw_context_weights(
        pilot_validation_regrets, mu, cap_quantile=cap_quantile,
        max_weight=max_weight
    )
    cv_train_mse, cv_train_weighted_mse = cross_validate_mu_diagnostics(
        algo_name=algo_name,
        regressor=regressor,
        mu=mu,
        X_train=X_train,
        c_train=c_train,
        graph_params=graph_params,
        weight_regressor=weight_regressor,
        weight_param_grid=weight_param_grid,
        context_n_folds=context_n_folds,
        context_weight_cv=context_weight_cv,
        cap_quantile=cap_quantile,
        max_weight=max_weight,
        diagnostic_cv_folds=diagnostic_cv_folds,
        random_regr=random_regr,
    )
    return {
        'holdout_mse': compute_prediction_mse(predictors, X_validation, c_validation),
        'holdout_regret': _mean_abs(holdout_regrets),
        'pilot_regret_weighted_holdout_mse': compute_prediction_mse(
            predictors, X_validation, c_validation, sample_weight=holdout_weights
        ),
        'cv_train_mse': cv_train_mse,
        'cv_train_weighted_mse': cv_train_weighted_mse,
    }


def cross_validate_mu_diagnostics(algo_name, regressor, mu, X_train, c_train,
                                  graph_params, weight_regressor=None,
                                  weight_param_grid=None, context_n_folds=5,
                                  context_weight_cv=3, cap_quantile=0.9,
                                  max_weight=1.0, diagnostic_cv_folds=5,
                                  random_regr=False):
    n_train = X_train.shape[1]
    n_splits = min(diagnostic_cv_folds, n_train)
    if n_splits < 2:
        return np.nan, np.nan

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    oracle = ShortestPathOracle(graph_params)
    fold_mse = []
    fold_weighted_mse = []

    for train_index, val_index in splitter.split(np.arange(n_train)):
        X_fit = X_train[:, train_index]
        c_fit = c_train[:, train_index]
        X_val = X_train[:, val_index]
        c_val = c_train[:, val_index]

        fit_dict = generateInstanceDict(X_fit, c_fit, oracle)
        val_dict = generateInstanceDict(X_val, c_val, oracle)

        pilot_predictors = get_weighted_predictors(
            regressor, c_fit, X_fit, random_regr=random_regr
        )
        fit_regrets, fit_x_reg, val_pilot_regrets, _ = get_regret(
            pilot_predictors, X_fit, c_fit, fit_dict, X_val, c_val, val_dict, oracle
        )

        if algo_name == 'reweight_LS':
            candidate_predictors, _ = fit_reweight_predictors(
                regressor, fit_x_reg, mu, c_fit, X_fit, random_regr=random_regr
            )
        elif algo_name == 'context_reweight_LS':
            candidate_predictors, _, _ = fit_context_reweight_predictors(
                regressor, fit_regrets, mu, c_fit, X_fit,
                weight_regressor=weight_regressor,
                weight_param_grid=weight_param_grid,
                n_folds=context_n_folds,
                weight_cv=context_weight_cv,
                cap_quantile=cap_quantile,
                max_weight=max_weight,
                random_regr=random_regr,
            )
        else:
            raise ValueError('unknown algo_name for mu diagnostics: %s' % algo_name)

        val_weights = get_raw_context_weights(
            val_pilot_regrets, mu, cap_quantile=cap_quantile,
            max_weight=max_weight
        )
        fold_mse.append(compute_prediction_mse(candidate_predictors, X_val, c_val))
        fold_weighted_mse.append(
            compute_prediction_mse(candidate_predictors, X_val, c_val,
                                   sample_weight=val_weights)
        )

    return float(np.mean(fold_mse)), float(np.mean(fold_weighted_mse))

def feasible_least_squares(regressor, x_star_regr, mu, c_train, X_train, trainDict,
                           c_test, X_test, testDict, oracle, random_regr=False):
    '''
    One-step plug in 
    regressor: type of Scikit-Learn regressor
    x_star_regr: x^* - x_hat^*
    c_train: training data, cost vector realizations
    X_train: training data 
    '''
    weighted_predictors, weights = fit_reweight_predictors(
        regressor, x_star_regr, mu, c_train, X_train, random_regr=random_regr
    )
    [train_reg,train_x_reg, test_reg, test_x_reg] = get_regret(weighted_predictors,
            X_train, c_train, trainDict, X_test, c_test, testDict, oracle, quiet=False)
    return [weighted_predictors, train_reg,train_x_reg, test_reg, test_x_reg]


def feasible_context_least_squares(regressor, regrets, mu, c_train, X_train, trainDict,
                                   c_test, X_test, testDict, oracle,
                                   weight_regressor=None, weight_param_grid=None,
                                   n_folds=5, weight_cv=3, cap_quantile=0.9,
                                   max_weight=1.0, random_regr=False):
    '''
    One-step pilot + cross-fitted alpha regression + weighted least squares.
    '''
    weighted_predictors, raw_weights, fitted_weights = fit_context_reweight_predictors(
        regressor, regrets, mu, c_train, X_train,
        weight_regressor=weight_regressor,
        weight_param_grid=weight_param_grid,
        n_folds=n_folds,
        weight_cv=weight_cv,
        cap_quantile=cap_quantile,
        max_weight=max_weight,
        random_regr=random_regr
    )
    [train_reg, train_x_reg, test_reg, test_x_reg] = get_regret(
        weighted_predictors, X_train, c_train, trainDict,
        X_test, c_test, testDict, oracle, quiet=False
    )
    return [weighted_predictors, train_reg, train_x_reg, test_reg, test_x_reg,
            raw_weights, fitted_weights]
    
def runSPO(graph_params, trainDict, testDict):
    #warm start with LS
    learning_params = {
        'gradient': 'LS',
        'step_size_fn': 'long_dynamic'
    }
    oracle = ShortestPathOracle(graph_params)

    learner = SGDLearner(learning_params)
    p = trainDict[0]['features'].shape[0]
    d = trainDict[0]['c'].shape[0]
    LSmodel = LinearModel(p, d)
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
                                 num_reweights = 1, random_regr=False,
                                 context_mixture_weights=None,
                                 weight_regressor=None, weight_param_grid=None,
                                 context_n_folds=5, context_weight_cv=3,
                                 context_cap_quantile=0.9, context_max_weight=1.0,
                                 run_spo=True, compute_mu_diagnostics=False,
                                 diagnostic_cv_folds=5): 
    '''
    Run a replication under fixed data parameters 
    Assume pre-generated test dataset (to reduce noise in test evaluation)

    The optional context_* arguments activate the new context-only reweighting
    path, where scalar regret-based raw weights are regressed on context and
    then reused as sample weights in the final least-squares refit.
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
    validationDict = generateInstanceDict(X_validation, c_validation, oracle)

    # Learn Initial (LS) predictor 
    start_time = time.time()
    predictors = get_weighted_predictors(regressor, c_train, X_train, random_regr=random_regr )
    [regrets, x_star_regr,regrets_tst, x_star_regr_tst] = get_regret(predictors, X_train, c_train, trainDict,
                                                                      X_test, c_test, testDict, oracle)
    validation_regrets, validation_x_reg = computeDataSetRegret(
        predictors, X_validation, c_validation, validationDict, oracle
    )
    ls_time = time.time() - start_time
    res = {'n_train':n_train,
        'polykernel_degree': polykernel_degree,
        'n_test': n_test,
           'time': ls_time,
        'algo': 'LS',
        'tr_regret': _mean_abs(regrets),
        'tst_regret': _mean_abs(regrets_tst)
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
                'tr_regret': _mean_abs(train_reg),
                'tst_regret': _mean_abs(test_reg)
            }
            if compute_mu_diagnostics:
                res.update(evaluate_mu_diagnostics(
                    weighted_predictors,
                    mixture_weights[k],
                    X_validation,
                    c_validation,
                    validationDict,
                    validation_regrets,
                    regressor,
                    'reweight_LS',
                    X_train,
                    c_train,
                    graph_params,
                    weight_regressor=weight_regressor,
                    weight_param_grid=weight_param_grid,
                    context_n_folds=context_n_folds,
                    context_weight_cv=context_weight_cv,
                    cap_quantile=context_cap_quantile,
                    max_weight=context_max_weight,
                    diagnostic_cv_folds=diagnostic_cv_folds,
                    random_regr=random_regr,
                ))
            results.append(res)
            
            #For multiple reweights use the new weights
            x_star_regr = train_x_reg

    if context_mixture_weights is not None:
        for mu in context_mixture_weights:
            start_time = time.time()
            [weighted_predictors, train_reg, train_x_reg, test_reg, test_x_reg,
             raw_weights, fitted_weights] = feasible_context_least_squares(
                regressor, regrets, mu, c_train, X_train, trainDict,
                c_test, X_test, testDict, oracle,
                weight_regressor=weight_regressor,
                weight_param_grid=weight_param_grid,
                n_folds=context_n_folds,
                weight_cv=context_weight_cv,
                cap_quantile=context_cap_quantile,
                max_weight=context_max_weight,
                random_regr=random_regr
            )

            res = {
                'n_train': n_train,
                'polykernel_degree': polykernel_degree,
                'n_test': n_test,
                'time': time.time() - start_time + ls_time,
                'algo': 'context_reweight_LS',
                'mixture_weight': mu,
                'tr_regret': _mean_abs(train_reg),
                'tst_regret': _mean_abs(test_reg),
                'avg_raw_weight': np.mean(raw_weights),
                'avg_fitted_weight': np.mean(fitted_weights)
            }
            if compute_mu_diagnostics:
                res.update(evaluate_mu_diagnostics(
                    weighted_predictors,
                    mu,
                    X_validation,
                    c_validation,
                    validationDict,
                    validation_regrets,
                    regressor,
                    'context_reweight_LS',
                    X_train,
                    c_train,
                    graph_params,
                    weight_regressor=weight_regressor,
                    weight_param_grid=weight_param_grid,
                    context_n_folds=context_n_folds,
                    context_weight_cv=context_weight_cv,
                    cap_quantile=context_cap_quantile,
                    max_weight=context_max_weight,
                    diagnostic_cv_folds=diagnostic_cv_folds,
                    random_regr=random_regr,
                ))
            results.append(res)
    
    if run_spo:
        start_time = time.time()
        tr_reg, tst_reg = runSPO(graph_params, trainDict, testDict)
        
        res = {'n_train':n_train,
               'polykernel_degree': polykernel_degree,
               'n_test': n_test,
               'time': time.time() - start_time,
               'algo': 'SPO',
               'tr_regret': _mean_abs(tr_reg),
               'tst_regret': _mean_abs(tst_reg)
              }
        results.append(res)

    
    return results
