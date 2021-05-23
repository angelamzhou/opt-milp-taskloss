import copy
import random
from gradients_object import *
import numpy as np

class SGDLearner(object):
    
    def __init__(self, params):
        gradient_fns = {
            'spo_plus': SPO_plus(params),
            'decision_loss_mse': decision_loss_mse(params),
            'LS': LS(params),
            'LS_weighted': LS_Weighed(params)
        }
        
        gradient = params['gradient'] if 'gradient' in params else 'spo_plus'
        self.subgrad = gradient_fns[gradient]
        
        step_size_fns = {
            'constant': lambda x: params['step_size'] if 'step_size' in params else 1e-4,
            'short_dynamic': lambda x: 2/(x+2),
            'long_dynamic': lambda x: 1e-2/np.sqrt(x+2)
        }
        
        step_size = params['step_size_fn'] if 'step_size_fn' in params else 'constant'
        self.step_size = step_size_fns[step_size]
        
    def learn(self, oracle, model, 
              instances_train, instances_validation = None,
              epochs = 1000, batch_size = 10, validation_period = 200,
             verbose = False):
    
        validation = instances_validation is not None

        if validation:
            task_loss = 0
            for j in range(len(instances_validation)):
                task_loss += eval_task_loss(instances_validation[j], model, oracle)
            
            best_loss = task_loss/len(instances_validation)
            best_model = copy.copy(model)
        
        sample_ordering = list(range(len(instances_train)))

        for i in range(epochs):
            if verbose: print('EPOCH %d ***'%(i))
    
            random.shuffle(sample_ordering)
            G = np.zeros(model.model.shape)

            for j in range(batch_size):
                instance = instances_train[sample_ordering[j]]
                G += self.subgrad.computeGrad(instance, oracle, model)
            
            G = 1/batch_size*G
            if verbose: print('G ', G)

            model.update(G, self.step_size(i))
            
            if validation and (i % validation_period == 0):
                task_loss = 0
                for j in range(len(instances_validation)):
                    task_loss += eval_task_loss(instances_validation[j], model, oracle)
                
                task_loss = task_loss/len(instances_validation)
                
                if task_loss < best_loss:
                    best_loss = task_loss
                    best_model = copy.copy(model)
                    
        
        
        task_loss = 0
        for j in range(len(instances_train)):
            task_loss += eval_task_loss(instances_train[j], model, oracle)
        task_loss = task_loss/len(instances_train)
        print('avg train loss', task_loss)
        
        if validation:
            print('best loss', best_loss)
            return best_model
        else:
            return model