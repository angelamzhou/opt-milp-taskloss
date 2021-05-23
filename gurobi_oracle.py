import gurobipy as gp
import numpy as np

class GurobiSolver(object):
    '''
    Gurobi Oracle Solver
    '''

    def __init__(self):
        pass

    def init_model(self, model_params, verbose = False):
        
        self.A = model_params['A'] if 'A' in model_params else None
        self.b = model_params['b'] if 'b' in model_params else None
        
        #self.relax = model_params['relax'] if 'relax' in model_params else False
        type_dict = {
            'binary': 'B',
            'integer': 'I',
            'continuous': 'C'
        }
        obj_dict = {
            'max': -1,
            'min': 1,
        }
        self.var_type = model_params['var_type'] if 'var_type' in model_params else 'binary'
        self.var_type = type_dict[self.var_type]
        
        self.objective = model_params['objective'] if 'objective' in model_params else 'max'
        self.objective = obj_dict[self.objective]

        self.model = gp.Model('solver')
        
        if not verbose: self.model.setParam("OutputFlag", 0)

        self.x = self.model.addMVar((self.A.shape[1],), obj=0, vtype=self.var_type, name="")
        self.model.addConstr(self.A @ self.x <= self.b)
        
    def solve(self,c):
        self.model.setObjective(c @ self.x, self.objective)
        self.model.optimize()
        return self.model.objVal, np.array([v.x for v in self.model.getVars()])