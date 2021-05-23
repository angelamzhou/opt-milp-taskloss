import numpy as np

class ShortPathOracle(object):
    '''
    Gurobi Oracle Solver
    '''

    def __init__(self):
        pass

    def init_model(self, model_params, verbose = False):
        pass
        
        
    def solve(self,c):
        if c[0] < c[1]:
            opt_sol = np.array([1,0])
        else:
            opt_sol = np.array([0,1])

        return min(c), opt_sol