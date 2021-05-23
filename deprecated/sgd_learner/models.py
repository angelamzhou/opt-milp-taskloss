import numpy as np

class LinearModel(object):
    
    def __init__(self, p, d):
        self.p = p
        self.d = d
        self.model = np.zeros((d,p))
        self.model_avg = np.zeros((d,p))
        self.step_size_sum = 0


    def forward(self, features, avg = False):
        if avg:
            return (self.model_avg @ features.reshape((self.p,1))).reshape((self.d,))
        else:
            return (self.model @ features.reshape((self.p,1))).reshape((self.d,))

    def backward(self, features, weights):
        return (weights.reshape((self.d,1)) @ features.reshape((1,self.p)))
    
    def update(self, grad, step_size):
        self.step_size_sum += step_size
        step_avg = step_size/self.step_size_sum
        
        self.model = self.model - step_size*grad
        self.model_avg = (1 - step_avg)*self.model_avg + step_avg*self.model