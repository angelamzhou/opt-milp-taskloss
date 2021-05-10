def SPO_plus(instance, oracle, model):
    spo_plus_cost = 2*model.forward(instance['features']) - instance['c']
    oracle.init_model(instance)
    obj, spo_opt_sol = oracle.solve(spo_plus_cost)
    
    if instance['objective'] == 'max':
        weights = -2*(instance['opt_sol'] - spo_opt_sol)
    else:
        weights = 2*(instance['opt_sol'] - spo_opt_sol)
    return model.backward(instance['features'], weights)

def decision_loss_mse(instance, oracle, model, mu = 0.5):
    estimated_cost = model.forward(instance['features'])
    oracle.init_model(instance)
    try:
        obj, opt_sol = oracle.solve(estimated_cost)
        weights = 2*(mu*abs(instance['opt_val'] - instance['c'] @ opt_sol) + (1-mu))*(estimated_cost - instance['c'])
    except:
        print(estimated_cost)
    return model.backward(instance['features'], weights)

def LS(instance, oracle, model):
    estimated_cost = model.forward(instance['features'])
    weights = 2*(estimated_cost - instance['c'])
    return model.backward(instance['features'], weights)

def eval_task_loss(instance, model, oracle):
    
    estimated_cost = model.forward(instance['features'])
    oracle.init_model(instance)
    obj, opt_sol = oracle.solve(estimated_cost)
    
    return abs(instance['opt_val'] - instance['c'] @ opt_sol)
