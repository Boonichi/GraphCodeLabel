import sys

_model_entry_points = {}

def register_model(fn):
    # Look up model 
    model_name = fn.__name__
    
    _model_entry_points[model_name] = fn
    
    return fn

def model_entrypoint(model_name):
    
    return _model_entry_points[model_name]

def is_model(model_name):
    '''Check exist model'''
    return model_name in _model_entry_points    

def create_model(model_name : str, **kwargs):
    print(_model_entry_points)
    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
    else:
        RuntimeError("Unknown model (%s)" % model_name)
    
    model = create_fn(**kwargs)
    print(model)
    return model 