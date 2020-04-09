import torch

def set_default_option(defopt, opt=None):
    if opt is None:
        opt = {}
    defopt.update(opt)
    return defopt

def extract_differentiable_tensors(model):
    res = list(model.parameters())

    # add differentiable and non-parameters attribute in the model
    for varname in model.__dict__:
        var = model.__dict__[varname]
        if not isinstance(var, torch.Tensor): continue
        if var.requires_grad:
            res.append(var)
    return res
