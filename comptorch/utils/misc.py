import torch

def set_default_option(defopt, opt=None):
    if opt is None:
        opt = {}
    defopt.update(opt)
    return defopt
