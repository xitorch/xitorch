import itertools
import torch
import xitorch as xt
import pytest
import argparse
from typing import Mapping, List, ClassVar

__all__ = ["device_dtype_float_test"]

def device_dtype_float_test(only64:int=False, onlycpu:bool=False,
        additional_kwargs:Mapping[str,List]={}):
    dtypes = [torch.float, torch.float64]
    devices = [torch.device("cpu"), torch.device("cuda")]
    if only64:
        dtypes = [torch.float64]
    if onlycpu or not torch.cuda.is_available():
        devices = [torch.device("cpu")]
    kwargs_vals = additional_kwargs.values()
    argnames = ",".join(["dtype", "device"] + list(additional_kwargs.keys()))
    params = [*itertools.product(dtypes, devices, *kwargs_vals)]
    return pytest.mark.parametrize(argnames, params)
