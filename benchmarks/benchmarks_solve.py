import os
import warnings
import torch
from xitorch.linalg import solve, symeig
from xitorch import LinearOperator
from xitorch._utils.tensor import create_random_square_matrix

thisfile_dir = os.path.dirname(os.path.realpath(__file__))

class SolveMatrixTimeSuite:
    params = [
        [True, False],
        [(-1.0, 1.0), (0.0, 1.0), (0.2, 1.0), (0.5, 1.0)],
        [100, 350, 700],
    ]
    param_names = ["is_hermitian", "minmaxeival", "n"]

    def setup(self, is_hermitian, minmaxeival, n):
        seed = 123
        ncols = 50
        torch.manual_seed(seed)
        min_eival, max_eival = minmaxeival
        A = create_random_square_matrix(
            n, is_hermitian=is_hermitian,
            min_eival=min_eival, max_eival=max_eival,
            seed=seed)
        self.A = LinearOperator.m(A, is_hermitian=is_hermitian)
        X = torch.randn(n, ncols, dtype=A.dtype)
        self.B = self.A.mm(X)

    def time_matrix_AB(self, *args, **kwargs):
        with warnings.catch_warnings(record=True) as ws:
            X = solve(self.A, self.B)
            # see warnings about convergence and turns it to an error
            _catch_convergence_warnings(ws)

class SymeigMatrixTimeSuite:
    params = [
        [(-1.0, 1.0), (0.0, 1.0), (0.2, 1.0), (0.5, 1.0)],
        [100, 350, 700],
    ]
    param_names = ["minmaxeival", "n"]

    def setup(self, minmaxeival, n):
        seed = 123
        ncols = 50
        torch.manual_seed(seed)
        min_eival, max_eival = minmaxeival
        A = create_random_square_matrix(
            n, is_hermitian=True,
            min_eival=min_eival, max_eival=max_eival,
            seed=seed)
        self.A = LinearOperator.m(A, is_hermitian=True)

    def time_matrix_AB(self, *args, **kwargs):
        with warnings.catch_warnings(record=True) as ws:
            X = symeig(self.A, neig=10, mode="lowest")
            # see warnings about convergence and turns it to an error
            _catch_convergence_warnings(ws)

def _catch_convergence_warnings(ws):
    for wnings in ws:
        w = str(wnings.message)
        if "converge" in w.lower():
            raise RuntimeError(w)
