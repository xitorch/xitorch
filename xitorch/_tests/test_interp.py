import torch
from torch.autograd import gradcheck, gradgradcheck
from xitorch.interpolate.interp1 import Interp1D
from xitorch._tests.utils import device_dtype_float_test

@device_dtype_float_test(only64=True)
def test_interp1_cspline(dtype, device):
    dtype_device_kwargs = {"dtype": dtype, "device": device}
    x = torch.tensor([0.0, 0.2, 0.3, 0.5, 0.8, 1.0], **dtype_device_kwargs).requires_grad_()
    y1 = torch.tensor([1.0, 2.1, 1.5, 1.1, 2.3, 2.5], **dtype_device_kwargs).requires_grad_()
    y2 = torch.tensor([[1.0, 2.1, 1.5, 1.1, 2.3, 2.5],
                       [0.0, 1.2, 2.2, 0.4, 3.2, 1.2]], **dtype_device_kwargs).requires_grad_()
    xq1 = torch.linspace(0, 1, 10, **dtype_device_kwargs).requires_grad_()
    xq2 = torch.linspace(0, 1, 4, **dtype_device_kwargs).requires_grad_()

    # true results (obtained from scipy.interpolate.CubicSpline)
    yq11_true = torch.tensor([1.        , 1.9264674 , 2.01250451, 1.3127193 , 1.04744964,
                              1.25577696, 1.7445744 , 2.22904002, 2.46166342, 2.5       ],
                        **dtype_device_kwargs)
    yq12_true = torch.tensor([1.       , 1.3127193, 1.7445744, 2.5      ], **dtype_device_kwargs)
    yq21_true = torch.tensor([[1.        , 1.9264674 , 2.01250451, 1.3127193 , 1.04744964,
                               1.25577696, 1.7445744 , 2.22904002, 2.46166342, 2.5       ],
                              [0.        , 0.4456999 , 1.47582714, 2.13368966, 0.86756719,
                               0.5025028 , 1.82654566, 3.11001848, 2.77726805, 1.2       ]],
                               **dtype_device_kwargs)
    yq22_true = torch.tensor([[1.       , 1.3127193, 1.7445744, 2.5      ],
                              [0.        , 2.13368966, 1.82654566, 1.2       ]],
                              **dtype_device_kwargs)

    def interp(x, y, xq):
        return Interp1D(x, y, method="cspline", bc_type="natural")(xq)

    yq11 = interp(x, y1, xq1)
    yq12 = interp(x, y1, xq2)
    yq21 = interp(x, y2, xq1)
    yq22 = interp(x, y2, xq2)
    assert torch.allclose(yq11, yq11_true)
    assert torch.allclose(yq12, yq12_true)
    assert torch.allclose(yq21, yq21_true)
    assert torch.allclose(yq22, yq22_true)

    gradcheck(interp, (x, y1, xq1))
    gradcheck(interp, (x, y1, xq2))
    gradcheck(interp, (x, y2, xq1))
    gradcheck(interp, (x, y2, xq2))

    gradgradcheck(interp, (x, y1, xq1))
    gradgradcheck(interp, (x, y1, xq2))
    gradgradcheck(interp, (x, y2, xq1))
    gradgradcheck(interp, (x, y2, xq2))

@device_dtype_float_test(only64=True)
def test_extrap(dtype, device):
    dtype_device_kwargs = {"dtype": dtype, "device": device}
    x = torch.tensor([0.0, 0.2, 0.3, 0.5, 0.8, 1.0], **dtype_device_kwargs).requires_grad_()
    y1 = torch.tensor([[1.0, 2.1, 1.5, 1.1, 2.3, 2.5],
                       [0.0, 1.2, 2.2, 0.4, 3.2, 1.2]], **dtype_device_kwargs).requires_grad_()
    xq1 = torch.tensor([0.0, 1./3, 2./3, 3./3, -1./3, -1.0, -4./3, 4./3, 6./3, 7./3, 9./3], **dtype_device_kwargs).requires_grad_()

    # true results (obtained from scipy.interpolate.CubicSpline)
    nan = float("nan")
    yq_nan_true = torch.tensor([
        [1., 1.3127193 , 1.7445744 , 2.5, nan, nan, nan, nan, nan, nan, nan],
        [0., 2.13368966, 1.82654566, 1.2, nan, nan, nan, nan, nan, nan, nan],
    ], **dtype_device_kwargs)
    yq_mir_true = torch.tensor([
        [1., 1.3127193 , 1.7445744 , 2.5, 1.3127193 , 1., 1.7445744 , 1.7445744 , 1., 1.3127193 , 2.5],
        [0., 2.13368966, 1.82654566, 1.2, 2.13368966, 0., 1.82654566, 1.82654566, 0., 2.13368966, 1.2],
    ], **dtype_device_kwargs)
    extraps = ["nan", "mirror"][:1]
    yq_trues = [yq_nan_true, yq_mir_true][:1]

    def interp(x, y, xq, extrap):
        return Interp1D(x, y, extrap=extrap, method="cspline", bc_type="natural")(xq)

    for extrap, yq_true in zip(extraps, yq_trues):
        print("Extrap: %s" % extrap)
        yq = interp(x, y1, xq1, extrap=extrap)
        assert torch.allclose(yq, yq_true, equal_nan=True)

if __name__ == "__main__":
    test_interp1_cspline()
