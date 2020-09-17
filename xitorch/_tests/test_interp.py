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
        return Interp1D(x, y)(xq)

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

if __name__ == "__main__":
    test_interp1_cspline()
