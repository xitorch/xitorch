# `xitorch`: differentiable scientific computing library

[Build](https://img.shields.io/github/workflow/status/mfkasim1/xitorch/Python%20package)
[Docs](https://img.shields.io/readthedocs/xitorch)
[License](https://img.shields.io/github/license/mfkasim1/xitorch)

`xitorch` is a PyTorch-based library of differentiable functions and functionals that
can be widely used in scientific computing applications as well as deep learning.

The documentation can be found at: https://xitorch.readthedocs.io/

## Modules

* [`linalg`](xitorch/linalg/): Linear algebra and sparse linear algebra module
* [`optimize`](xitorch/optimize/): Optimization and root finder module
* [`integrate`](xitorch/integrate/): Quadrature and integration module

## Requirements

* python 3.6 or higher
* pytorch 1.6 or higher (install [here](https://pytorch.org/))

## Getting started

After fulfilling all the requirements, type the commands below to install `xitorch`

    python -m pip install xitorch

Or if you want to install from source:

    git clone https://github.com/mfkasim1/xitorch/
    cd xitorch
    python -m pip install -r requirements.txt -e .

## Gallery

Neural mirror design ([example 01](examples/01-mirror-design/)):

![neural mirror design](examples/01-mirror-design/images/mirror.gif)

Initial velocity optimization in molecular dynamics ([example 02](examples/02-molecular-dynamics/)):

![molecular dynamics](examples/02-molecular-dynamics/images/md.gif)
