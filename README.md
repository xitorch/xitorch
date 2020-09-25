# `xitorch`: differentiable scientific computing library

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

    git clone https://github.com/mfkasim91/xitorch/
    cd xitorch
    python -m pip install -e .

## Gallery

Neural mirror design ([example 01](examples/01-mirror-design/)):

![neural mirror design](examples/01-mirror-design/images/mirror.gif)

Initial velocity optimization in molecular dynamics ([example 02](examples/02-molecular-dynamics/)):

![molecular dynamics](examples/02-molecular-dynamics/images/md.gif)
