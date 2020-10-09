# Guide to add a new function

* Put the C/C++ files in `cfuncs` directory with `.c` or `.cpp` extension
  as well as the header (`.h` files)
* If you are making a new directory for your source inside `cfuncs`, add the
  directory in `sp_source_dirs` variable in `setup.py` (in the root directory)
* List the header of your function in `cfuncs/includes.h`
* Add your function in `functions.yaml`
* Add the docstring in `__init__.py`

## YAML file guide

An example of function in `functions.yaml` is

    - name: mysquare
      inp: x
      out: out
      cfuncs:
        f2f:
          cpu: mysquare<float>
        d2d:
          cpu: mysquare<double>
      derivs:
      - 2 * grad_out * x

Here are the details of the fields:

* `"name"`: name of your square, it will be accessible through
  `xitorch.special.[name]`
* `"inp"`: signature of the inputs in the API (separated by a comma, without space)
* `"out"`: signature of the outputs in the API (separated by a comma, without space)
* `"cfuncs"`: dictionary indicating the name of the C/C++ functions based on
  the dtype signature and device (e.g. `"f2f"` means `float` input to `float`
  output, `"dd2d"` means `double, double` input to `double` output)
* `"derivs"`: the list of expression of the derivatives. Set it to `0` if the
  derivatives are not implemented yet. The variables correspond to the ones
  specified in `inp` and `out`. The gradient is prefixed with `grad_`
