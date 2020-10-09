# Guide to add a new function

* Put the C/C++ files in `cfuncs` directory with `.c` or `.cpp` extension
  as well as the header (`.h` files)
* If you are making a new directory for your source inside `cfuncs`, add the
  directory in `sp_source_dirs` variable in `setup.py` (in the root directory)
* List the header of your function in `cfuncs/includes.h`
* Add your function in `functions.json`

## JSON file guide

An example of function in `functions.json` is

    {
      "name": "mysquare",
      "num_inp": 1,
      "num_out": 1,
      "cfuncs": {
        "f2f": {
          "cpu": "mysquare<float>"
        },
        "d2d": {
          "cpu": "mysquare<double>"
        }
      },
      "derivs": [
        "2 * gouts[0] * inps[0]"
      ]
    }

Here are the details of the fields:

* `"name"`: name of your square, it will be accessible through
  `xitorch.special.[name]`
* `"num_inp"`: number of inputs in the C/C++ functions
* `"num_out"`: number of outputs in the C/C++ functions
* `"cfuncs"`: dictionary indicating the name of the C/C++ functions based on
  the dtype signature and device (e.g. `"f2f"` means `float` input to `float`
  output, `"dd2d"` means `double, double` input to `double` output)
* `"derivs"`: the list of expression of the derivatives. Set it to `0` if the
  derivatives are not implemented yet. The variables are:
  * `gouts`: gradients of the outputs
  * `inps`: the input tensors
