## Implementation of the forward models

This directory is dedicated for implementation of the forward models.
To add a new implementation here, please follow the rules:

* All functions in the functional input are assumed to produce a single output tensor (so no need to use `torch.cat`)
* The input arguments for the functions (in the functional input) should be made as another argument in the functional
