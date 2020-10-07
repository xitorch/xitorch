#pragma once

#include "batchfuncs.h"
#include <cstdint>
#include <torch/torch.h>
#include <special/cfuncs/includes.h>

{%- for func in functions %}
{%- for dtype in func.dtypes.strip().split(",") %}
{%- set dtype_full = {"d": "double", "f": "float"}[dtype] %}

{%- if func.num_io.strip() == "1,1"%}

void apply_{{func.name}}_{{dtype}}(torch::Tensor& self, torch::Tensor& out) {
  auto self_data = self.data_ptr<{{dtype_full}}>();
  auto out_data = out.data_ptr<{{dtype_full}}>();

  auto self_numel = self.numel();

  for (int64_t i = 0; i < self_numel; ++i) {
    auto* self_working_ptr = &self_data[i];
    auto* out_working_ptr = &out_data[i];
    *out_working_ptr = {{func.funcname}}<{{dtype_full}}>(*self_working_ptr);
  }
}
{%- endif %}

{%- endfor %}
{%- endfor %}
