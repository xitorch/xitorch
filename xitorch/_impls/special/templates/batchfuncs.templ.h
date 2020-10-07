#pragma once

#include <torch/torch.h>

{%- for func in functions %}
{%- for dtype in func.dtypes.split(",") %}

{%- if func.num_io.strip() == "1,1"%}
void apply_{{func.name}}_{{dtype}}(torch::Tensor& self, torch::Tensor& out);
{%- endif %}

{%- endfor %}
{%- endfor %}
