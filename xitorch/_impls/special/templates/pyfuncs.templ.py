import torch
from xitorch.special_impl import *

__all__ = [
    {%- for func in functions %}
    "{{func.name}}",
    {%- endfor %}
]
{% for func in functions %}
{%- if func.num_io.strip() == "1,1"%}
def {{func.name}}(a, out=None):
    if out is None:
        out = torch.empty_like(a)
    return PyFunc_{{func.name}}.apply(a, out)

def _{{func.name}}(a, out):
    if 0: # dummy for the template
        pass
    {%- if "d" in func.dtypes %}
    elif a.dtype == torch.float64:
        apply_{{func.name}}_d(a, out)
    {%- endif %}
    {%- if "f" in func.dtypes %}
    elif a.dtype == torch.float32:
        apply_{{func.name}}_f(a, out)
    {%- endif %}
    else:
        raise TypeError("No implementation of {{func.name}} for type: %s" % a.dtype)
    return out
{%- endif %}

class PyFunc_{{func.name}}(torch.nn.Module):
    @staticmethod
    def forward(ctx, *inps):
        res = _{{func.name}}(*inps)
        ctx.save_for_backward(inps)
        return res

    @staticmethod
    def backward(ctx, *gouts):
        inps = ctx.saved_tensors
        derivs = [
            {%- for deriv in func.derivs %}
            {{deriv}},
            {%- endfor %}
        ]
        all_derivs = derivs + ([None] * (len(inps)-len(derivs)))
        return tuple(all_derivs)

{%- endfor %}
