import torch
from _xitorch_special_impl import *
{%- from "macros.jinja" import fulldtype, liststr, apply_funcname, ifs %}

# generated file

__all__ = [
    {%- for func in functions %}
    "{{func.name}}",
    {%- endfor %}
]

def get_signature_and_device(inps, outs):
    dtype_map = {
        torch.float32: "f",
        torch.float64: "d"
    }
    idtypes = "".join([dtype_map[inp.dtype] for inp in inps])
    odtypes = "".join([dtype_map[out.dtype] for out in outs])
    signature = idtypes + "2" + odtypes
    device = str(inps[0].device).split(":")[0]
    return signature, device

###################### generated functions ######################

{%- for func in functions %}
{%- set num_inp = func.num_inp %}
{%- set num_out = func.num_out %}
{%- set inp_sig = liststr(num_inp, "inp") %}
{%- set out_sig = liststr(num_out, "out", suffix="=None") %}
{%- set out_sig2 = liststr(num_out, "out") %}
{%- set gout_sig = liststr(num_out, "gout") %}

################# {{func.name}} #################
def {{func.name}}({{inp_sig}}, {{out_sig}}):
    {%- for i in range(num_out) %}
    if out{{i}} is None:
        out{{i}} = torch.empty_like(inp0) # TODO: fix this ???
    {%- endfor %}
    return PyFunc_{{func.name}}.apply({{inp_sig}}, {{out_sig2}})

def _{{func.name}}({{inp_sig}}, {{out_sig2}}):
    signature, device = get_signature_and_device(inps=[{{inp_sig}}], outs=[{{out_sig2}}])
    {%- for sig in func.cfuncs %}
    {{ifs(loop.index0)}} signature == "{{sig}}":
        {%- for device in func.cfuncs[sig] %}
        {{ifs(loop.index0)}} device == "{{device}}":
            {{apply_funcname(func.name, device, sig)}}({{inp_sig}}, {{out_sig2}})
            return {{out_sig2}}
        {%- endfor %}
    {%- endfor %}
    raise RuntimeError("The function {{func.name}} has no %s (device: %s) defined." % \
                       (signature.replace("2", "->"), device))

class PyFunc_{{func.name}}(torch.autograd.Function):
    @staticmethod
    def forward(ctx, {{inp_sig}}, {{out_sig2}}):
        res = _{{func.name}}({{inp_sig}}, {{out_sig2}})
        ctx.save_for_backward({{inp_sig}}, {{out_sig2}})
        return res

    @staticmethod
    def backward(ctx, {{gout_sig}}):
        {%- if func.derivs == 0 %}
        raise NotImplementedError("Backward of {{func.name}} is not implemented. ")

        {%- else %}
        {{inp_sig}}, {{out_sig2}} = ctx.saved_tensors
        derivs = (
            {%- for deriv in func.derivs %}
            {{deriv}},
            {%- endfor %}
            {%- for i in range(num_out) %}
            None,
            {%- endfor %}
        )
        return derivs

        {%- endif %}

{%- endfor %}
