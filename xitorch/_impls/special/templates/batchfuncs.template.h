#pragma once

#include <torch/torch.h>

// generated file

{%- from "macros.jinja" import fulldtype, liststr, apply_funcname, joinlist %}

{%- for func in functions %}
{%- for sig in func.cfuncs %}

{%- set inps = func.inp.split(",") %}
{%- set outs = func.out.split(",") %}
{%- set num_inp = inps|length %}
{%- set num_out = outs|length %}
{%- set inp_sig = joinlist(inps, ", ", prefix="torch::Tensor& ") %}
{%- set out_sig = joinlist(outs, ", ", prefix="torch::Tensor& ") %}

{%- for device in func.cfuncs[sig] %}
void {{apply_funcname(func.name, device, sig)}}({{inp_sig}}, {{out_sig}});
{%- endfor %}

{%- endfor %}
{%- endfor %}
