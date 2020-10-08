#pragma once

#include <torch/torch.h>

// generated file

{%- from "macros.jinja" import fulldtype, liststr, apply_funcname %}

{%- for func in functions %}
{%- for sig in func.cfuncs %}
{%- set inp_dtypes, out_dtypes = sig.split("2") %}
{%- set num_inp = inp_dtypes|length %}
{%- set num_out = out_dtypes|length %}

{%- for device in func.cfuncs[sig] %}
void {{apply_funcname(func.name, device, sig)}}({{liststr(num_inp,"torch::Tensor& self")}}, {{liststr(num_out, "torch::Tensor& out")}});
{%- endfor %}

{%- endfor %}
{%- endfor %}
