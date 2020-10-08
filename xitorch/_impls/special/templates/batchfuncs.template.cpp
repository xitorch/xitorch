#pragma once

#include "batchfuncs.h"
#include <cstdint>
#include <torch/torch.h>
#include <special/cfuncs/includes.h>

// generated file

{%- from "macros.jinja" import fulldtype, liststr, apply_funcname %}

{%- for func in functions %}
{%- for sig in func.cfuncs %}
{%- set inp_dtypes, out_dtypes = sig.split("2") %}
{%- set num_inp = inp_dtypes|length %}
{%- set num_out = out_dtypes|length %}

{%- for device in func.cfuncs[sig] %}
{%- set cfuncname = func.cfuncs[sig][device] %}

void {{apply_funcname(func.name, device, sig)}}({{liststr(num_inp,"torch::Tensor& self")}}, {{liststr(num_out, "torch::Tensor& out")}}) {
  {%- for i in range(num_inp) %}
  auto self_data{{i}} = self{{i}}.data_ptr<{{fulldtype(inp_dtypes[i])}}>();
  {%- endfor %}
  {%- for i in range(num_out) %}
  auto out_data{{i}} = out{{i}}.data_ptr<{{fulldtype(out_dtypes[i])}}>();
  {%- endfor %}

  // expand it to multiple inputs?
  auto self_numel = self0.numel();

  for (int64_t i = 0; i < self_numel; ++i) {
    {%- for i in range(num_inp) %}
    auto* self_working_ptr{{i}} = &self_data{{i}}[i];
    {%- endfor %}
    {%- for i in range(num_out) %}
    auto* out_working_ptr{{i}} = &out_data{{i}}[i];
    {%- endfor %}
    // {{device}}
    {{liststr(num_out, "*out_working_ptr")}} = {{cfuncname}}({{liststr(num_inp, "*self_working_ptr")}});
  }
}

{%- endfor %}

{%- endfor %}
{%- endfor %}
