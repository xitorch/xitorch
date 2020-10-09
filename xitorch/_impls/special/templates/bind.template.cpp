#include <torch/extension.h>
#include <special/generated/batchfuncs.h>
{%- from "macros.jinja" import fulldtype, liststr, apply_funcname %}

// generated file

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Implementation of special functions";

  {%- for func in functions %}
  {%- for sig in func.cfuncs %}

  {%- for device in func.cfuncs[sig] %}
  {%- set funcname = apply_funcname(func.name, device, sig) %}
  m.def("{{funcname}}", &{{funcname}}, "");
  {%- endfor %}

  {%- endfor %}
  {%- endfor %}
}
