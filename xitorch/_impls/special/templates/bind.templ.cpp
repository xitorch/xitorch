#pragma once

#include <special/generated/batchfuncs.h>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(special_impl, m) {
  m.doc() = "Implementation of special functions";
  {%- for func in functions %}
  {%- for dtype in func["dtypes"].split(",") %}
  m.def("apply_{{ func['name'] }}_{{ dtype }}", &apply_{{ func["name"] }}_{{ dtype }}, "");
  {%- endfor %}
  {%- endfor %}
}
