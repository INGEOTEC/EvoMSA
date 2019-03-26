# Copyright 2019 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cpython.list cimport PyList_Append
from cpython cimport array

cdef class TextModelPredict:
    cdef public array.array coef
    cdef public array.array intercept
    cdef public Py_ssize_t nklass
    cdef public object textModel

    def __cinit__(self, object textModel, array.array coef, array.array intercept):
        self.textModel = textModel
        self.coef = coef
        self.intercept = intercept
        self.nklass = len(intercept)

    def transform(self, list data, list output):
        cdef str x
        cdef Py_ssize_t k, init, j
        cdef double v
        cdef array.array _output
        cdef double *_output_c
        cdef double *coef_c = self.coef.data.as_doubles
        for x in data:
            _output = array.copy(self.intercept)
            self._transform(x, _output)
            PyList_Append(output, _output)

    cpdef void _transform(self, str x, array.array _output):
        cdef Py_ssize_t k, init, j
        cdef double v
        cdef double *coef_c = self.coef.data.as_doubles
        _output_c = _output.data.as_doubles
        for k, v in self.textModel[x]:
            init = self.nklass * k
            for j in range(self.nklass):
                _output_c[j] = _output_c[j] + coef_c[init + j] * v

    def __getitem__(self, text):
        _output = array.copy(self.intercept)
        self._transform(text, _output)
        return _output
