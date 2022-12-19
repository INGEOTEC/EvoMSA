# Copyright 2022 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from scipy.stats import kruskal
import numpy as np


class KruskalFS(object):
    def __init__(self, alpha=0.05) -> None:
        self._alpha = alpha

    def fit(self, X, y):
        labels = np.unique(y)
        res = [kruskal(*[X[y==l, i] for l in labels]).pvalue
               for i in range(X.shape[1])]
        self._pvalues = np.array(res)
        return self

    def get_support(self, indices=False):
        mask = self._pvalues < self._alpha
        if indices:
            return np.where(mask)[0]
        return mask