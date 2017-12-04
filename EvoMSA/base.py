# Copyright 2017 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from b4msa.command_line import load_json
from b4msa.textmodel import TextModel
from b4msa.classifier import SVC
from sklearn.model_selection import KFold
from EvoDAG.model import EvoDAGE
import numpy as np


class EvoMSA(object):
    def __init__(self, b4msa_params=None, evodag_args=dict(),
                 b4msa_args=dict(), n_jobs=1, n_splits=5, seed=0):
        if b4msa_params is None:
            b4msa_params = os.path.join(os.path.dirname(__file__),
                                        'conf', 'default_parameters.json')
        b4msa_params = self.read_json(b4msa_params)
        b4msa_params.update(b4msa_args)
        self._b4msa_args = b4msa_args
        self._evodag_args = evodag_args
        self._n_jobs = n_jobs
        self._n_splits = n_splits
        self._seed = seed
        self._svc_models = None
        self._evodag_model = None

    def model(self, X):
        if not isinstance(X[0], list):
            X = [X]
        m = []
        kwargs = self._b4msa_args
        for x in X:
            m.append(TextModel(x, **kwargs))
        self._textModel = m

    def vector_space(self, X):
        if not isinstance(X[0], list):
            X = [X]
        return [[m[_] for _ in x] for m, x in zip(self._textModel, X)]

    def kfold_decision_function(self, X, y):
        hy = [None for x in y]
        for tr, ts in KFold(n_splits=self._n_splits,
                            shuffle=True, random_state=self._seed).split(X):
            c = SVC(model=None)
            c.fit([X[x] for x in tr], [y[x] for x in tr])
            _ = c.decision_function([X[x] for x in ts])
            [hy.__setitem__(k, self.tolist(v)) for k, v in zip(ts, _)]
        return hy

    def predict(self, X):
        X = self.transform(X)
        hy = self._evodag_model.predict(X)
        le = self._svc_models[0].le
        return le.inverse_transform(hy)

    def transform(self, X, y=None):
        D = None
        for m, t in zip(self._svc_models, self._textModel):
            x = [t[_] for _ in X]
            if y is not None:
                d = self.kfold_decision_function(x, y)
                y = None
            else:
                d = [self.tolist(_) for _ in m.decision_function(x)]
            if D is None:
                D = d
            else:
                [x.__iadd__(y) for x, y in zip(D, d)]
        return np.array(D)

    def fit(self, X, y):
        self.model(X)
        Xvs = self.vector_space(X)
        if not isinstance(y[0], list):
            y = [y]
        svc_models = []
        for t, x, y0 in zip(self._textModel, Xvs, y):
            c = SVC(model=t)
            c.fit(x, y0)
            svc_models.append(c)
        self._svc_models = svc_models
        if isinstance(X[0], list):
            X = X[0]
        D = self.transform(X, y[0])
        self._evodag_model = EvoDAGE(n_jobs=self._n_jobs,
                                     seed=self._seed,
                                     **self._evodag_args).fit(D, svc_models[0].le.transform(y[0]))
        return self

    @staticmethod
    def tolist(x):
        if isinstance(x, list):
            return x
        elif isinstance(x, np.ndarray):
            return x.tolist()
        else:
            return [x]

    @staticmethod
    def read_json(fname):
        kw = load_json(fname)
        if isinstance(kw, list):
            kw = kw[0]
        return kw
