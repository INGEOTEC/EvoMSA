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
from sklearn.preprocessing import LabelEncoder
from EvoDAG.model import EvoDAGE
from sklearn.linear_model import LogisticRegression
from .calibration import Calibration
import numpy as np
import logging
from multiprocessing import Pool
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def kfold_decision_function(args):
    X, y, tr, ts = args
    c = SVC(model=None)
    c.fit([X[x] for x in tr], [y[x] for x in tr])
    return ts, c.decision_function([X[x] for x in ts])


class EvoMSA(object):
    def __init__(self, use_ts=True, b4msa_params=None, evodag_args=dict(),
                 b4msa_args=dict(), n_jobs=1, n_splits=5, seed=0, logistic_regression=False,
                 logistic_regression_args=None, probability_calibration=False):
        self._use_ts = use_ts
        if b4msa_params is None:
            b4msa_params = os.path.join(os.path.dirname(__file__),
                                        'conf', 'default_parameters.json')
        b4msa_params = self.read_json(b4msa_params)
        b4msa_params.update(b4msa_args)
        self._b4msa_args = b4msa_params
        self._evodag_args = evodag_args
        self._n_jobs = n_jobs
        self._n_splits = n_splits
        self._seed = seed
        self._svc_models = None
        self._evodag_model = None
        self._logger = logging.getLogger('EvoMSA')
        self._le = None
        self._logistic_regression = None
        if logistic_regression:
            p = dict(random_state=self._seed, class_weight='balanced')
            if logistic_regression_args is not None:
                p.update(logistic_regression_args)
            self._logistic_regression = LogisticRegression(**p)
        self._exogenous = None
        self._probability_calibration = probability_calibration

    def model(self, X):
        if not isinstance(X[0], list):
            X = [X]
        m = []
        kwargs = self._b4msa_args
        self._logger.info("Starting TextModel")
        self._logger.info(str(kwargs))
        for x in X:
            m.append(TextModel(x, **kwargs))
        self._textModel = m

    def vector_space(self, X):
        if not isinstance(X[0], list):
            X = [X]
        return [[m[_] for _ in x] for m, x in zip(self._textModel, X)]

    def kfold_decision_function(self, X, y):
        hy = [None for x in y]
        args = []
        for tr, ts in KFold(n_splits=self._n_splits,
                            shuffle=True, random_state=self._seed).split(X):
            args.append([X, y, tr, ts])
        if self._n_jobs == 1:
            res = [kfold_decision_function(x) for x in tqdm(args, total=len(args))]
        else:
            p = Pool(self._n_jobs, maxtasksperchild=1)
            res = [x for x in tqdm(p.imap_unordered(kfold_decision_function, args),
                                   total=len(args))]
            p.close()
        for ts, df in res:
            [hy.__setitem__(k, self.tolist(v)) for k, v in zip(ts, df)]
        return hy

    def predict(self, X):
        pr = self.predict_proba(X)
        return self._le.inverse_transform(pr.argmax(axis=1))

    def predict_proba(self, X):
        X = self.transform(X)
        if self._logistic_regression is not None:
            X = self._evodag_model.raw_decision_function(X)
            return self._logistic_regression.predict_proba(X)
        # elif self._probability_calibration:
        #     df = self._evodag_model._decision_function_raw(X, cpu_cores=self._n_jobs)
        #     return self._calibration_coef.predict_proba(df)
        return self._evodag_model.predict_proba(X)

    def raw_decision_function(self, X):
        X = self.transform(X)
        return self._evodag_model.raw_decision_function(X)

    def decision_function(self, X):
        X = self.transform(X)
        return self._evodag_model.decision_function(X)

    @property
    def exogenous(self):
        return self._exogenous

    @exogenous.setter
    def exogenous(self, a):
        self._exogenous = a

    def append_exogenous(self, d):
        e = self.exogenous
        if e is not None:
            return np.concatenate((d, e), axis=1)
        return d

    def transform(self, X, y=None):
        D = None
        for m, t in zip(self._svc_models, self._textModel):
            if m is None:
                y = None
                continue
            x = [t[str(_)] for _ in X]
            if y is not None:
                d = self.kfold_decision_function(x, y)
                y = None
            else:
                d = [self.tolist(_) for _ in m.decision_function(x)]
            if D is None:
                D = d
            else:
                [v.__iadd__(w) for v, w in zip(D, d)]
        _ = np.array(D)
        return self.append_exogenous(_)

    def fit_svm(self, X, y):
        n_use_ts = not self._use_ts
        self.model(X)
        Xvs = self.vector_space(X)
        if not isinstance(y[0], list):
            y = [y]
        svc_models = []
        for t, x, y0 in zip(self._textModel, Xvs, y):
            if n_use_ts:
                svc_models.append(None)
                n_use_ts = False
                continue
            c = SVC(model=None)
            c.fit(x, y0)
            svc_models.append(c)
        self._svc_models = svc_models

    def fit(self, X, y, test_set=None):
        self.fit_svm(X, y)
        if isinstance(y[0], list):
            y = y[0]
        if isinstance(X[0], list):
            X = X[0]
        D = self.transform(X, y)
        if test_set is not None:
            test_set = self.transform(test_set)
        svc_models = self._svc_models
        if svc_models[0] is not None:
            self._le = svc_models[0].le
        else:
            self._le = LabelEncoder()
            self._le.fit(y)
        klass = self._le.transform(y)
        if self._probability_calibration:
            probability_calibration = Calibration
        else:
            probability_calibration = None
        _ = dict(n_jobs=self._n_jobs, seed=self._seed,
                 probability_calibration=probability_calibration)
        self._evodag_args.update(_)
        self._evodag_model = EvoDAGE(**self._evodag_args).fit(D, klass,
                                                              test_set=test_set)
        if self._logistic_regression is not None:
            self._logistic_regression.fit(self._evodag_model.raw_decision_function(D),
                                          klass)
        # elif self._probability_calibration:
        #     self.probability_calibration(X, klass)
        return self

    # def probability_calibration(self, X, y):
    #     from .calibration import EnsembleCalibration
    #     X = self.transform(X)
    #     df = self._evodag_model._decision_function_raw(X, cpu_cores=self._n_jobs)
    #     self._calibration_coef = EnsembleCalibration().fit(df, y)

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
