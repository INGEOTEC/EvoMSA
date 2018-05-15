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
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import importlib
from sklearn.linear_model import LogisticRegression
from .calibration import CalibrationLR
from .model import Identity, BaseTextModel, BaseClassifier
import numpy as np
import logging
from multiprocessing import Pool
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


class LabelEncoderWrapper(object):
    def __init__(self):
        self._m = {}

    def fit(self, y):
        try:
            n = [int(x) for x in y]
        except ValueError:
            return LabelEncoder().fit(y)
        self.classes_ = np.unique(n)
        self._m = {v: k for k, v in enumerate(self.classes_)}
        self._inv = {v: k for k, v in self._m.items()}
        return self

    def transform(self, y):
        return np.array([self._m[int(x)] for x in y])

    def inverse_transform(self, y):
        return np.array([self._inv[int(x)] for x in y])


def kfold_decision_function(args):
    cl, X, y, tr, ts, seed = args
    c = cl(random_state=seed)
    c.fit([X[x] for x in tr], [y[x] for x in tr])
    _ = c.decision_function([X[x] for x in ts])
    return ts, _


def transform(args):
    k, m, t, X = args
    x = [t[_] for _ in X]
    df = m.decision_function(x)
    d = [EvoMSA.tolist(_) for _ in df]
    return (k, d)


def vector_space(args):
    k, t, X = args
    return k, [t[_] for _ in X]


class EvoMSA(object):
    def __init__(self, b4msa_params=None, evodag_args=dict(fitness_function='macro-F1'),
                 b4msa_args=dict(), n_jobs=1, n_splits=5, seed=0, logistic_regression=False,
                 models=[['EvoMSA.model.B4MSATextModel', 'EvoMSA.model.B4MSAClassifier']],
                 evodag_class="EvoDAG.model.EvoDAGE", logistic_regression_args=None, probability_calibration=False):
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
        self._exogenous_model = None
        self._probability_calibration = probability_calibration
        self.models = models
        self._evodag_class = self.get_class(evodag_class)

    def get_class(self, m):
        if isinstance(m, str):
            a = m.split('.')
            p = importlib.import_module('.'.join(a[:-1]))
            return getattr(p, a[-1])
        return m

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, models):
        if models is None:
            return
        if not isinstance(models, list):
            models = [models]
        self._models = []
        for m in models:
            if isinstance(m, list):
                textmodel, classifier = m
                tm = self.get_class(textmodel)
                cl = self.get_class(classifier)
            else:
                tm = Identity
                cl = self.get_class(m)
            assert issubclass(tm, BaseTextModel)
            assert issubclass(cl, BaseClassifier)
            self._models.append([tm, cl])

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, v):
        self._n_jobs = v

    def predict(self, X):
        pr = self.predict_proba(X)
        return self._le.inverse_transform(pr.argmax(axis=1))

    def predict_proba(self, X):
        X = self.transform(X)
        if self._logistic_regression is not None:
            X = self._evodag_model.raw_decision_function(X)
            return self._logistic_regression.predict_proba(X)
        try:
            return self._evodag_model.predict_proba(X)
        except AttributeError:
            index = self._evodag_model.predict(X)
            res = np.zeros((index.shape[0], self._le.classes_.shape[0]))
            res[np.arange(index.shape[0]), index] = 1
            return res

    def raw_decision_function(self, X):
        X = self.transform(X)
        return self._evodag_model.raw_decision_function(X)

    def decision_function(self, X):
        X = self.transform(X)
        return self._evodag_model.decision_function(X)

    @property
    def exogenous_model(self):
        return self._exogenous_model

    @exogenous_model.setter
    def exogenous_model(self, v):
        if isinstance(v, list):
            for x in v:
                x.n_jobs = self.n_jobs
        else:
            v.n_jobs = self.n_jobs
        self._exogenous_model = v

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

    def append_exogenous_model(self, D, X):
        if self.exogenous_model is None:
            return D
        ex = self.exogenous_model
        if not isinstance(ex, list):
            ex = [ex]
        L = [D]
        for x in ex:
            _ = x.predict_proba(X)
            df = _ * 2 - 1
            df[df > 1] = 1
            df[df < -1] = -1
            L.append(df)
        return np.concatenate(L, axis=1)

    def model(self, X):
        if not isinstance(X[0], list):
            X = [X]
        m = []
        kwargs = self._b4msa_args
        self._logger.info("Starting TextModel")
        self._logger.info(str(kwargs))
        for x in X:
            for tm, cl in self.models:
                m.append(tm(x, **kwargs))
        self._textModel = m

    def vector_space(self, X):
        if not isinstance(X[0], list):
            X = [X]
        args = []
        i = 0
        k = 0
        nmodels = len(self.models)
        for x in X:
            for _ in range(nmodels):
                t = self._textModel[k]
                k += 1
                args.append((i, t, x))
                i += 1
        if self.n_jobs > 1:
            p = Pool(self.n_jobs, maxtasksperchild=1)
            res = [x for x in tqdm(p.imap_unordered(vector_space, args), total=len(args))]
            res.sort(key=lambda x: x[0])
            p.close()
        else:
            res = [vector_space(x) for x in tqdm(args)]
        return [x[1] for x in res]

    def kfold_decision_function(self, cl, X, y):
        hy = [None for x in y]
        args = []
        for tr, ts in KFold(n_splits=self._n_splits,
                            shuffle=True, random_state=self._seed).split(X):
            args.append([cl, X, y, tr, ts, self._seed])
        if self.n_jobs == 1:
            res = [kfold_decision_function(x) for x in tqdm(args, total=len(args))]
        else:
            p = Pool(self.n_jobs, maxtasksperchild=1)
            res = [x for x in tqdm(p.imap_unordered(kfold_decision_function, args),
                                   total=len(args))]
            p.close()
        for ts, df in res:
            [hy.__setitem__(k, self.tolist(v)) for k, v in zip(ts, df)]
        return hy
    
    def _transform(self, X, models, textModel):
        if len(models) == 0:
            return []
        args = [[i_m[0], i_m[1], t, X] for i_m, t in zip(enumerate(models), textModel) if i_m[1] is not None]
        if self.n_jobs > 1:
            p = Pool(self.n_jobs, maxtasksperchild=1)
            res = [x for x in tqdm(p.imap_unordered(transform, args), total=len(args))]
            res.sort(key=lambda x: x[0])
            p.close()
        else:
            res = [transform(x) for x in tqdm(args)]
        res = [x[1] for x in res]
        D = res[0]
        [[v.__iadd__(w) for v, w in zip(D, d)] for d in res[1:]]
        return D

    def transform(self, X, y=None):
        if y is None or self._svc_models[0] is None:
            D = self._transform(X, self._svc_models, self._textModel)
        else:
            cnt = len(self.models)
            D = self._transform(X, self._svc_models[cnt:], self._textModel[cnt:])
            Di = None
            for t_cl, t in zip(self.models, self._textModel):
                cl = t_cl[1]
                x = [t[_] for _ in X]
                d = self.kfold_decision_function(cl, x, y)
                if Di is None:
                    Di = d
                else:
                    [v.__iadd__(w) for v, w in zip(Di, d)]
            [v.__iadd__(w) for v, w in zip(Di, D)]
            D = Di
        _ = np.array(D)
        return self.append_exogenous_model(self.append_exogenous(_), X)

    def fit_svm(self, X, y):
        self.model(X)
        Xvs = self.vector_space(X)
        if not isinstance(y[0], list):
            y = [y]
        svc_models = []
        k = 0
        nmodels = len(self.models)
        for y0 in y:
            for j in range(nmodels):
                x = Xvs[k]
                cl = self.models[j][1]
                k += 1
                c = cl(random_state=self._seed)
                c.fit(x, y0)
                svc_models.append(c)
        self._svc_models = svc_models

    def fit(self, X, y, test_set=None):
        if isinstance(y[0], list):
            le = []
            Y = []
            for y0 in y:
                _ = LabelEncoderWrapper().fit(y0)
                le.append(_)
                Y.append(_.transform(y0).tolist())
            self._le = le[0]
            y = Y
        else:
            self._le = LabelEncoderWrapper().fit(y)
            y = self._le.transform(y).tolist()
        self.fit_svm(X, y)
        if isinstance(y[0], list):
            y = y[0]
        if isinstance(X[0], list):
            X = X[0]
        D = self.transform(X, y)
        if test_set is not None:
            if isinstance(test_set, list):
                test_set = self.transform(test_set)
        if self._probability_calibration:
            probability_calibration = CalibrationLR
        else:
            probability_calibration = None
        _ = dict(n_jobs=self.n_jobs, seed=self._seed,
                 probability_calibration=probability_calibration)
        self._evodag_args.update(_)
        y = np.array(y)
        try:
            _ = self._evodag_class(**self._evodag_args).fit(D, y, test_set=test_set)
            self._evodag_model = _
        except TypeError:
            self._evodag_model = self._evodag_class().fit(D, y)
        if self._logistic_regression is not None:
            self._logistic_regression.fit(self._evodag_model.raw_decision_function(D), y)
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
