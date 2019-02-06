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
import importlib
import numpy as np
import logging
from multiprocessing import Pool
from b4msa.command_line import load_json
from b4msa.textmodel import TextModel
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from .calibration import CalibrationLR
from .model import Identity, BaseTextModel
from .utils import LabelEncoderWrapper
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def kfold_decision_function(args):
    cl, X, y, tr, ts, seed = args
    try:
        c = cl(random_state=seed)
    except TypeError:
        c = cl()
    if isinstance(X, (list, tuple)):
        c.fit([X[x] for x in tr], [y[x] for x in tr])
        _ = c.decision_function([X[x] for x in ts])
    else:
        y = np.array(y)
        c.fit(X[tr], y[tr])
        _ = c.decision_function(X[ts])
    return ts, _


def transform(args):
    k, m, t, X = args
    try:
        x = t.tonp(t.transform(X))
    except AttributeError:
        x = t.tonp([t[_] for _ in X])
    df = m.decision_function(x)
    d = [EvoMSA.tolist(_) for _ in df]
    return (k, d)


def vector_space(args):
    k, t, X = args
    try:
        res = t.tonp(t.transform(X))
    except AttributeError:
        res = t.tonp([t[_] for _ in X])
    return k, res


DEFAULT_CL = dict(fitness_function='macro-F1',
                  random_generations=1000,
                  orthogonal_selection=True)


DEFAULT_R = dict(random_generations=1000,
                 classifier=False,
                 orthogonal_selection=True)


class EvoMSA(object):
    """
    This is the main entry to create an EvoMSA model

    Let us start with an example to show how to create an EvoMSA model.
    The first thing would be to read the dataset,
    EvoMSA has a dummy dataset to test its functionality, so lets used it.

    Read the dataset

    >>> from EvoMSA import base
    >>> from b4msa.utils import tweet_iterator
    >>> import os
    >>> tweets = os.path.join(os.path.dirname(base.__file__), 'tests', 'tweets.json')
    >>> D = [[x['text'], x['klass']] for x in tweet_iterator(tweets)]

    Once the dataset is loaded, it is time to create an EvoMSA model

    >>> from EvoMSA.base import EvoMSA
    >>> evo = EvoMSA().fit([x[0] for x in D], [x[1] for x in D])

    Predict a sentence in Spanish

    >>> evo.predict(['EvoMSA esta funcionando'])

    :param b4msa_params: File in json containing TextModel's arguments, i.e., B4MSATextModel
    :type b4msa_params: str
    :param b4msa_args: Arguments pass to TextModel updating the default arguments
    :type b4msa_args:  dict
    :param evodag_args: Arguments pass to EvoDAG
    :type evodag_args: dict
    :param n_jobs: Multiprocessing default 1 process
    :type n_jobs: int
    :param n_splits: Number of folds to train EvoDAG or evodag_class
    :type n_splits: int
    :param seed: Seed used default 0
    :type seed: int
    :param classifier: EvoMSA as classifier default True
    :type classifier: bool
    :param models: Models used as list of pairs
    :type models: list
    :param evodag_class: Classifier or regressor used to ensemble the outputs of :attr:`models` default :class:`EvoDAG.model.EvoDAGE`
    :type evodag_class: str or class
    :param logistic_regression: Use Logistic Regression as final output defautl False
    :type logistic_regression: bool
    :param logistic_regression_args: Parameters pass to the Logistic Regression
    :type logistic_regression_args: dict
    :param probability_calibration: Use a probability calibration algorithm default False
    :type probability_calibration: bool
    """

    def __init__(self, b4msa_params=None, b4msa_args=dict(),
                 evodag_args=dict(), n_jobs=1, n_splits=5, seed=0,
                 classifier=True,
                 models=[["b4msa.textmodel.TextModel", "sklearn.svm.LinearSVC"]],
                 evodag_class="EvoDAG.model.EvoDAGE",
                 logistic_regression=False, logistic_regression_args=None,
                 probability_calibration=False):
        if b4msa_params is None:
            b4msa_params = os.path.join(os.path.dirname(__file__),
                                        'conf', 'default_parameters.json')
        b4msa_params = self.read_json(b4msa_params)
        b4msa_params.update(b4msa_args)
        self._b4msa_args = b4msa_params
        self._evodag_args = evodag_args
        if classifier:
            _ = DEFAULT_CL.copy()
        else:
            _ = DEFAULT_R.copy()
        _.update(self._evodag_args)
        self._evodag_args = _
        self._n_jobs = n_jobs
        self._n_splits = n_splits
        self._seed = seed
        self._svc_models = None
        self._evodag_model = None
        self._logger = logging.getLogger('EvoMSA')
        self._le = None
        self._logistic_regression = None
        self._classifier = classifier
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

    def fit(self, X, y, test_set=None):
        """
        Train the model using a training set or pairs: text, dependent variable (e.g. class)

        :param X: Independent variables
        :type X: dict or list
        :param y: Dependent variable.
        :type y: list
        :return: EvoMSA instance, i.e., self
        """

        if isinstance(y[0], list):
            le = []
            Y = []
            for y0 in y:
                _ = LabelEncoderWrapper(classifier=self.classifier).fit(y0)
                le.append(_)
                Y.append(_.transform(y0).tolist())
            self._le = le[0]
            y = Y
        else:
            self._le = LabelEncoderWrapper(classifier=self.classifier).fit(y)
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
            _ = self._evodag_class(**self._evodag_args)
            _.fit(D, y, test_set=test_set)
            self._evodag_model = _
        except TypeError:
            self._evodag_model = self._evodag_class().fit(D, y)
        if self._logistic_regression is not None:
            self._logistic_regression.fit(self._evodag_model.raw_decision_function(D), y)
        return self

    @property
    def classifier(self):
        """Whether EvoMSA is acting as classifier"""

        return self._classifier

    def get_class(self, m):
        if isinstance(m, str):
            if os.path.isfile(m):
                return m
            a = m.split('.')
            p = importlib.import_module('.'.join(a[:-1]))
            return getattr(p, a[-1])
        return m

    @property
    def models(self):
        """Models used as list of pairs

        :rtype: list
        """

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
            assert isinstance(tm, str) or issubclass(tm, BaseTextModel) or issubclass(tm, TextModel)
            # assert issubclass(cl, BaseClassifier)
            self._models.append([tm, cl])

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, v):
        self._n_jobs = v
        try:
            self._evodag_model._m._n_jobs = v
        except AttributeError:
            pass

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
            L.append(_)
        _ = np.concatenate(L, axis=1)
        return _

    @staticmethod
    def load_model(fname):
        """Read model from file. The model must be stored using gzip and pickle

        :param fname: filename
        :type fname: str (path)
        """
        import gzip
        import pickle
        with gzip.open(fname, 'r') as fpt:
            return pickle.load(fpt)

    def model(self, X):
        if not isinstance(X[0], list):
            X = [X]
        m = []
        kwargs = self._b4msa_args
        self._logger.info("Starting TextModel")
        self._logger.info(str(kwargs))
        for x in X:
            for tm, cl in self.models:
                if isinstance(tm, str):
                    m.append(self.load_model(tm))
                else:
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
                try:
                    x = t.tonp(t.transform(X))
                except AttributeError:
                    x = t.tonp([t[_] for _ in X])
                d = self.kfold_decision_function(cl, x, y)
                if Di is None:
                    Di = d
                else:
                    [v.__iadd__(w) for v, w in zip(Di, d)]
            [v.__iadd__(w) for v, w in zip(Di, D)]
            D = Di
        _ = np.array(D)
        _ = self.append_exogenous_model(self.append_exogenous(_), X)
        _[~np.isfinite(_)] = 0
        return _

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
                try:
                    c = cl(random_state=self._seed)
                except TypeError:
                    c = cl()
                c.fit(x, y0)
                svc_models.append(c)
        self._svc_models = svc_models

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

    def __getstate__(self):
        """Remove attributes unable to pickle"""

        r = self.__dict__.copy()
        try:
            del r['_logger']
        except KeyError:
            pass
        return r
