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
from multiprocessing import cpu_count
import importlib
import numpy as np
import logging
from multiprocessing import Pool
from b4msa.command_line import load_json
from microtc.textmodel import TextModel
from b4msa.lang_dependency import get_lang
from sklearn.model_selection import KFold
from .model import Identity, BaseTextModel, EvoMSAWrapper
from .utils import LabelEncoderWrapper, download
from microtc.utils import load_model
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
        x = t.transform(X)
    except AttributeError:
        x = t.tonp([t[_] for _ in X])
    df = m.decision_function(x)
    d = [EvoMSA.tolist(_) for _ in df]
    return (k, d)


def vector_space(args):
    k, t, X = args
    try:
        res = t.transform(X)
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
    >>> from microtc.utils import tweet_iterator
    >>> import os
    >>> tweets = os.path.join(os.path.dirname(base.__file__), 'tests', 'tweets.json')
    >>> D = [[x['text'], x['klass']] for x in tweet_iterator(tweets)]

    Once the dataset is loaded, it is time to create an EvoMSA model

    >>> from EvoMSA.base import EvoMSA
    >>> evo = EvoMSA().fit([x[0] for x in D], [x[1] for x in D])

    Predict a sentence in Spanish

    >>> evo.predict(['EvoMSA esta funcionando'])

    :param b4msa_args: Arguments pass to TextModel updating the default arguments
    :type b4msa_args:  dict
    :param evodag_args: Arguments pass to EvoDAG
    :type evodag_args: dict
    :param n_jobs: Multiprocessing default 1 process, <= 0 to use all processors
    :type n_jobs: int
    :param n_splits: Number of folds to train EvoDAG or evodag_class
    :type n_splits: int
    :param seed: Seed used default 0
    :type seed: int
    :param classifier: EvoMSA as classifier default True
    :type classifier: bool
    :param models: Models used as list of pairs (see flags: TR, TH and Emo)
    :type models: list
    :param evodag_class: Classifier or regressor used to ensemble the outputs of :attr:`models` default :class:`EvoDAG.model.EvoDAGE`
    :type evodag_class: str or class
    :param TR: Use b4msa.textmodel.TextModel, sklearn.svm.LinearSVC on the training set
    :type TR: bool
    :param Emo: Use EvoMSA.model.EmoSpace[Ar|En|Es], sklearn.svm.LinearSVC
    :type Emo: bool
    :param TH: Use EvoMSA.model.ThumbsUpDown[Ar|En|Es], sklearn.svm.LinearSVC
    :type TH: bool
    :param HA: Use HA datasets, sklearn.svm.LinearSVC
    :type HA: bool
    :param tm_n_jobs: Multiprocessing using on the Text Models, <= 0 to use all processors
    :type tm_n_jobs: int
    """

    def __init__(self, b4msa_args=dict(), evodag_args=dict(), n_jobs=1,
                 n_splits=5, seed=0, classifier=True, models=None, lang=None,
                 evodag_class="EvoDAG.model.EvoDAGE", TR=True, Emo=False, TH=False, HA=False,
                 tm_n_jobs=None):
        if models is None:
            models = []
        if TR:
            models.insert(0, ["b4msa.textmodel.TextModel", "sklearn.svm.LinearSVC"])
        lang = lang if lang is None else get_lang(lang)
        b4msa_args['lang'] = lang
        if Emo:
            models.append(["EvoMSA.model.%s" % self._emoSpace(lang),
                           "sklearn.svm.LinearSVC"])
        if TH:
            models.append(["EvoMSA.model.%s" % self._thumbsUpDown(lang),
                           "sklearn.svm.LinearSVC"])
        if HA:
            fname = download("%s.evoha" % lang)
            models.append([fname, "sklearn.svm.LinearSVC"])
        self._b4msa_args = b4msa_args
        self._evodag_args = evodag_args
        if classifier:
            _ = DEFAULT_CL.copy()
        else:
            _ = DEFAULT_R.copy()
        _.update(self._evodag_args)
        self._evodag_args = _
        
        self._n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        self._tm_n_jobs = tm_n_jobs if tm_n_jobs is None or tm_n_jobs > 0 else cpu_count()
        self._n_splits = n_splits
        self._seed = seed
        self._svc_models = None
        self._evodag_model = None
        self._logger = logging.getLogger('EvoMSA')
        self._le = None
        self._classifier = classifier
        self.models = models
        self._evodag_class = self.get_class(evodag_class)

    def _emoSpace(self, lang):
        m = None
        if lang == 'spanish':
            m = "EmoSpaceEs"
        elif lang == 'english':
            m = "EmoSpaceEn"
        elif lang == 'arabic':
            m = "EmoSpaceAr"
        assert m is not None
        return m

    def _thumbsUpDown(self, lang):
        m = None
        if lang == 'spanish':
            m = "ThumbsUpDownEs"
        elif lang == 'english':
            m = "ThumbsUpDownEn"
        elif lang == 'arabic':
            m = "ThumbsUpDownAr"
        assert m is not None
        return m

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
        _ = dict(n_jobs=self.n_jobs, seed=self._seed)
        self._evodag_args.update(_)
        y = np.array(y)
        try:
            _ = self._evodag_class(**self._evodag_args)
            _.fit(D, y, test_set=test_set)
            self._evodag_model = _
        except TypeError:
            self._evodag_model = self._evodag_class().fit(D, y)
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
            assert isinstance(tm, str) or (hasattr(tm, 'transform') and hasattr(tm, 'fit'))
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

    @property
    def tm_n_jobs(self):
        return self._tm_n_jobs

    @tm_n_jobs.setter
    def tm_n_jobs(self, v):
        self._tm_n_jobs = v

    def predict(self, X):
        if self.classifier:
            pr = self.predict_proba(X)
            return self._le.inverse_transform(pr.argmax(axis=1))
        return self.decision_function(X)

    def predict_proba(self, X):
        X = self.transform(X)
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
                    _ = load_model(tm)
                    if isinstance(_, EvoMSA):
                        _ = EvoMSAWrapper(evomsa=_)
                    m.append(_)
                elif isinstance(tm, type):
                    m.append(tm(**kwargs).fit(x))
                else:
                    m.append(tm)
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
        n_jobs = self.n_jobs if self.tm_n_jobs is None else self.tm_n_jobs
        if n_jobs > 1:
            p = Pool(self.n_jobs, maxtasksperchild=1)
            res = [x for x in tqdm(p.imap_unordered(vector_space, args), total=len(args))]
            res.sort(key=lambda x: x[0])
            p.close()
        else:
            res = [vector_space(x) for x in tqdm(args)]
        return [x[1] for x in res]

    def sklearn_kfold(self, cl, X, y):
        args = []
        klasses = np.unique(y)
        nclass = klasses.shape[0]
        for tr, ts in KFold(n_splits=self._n_splits,
                            shuffle=True, random_state=self._seed).split(X):
            tr_klasses = np.unique([y[i] for i in tr])
            if tr_klasses.shape[0] != nclass:
                for k in klasses:
                    if k not in tr_klasses:
                        candidate = [(i, x) for i, x in enumerate(ts) if y[x] == k][0]
                        tr = tr.tolist()
                        tr.append(candidate[1])
                        tr = np.array(tr)
            args.append([cl, X, y, tr, ts, self._seed])
        return args

    def kfold_decision_function(self, cl, X, y):
        hy = [None for x in y]
        args = self.sklearn_kfold(cl, X, y)
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
        n_jobs = self.n_jobs if self.tm_n_jobs is None else self.tm_n_jobs
        if n_jobs > 1:
            p = Pool(n_jobs, maxtasksperchild=1)
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
                    x = t.transform(X)
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
