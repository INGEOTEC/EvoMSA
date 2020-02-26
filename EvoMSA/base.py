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
from b4msa.lang_dependency import get_lang
from sklearn.model_selection import KFold
from .model import Identity, EvoMSAWrapper
from .utils import LabelEncoderWrapper, download
from microtc.utils import load_model, save_model
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
    k, cl, X = args
    df = cl.decision_function(X)
    d = [EvoMSA.tolist(_) for _ in df]
    return (k, d)


def vector_space(args):
    k, t, X, output = args
    if output is not None and os.path.isfile(output):
        return k, load_model(output)
    try:
        res = t.transform(X)
    except AttributeError:
        res = t.tonp([t[_] for _ in X])
    if output is not None:
        save_model(res, output)
    return k, res


DEFAULT_CL = dict(fitness_function='macro-F1',
                  random_generations=1000,
                  n_jobs=cpu_count(), seed=0,
                  orthogonal_selection=True)


DEFAULT_R = dict(random_generations=1000,
                 classifier=False,
                 n_jobs=cpu_count(), seed=0,
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
    >>> D = list(tweet_iterator(tweets))
    >>> X = [x['text'] for x in D]
    >>> y = [x['klass'] for x in D]

    Once the dataset is loaded, it is time to create an EvoMSA model

    >>> from EvoMSA.base import EvoMSA
    >>> evo = EvoMSA().fit(X, y)

    Predict a sentence in Spanish

    >>> evo.predict(['EvoMSA esta funcionando'])
    array(['P'], dtype='<U4')

    :param b4msa_args: Arguments pass to TextModel updating the default arguments
    :type b4msa_args:  dict
    :param stacked_method_args: Arguments pass to the stacked method
    :type stacked_method_args: dict
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
    :param stacked_method: Classifier or regressor used to ensemble the outputs of :attr:`models` default :class:`EvoDAG.model.EvoDAGE`
    :type stacked_method: str or class
    :param TR: Use b4msa.textmodel.TextModel, sklearn.svm.LinearSVC on the training set
    :type TR: bool
    :param Emo: Use EvoMSA.model.EmoSpace[Ar|En|Es], sklearn.svm.LinearSVC
    :type Emo: bool
    :param TH: Use EvoMSA.model.ThumbsUpDown[Ar|En|Es], sklearn.svm.LinearSVC
    :type TH: bool
    :param HA: Use HA datasets, sklearn.svm.LinearSVC
    :type HA: bool
    :param B4MSA: Pre-trained text model
    :type B4MSA:
    :param tm_n_jobs: Multiprocessing using on the Text Models, <= 0 to use all processors
    :type tm_n_jobs: int
    :param cache: Store the output of text models
    :type cache: str
    """

    def __init__(self, b4msa_args=dict(),
                 stacked_method="EvoDAG.model.EvoDAGE",
                 stacked_method_args=dict(),
                 n_jobs=1, n_splits=5, seed=0,
                 classifier=True, models=None, lang=None,
                 TR=True, Emo=False, TH=False, HA=False,
                 B4MSA=False, Aggress=False,
                 tm_n_jobs=None, cache=None):
        if models is None:
            models = []
        if TR:
            models.insert(0, ["b4msa.textmodel.TextModel",
                              "sklearn.svm.LinearSVC"])
        if lang is not None:
            assert len(lang) == 2
            lang = lang.lower()
            lang = "%s%s" % (lang[0].upper(), lang[1])
            b4msa_args['lang'] = get_lang(lang)
        if Emo or TH or HA or B4MSA:
            assert lang is not None and lang in ["Ar", "En", "Es"]
        if Emo:
            models.append([download("emo_%s.tm" % lang),
                           "sklearn.svm.LinearSVC"])
        if TH:
            models.append(["EvoMSA.model.ThumbsUpDown%s" % lang,
                           "sklearn.svm.LinearSVC"])
        if HA:
            models.append([download("ha_%s.tm" % lang),
                           "sklearn.svm.LinearSVC"])
        if B4MSA:
            models.append([download("b4msa_%s.tm" % lang),
                           "sklearn.svm.LinearSVC"])
        if Aggress:
            models.append(["EvoMSA.model.Aggressiveness%s" % lang,
                           "sklearn.svm.LinearSVC"])            
        self._b4msa_args = b4msa_args
        self._evodag_args = stacked_method_args
        _ = dict()
        if stacked_method == "EvoDAG.model.EvoDAGE":
            if classifier:
                _ = DEFAULT_CL.copy()
            else:
                _ = DEFAULT_R.copy()
        _.update(self._evodag_args)
        self._evodag_args = _
        self._n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        _ = tm_n_jobs
        self._tm_n_jobs = _ if _ is None or _ > 0 else cpu_count()
        self._n_splits = n_splits
        self._seed = seed
        self._svc_models = None
        self._evodag_model = None
        self._logger = logging.getLogger('EvoMSA')
        self._le = None
        self._classifier = classifier
        self.cache = cache
        self.models = models
        self._evodag_class = self.get_class(stacked_method)

    def first_stage(self, X, y):
        """Training EvoMSA's first stage

        :param X: Independent variables
        :type X: dict or list
        :param y: Dependent variable.
        :type y: list
        :return: List of vector spaces, i.e., second-stage's training set
        :rtype: list

        >>> import os
        >>> from EvoMSA import base
        >>> from microtc.utils import tweet_iterator
        >>> TWEETS = os.path.join(os.path.dirname(__file__), 'tests', 'tweets.json')
        >>> X = [x['text'] for x in tweet_iterator(TWEETS)]
        >>> y = [x['klass'] for x in tweet_iterator(TWEETS)]
        >>> evo = base.EvoMSA()
        >>> D = evo.first_stage(X, y)
        >>> D.shape
        (1000, 4)

        """

        # Instantiate Text Models
        self.model(X)
        # Transform text into a vector space - List of vector spaces
        X_vector_space = self.vector_space(X)
        # Train supervised learning algorithms
        self.fit_svm(X_vector_space, y)
        # KFold to train the stacked_method
        D = self.kfold_supervised_learning(X_vector_space, y)
        return D

    def fit(self, X, y, test_set=None):
        """
        Train the model using a training set or pairs: text,
        dependent variable (e.g., class) EvoMSA is a two-stage procedure;
        the first step is to transform the text into a vector space with
        dimensions related to the number of classes and then
        train a supervised learning algorithm.

        :param X: Independent variables
        :type X: dict or list
        :param y: Dependent variable.
        :type y: list
        :return: EvoMSA instance, i.e., self
        """

        self._le = LabelEncoderWrapper(classifier=self.classifier).fit(y)
        y = self._le.transform(y)
        # Training first stage
        D = self.first_stage(X, y)
        # After the first stage the cache is not needed
        self.cache = None
        # Transform test set to do transductive learning
        if test_set is not None:
            if isinstance(test_set, list):
                test_set = self.transform(test_set)
        # Training stacked_method
        # Start of the second stage
        _ = self._evodag_class(**self._evodag_args)
        if test_set is not None:
            _.fit(D, y, test_set=test_set)
        else:
            _.fit(D, y)
        self._evodag_model = _
        return self

    @property
    def stacked_method(self):
        """Method's instance used to ensemble the output of the first stage."""

        return self._evodag_model

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
            assert isinstance(tm, str) or (hasattr(tm, 'transform')
                                           and hasattr(tm, 'fit'))
            # Initializing the cache
            if self.cache is not None:
                self.cache.append(tm, ml=cl)
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

    @property
    def textModels(self):
        """Text Models

        :rtype: list
        """

        # Performing lazy loading
        # If the outputs are in the cache,
        # there is no need to load the model into memory
        solve = [(i, tm) for (i, tm), cache in
                 zip(enumerate(self._textModel), self.cache) if
                 isinstance(tm, str) and (cache is None or not
                                          os.path.isfile(cache))]
        for i, tm in solve:
            _ = load_model(tm)
            if isinstance(_, EvoMSA):
                _ = EvoMSAWrapper(evomsa=_)
            self._textModel[i] = _
        return self._textModel

    @property
    def cache(self):
        """Basename to store the output of the textmodels"""

        return self._cache

    @cache.setter
    def cache(self, value):
        from .utils import Cache
        self._cache = Cache(value)

    def predict(self, X, cache=None):
        """
        Predict the output of input X

        :param X: List of strings
        :type X: list
        :param cache: Basename to store the output of the text models.
        :type cache: str
        """

        if cache is not None:
            self.cache = cache
            [self.cache.append(tm) for tm, _ in self.models]
        if self.classifier:
            pr = self.predict_proba(X)
            output = self._le.inverse_transform(pr.argmax(axis=1))
        else:
            output = self.decision_function(X)
        self.cache = None
        return output

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
        m = []
        kwargs = self._b4msa_args
        self._logger.info("Starting TextModel")
        self._logger.info(str(kwargs))
        for tm, cl in self.models:
            if isinstance(tm, str):
                # Performing lazy loading
                m.append(tm)
            elif isinstance(tm, type):
                m.append(tm(**kwargs).fit(X))
            else:
                m.append(tm)
        self._textModel = m

    def vector_space(self, X):
        args = [(i, t, X, output) for (i, t), output in
                zip(enumerate(self.textModels), self.cache)]
        n_jobs = self.n_jobs if self.tm_n_jobs is None else self.tm_n_jobs
        if n_jobs > 1:
            p = Pool(self.n_jobs, maxtasksperchild=1)
            res = [x for x in tqdm(p.imap_unordered(vector_space, args),
                                   total=len(args))]
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
            res = [kfold_decision_function(x) for x in tqdm(args,
                                                            total=len(args))]
        else:
            p = Pool(self.n_jobs, maxtasksperchild=1)
            res = [x for x in tqdm(p.imap_unordered(kfold_decision_function,
                                                    args),
                                   total=len(args))]
            p.close()
        for ts, df in res:
            [hy.__setitem__(k, self.tolist(v)) for k, v in zip(ts, df)]
        return hy

    def kfold_supervised_learning(self, X_vector_space, y):
        """KFold to train the stacked_method, i.e., training set

        :rtype: np.array
        """

        D = None
        for (_, cl), Xvs, output in zip(self.models, X_vector_space,
                                        self.cache.ml_kfold()):
            if output is not None and os.path.isfile(output):
                d = load_model(output)
            else:
                d = self.kfold_decision_function(cl, Xvs, y)
                if output is not None:
                    save_model(d, output)
            if D is None:
                D = d
            else:
                [v.__iadd__(w) for v, w in zip(D, d)]
        D = np.array(D)
        D[~np.isfinite(D)] = 0
        return D

    def transform(self, X):
        Xvs = self.vector_space(X)
        args = [(i, cl, X) for (i, cl), X in zip(enumerate(self._svc_models),
                                                 Xvs)]
        n_jobs = self.n_jobs if self.tm_n_jobs is None else self.tm_n_jobs
        if n_jobs > 1:
            p = Pool(n_jobs, maxtasksperchild=1)
            res = [x for x in tqdm(p.imap_unordered(transform, args),
                                   total=len(args))]
            res.sort(key=lambda x: x[0])
            p.close()
        else:
            res = [transform(x) for x in tqdm(args)]
        res = [x[1] for x in res]
        D = res[0]
        [[v.__iadd__(w) for v, w in zip(D, d)] for d in res[1:]]
        _ = np.array(D)
        _[~np.isfinite(_)] = 0
        return _

    def fit_svm(self, Xvs, y):
        svc_models = []
        for (_, cl), X, output in zip(self.models, Xvs, self.cache.ml_train()):
            if output is not None and os.path.isfile(output):
                svc_models.append(load_model(output))
                continue
            try:
                c = cl(random_state=self._seed)
            except TypeError:
                c = cl()
            c.fit(X, y)
            svc_models.append(c)
            if output is not None:
                save_model(c, output)
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
