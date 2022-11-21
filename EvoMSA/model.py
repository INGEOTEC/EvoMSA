# Copyright 2018 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union
from microtc.utils import save_model, tweet_iterator
from b4msa.textmodel import TextModel
from EvoMSA.ConceptModelling.thumbs_up_down import ThumbsUpDown, _ARABIC, _ENGLISH, _SPANISH, PATH as ConPATH
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.svm import LinearSVC
import numpy as np
import os


class BaseTextModel(object):
    """Base class for text model

    :param corpus: Text to build the text model
    :type corpus: list or dict
    """

    def __init__(self, corpus=None, **kwargs):
        pass

    def fit(self, X):
        """
        Train the model

        :param X: Corpus
        :type X: list
        :rtype: instance
        """

        pass

    def __getitem__(self, x):
        pass

    def tokenize(self, text):
        pass

    def transform(self, X):
        return np.array([self.__getitem__(x) for x in X])


class BaseClassifier(object):
    """Base class for the classifier"""

    def __init__(self, random_state=0):
        pass

    def fit(self, X, y):

        """Method to train the classifier
        :param X: Independent variable
        :type X: np.array or csc_matrix
        :param y: Dependent variable
        :type y: np.array
        :rtype: self
        """

        return self

    def decision_function(self, X):
        """Classifier's decision function

        :param X: Independent variable
        :type X: np.array or csc_matrix
        :rtype: np.array
        """

        pass


class OutputClassifier(object):
    """LinearSVC that outputs the training set and test set using the environment varible OUTPUT"""

    def __init__(self, random_state=0, output=None):
        self._output = os.getenv('OUTPUT', output)
        assert self._output is not None
        self._random_state = random_state

    def fit(self, X, y):
        self.m = LinearSVC(random_state=self._random_state).fit(X, y)
        try:
            X = np.array(X.todense())
        except AttributeError:
            pass
        with open('%s_train.csv' % self._output, 'w') as fpt:
            for x, _y in zip(X, y):
                fpt.write(",".join([str(_) for _ in x]))
                fpt.write(",%s\n" % str(_y))
        return self

    def decision_function(self, X):
        hy = self.m.decision_function(X)
        try:
            X = np.array(X.todense())
        except AttributeError:
            pass
        with open('%s_test.csv' % self._output, 'w') as fpt:
            for x in X:
                fpt.write(",".join([str(_) for _ in x]))
                fpt.write("\n")
        return hy


class EvoMSAWrapper(BaseTextModel):
    def __init__(self, evomsa=None):
        assert evomsa is not None
        evomsa.n_jobs = 1
        self._evomsa = evomsa

    def transform(self, X):
        return self._evomsa.predict_proba(X)


class SVCWrapper(LinearSVC):
    """Wrapper to the LinearSVC class
    >>> from EvoMSA.model import SVCWrapper
    >>> s = SVCWrapper()
    >>> s.dual
    False
    """
    def __init__(self, *args, dual=False, **kwargs):
        super(SVCWrapper, self).__init__(*args, dual=dual, **kwargs)


class Identity(BaseTextModel, BaseClassifier):
    """Identity function used as either text model or classifier or regressor"""

    def __getitem__(self, x):
        return x

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        try:
            return X.toarray()
        except AttributeError:
            return X

    def predict_proba(self, X):
        return self.decision_function(X)


class LabeledDataSet(BaseTextModel, BaseClassifier):
    """Create a text classifier using b4msa.textmodel.TextModel and LinearSVC

    :param docs: do not use
    :type docs: None
    :param textModel: text model e.g., b4msa.textmodel.TextModel
    :param coef: coefficients obtained from LinearSVC
    :type coef: array
    :param intercept: bias obtained from LinearSVC
    :type intercept: array
    :param labels: list of labels or classes
    :type labels: list
    """
    def __init__(self, docs=None, textModel=None, coef=None, intercept=None, labels=None):
        assert docs is None
        self._textModel = textModel
        self._coef = coef
        self._intercept = intercept
        self._labels = labels
        self._text = os.getenv('TEXT', default='text')

    @property
    def textModel(self):
        from .cython_utils import TextModelPredict
        try:
            return self._textModelPredict
        except AttributeError:
            self._textModelPredict = TextModelPredict(self._textModel,
                                                      self._coef,
                                                      self._intercept)
        return self._textModelPredict

    def __getstate__(self):
        """Remove attributes before the pickle"""

        r = self.__dict__.copy()
        try:
            del r['_textModelPredict']
        except KeyError:
            pass
        return r

    def fit(self, X, y=None):
        return self

    def get_text(self, text):
        key = self._text
        if isinstance(text, (list, tuple)):
            return " | ".join([x[key] for x in text])
        elif isinstance(text, str):
            return text
        return text[key]

    def decision_function(self, X):
        return self.transform(X)

    def predict_proba(self, X):
        return self.decision_function(X)

    def transform(self, X):
        output = []
        self.textModel.transform([self.get_text(x) for x in X], output)
        return np.array(output)

    def __getitem__(self, x):
        return np.array(self.textModel[self.get_text(x)])

    @classmethod
    def _create_space(cls, fname, **kwargs):
        """Create the space from a file of json

        :param fname: Path to the file containing the json
        :type fname: str
        :param kwargs: Keywords pass to TextModel
        """
        import random
        from .utils import linearSVC_array
        from collections import Counter
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **kwargs):
                return x

        data = [x for x in tweet_iterator(fname)]
        random.shuffle(data)
        tm = TextModel(**kwargs).fit([x['text'] for x in data[:128000]])
        tm._num_terms = tm.model.num_terms
        # klass, nele = np.unique([x['klass'] for x in data], return_counts=True)
        _ = [(k, v) for k, v in Counter([x['klass'] for x in data]).items()]
        _.sort(key=lambda x: x[0])
        klass = [x[0] for x in _]
        nele = [x[1] for x in _]
        h = {v: k for k, v in enumerate(klass)}
        MODELS = []
        for ident, k in tqdm(enumerate(klass)):
            elepklass = [0 for __ in klass]
            cnt = nele[ident]
            cntpklass = int(cnt / (len(klass) - 1))
            D = [(x, 1) for x in data if x['klass'] == k]
            for x in data:
                if x['klass'] == k:
                    continue
                if elepklass[h[x['klass']]] > cntpklass:
                    continue
                elepklass[h[x['klass']]] = elepklass[h[x['klass']]] + 1
                D.append((x, -1))
            m = LinearSVC().fit(tm.tonp([tm[x[0]['text']] for x in D]), [x[1] for x in D])
            MODELS.append(m)
        coef, intercept = linearSVC_array(MODELS)
        return tm, coef, intercept, klass

    @classmethod
    def create_space(cls, fname, output=None, **kwargs):
        """Create the space from a file of json

        :param fname: Path to the file containing the json
        :type fname: str
        :param output: Path to store the model, it is cls.model_fname if None
        :type output: str
        :param kwargs: Keywords pass to TextModel
        """
        tm, coef, intercept, klass = cls._create_space(fname, **kwargs)
        if output is None:
            output = cls.model_fname()
        ins = cls(textModel=tm, coef=coef, intercept=intercept, labels=klass)
        save_model(ins, output)


class Corpus(BaseTextModel):
    """Text model using only words"""

    def __init__(self, corpus=None, **kwargs):
        self._text = os.getenv('TEXT', default='text')
        self._m = {}
        self._num_terms = 0
        self._training = True
        self._textModel = TextModel([''], token_list=[-1])
        if corpus is not None:
            self.fit(corpus)

    def get_text(self, text):
        return text[self._text]

    def fit(self, c):
        [self.__getitem__(x) for x in c]
        self._training = False
        return self

    @property
    def num_terms(self):
        return self._num_terms

    def tokenize(self, text):
        if isinstance(text, dict):
            text = self.get_text(text)
        if isinstance(text, (list, tuple)):
            tokens = []
            for _text in text:
                tokens.extend(self._textModel.tokenize(_text))
            return tokens
        else:
            return self._textModel.tokenize(text)

    def transform(self, texts):
        """Convert test into a vector

        :param texts: List of text to be transformed
        :type text: list

        :rtype: list

        Example:

        >>> from microtc.textmodel import TextModel
        >>> corpus = ['buenos dias catedras', 'catedras conacyt']
        >>> textmodel = TextModel().fit(corpus)
        >>> X = textmodel.transform(corpus)
        """
        return self._textModel.tonp([self.__getitem__(x) for x in texts])

    def __getitem__(self, d):
        tokens = []
        for t in self.tokenize(d):
            try:
                index, k = self._m[t]
                if self._training:
                    self._m[t] = [index, k+1]
            except KeyError:
                if not self._training:
                    continue
                index, k = self._num_terms, 1
                self._m[t] = [index, k]
                self._num_terms += 1
            tokens.append([index, k])
        return tokens


class AggressivenessAr(Corpus):
    """Arabic text model using an aggressive corpus"""

    def __init__(self, *args, **kwargs):
        fname = os.path.join(os.path.dirname(__file__), 'conf',
                             'aggressiveness.ar')
        corpus = []
        for x in tweet_iterator(fname):
            corpus += x['words']
        super(AggressivenessAr, self).__init__(corpus)


class AggressivenessEn(Corpus):
    """English text model using an aggressive corpus"""

    def __init__(self, *args, **kwargs):
        fname = os.path.join(os.path.dirname(__file__), 'conf',
                             'aggressiveness.en')
        corpus = []
        for x in tweet_iterator(fname):
            corpus += x['words']
        super(AggressivenessEn, self).__init__(corpus)


class AggressivenessEs(Corpus):
    """Spanish text model using an aggressive corpus"""

    def __init__(self, *args, **kwargs):
        fname = os.path.join(os.path.dirname(__file__), 'conf',
                             'aggressiveness.es')
        corpus = []
        for x in tweet_iterator(fname):
            corpus += x['words']
        super(AggressivenessEs, self).__init__(corpus)


class Bernoulli(BaseClassifier):
    """Bernoulli classifier"""

    def __init__(self, random_state=0):
        self._num_terms = -1

    @property
    def num_terms(self):
        return self._num_terms

    def fit(self, X, klass):
        self._num_terms = X.shape[1]
        klasses, prior = np.unique(klass, return_counts=True)
        self._prior = np.log(prior / prior.sum())
        pr = np.zeros((klasses.shape[0], self.num_terms))
        for i, k in zip(X, klass):
            index = i.indices
            if index.shape[0] > 0:
                pr[k, index] += 1
        pr = np.array([(x + 1) / ((klass == i).sum() + 2)
                       for x, i in zip(pr, klasses)])
        self._wj = np.log(pr)
        self._vj = np.log(1 - pr)
        return self

    def predict(self, X):
        return np.argmax(self.decision_function_raw(X), axis=1)

    def decision_function(self, X):
        _ = self.predict_proba(X)
        df = _ * 2 - 1
        df[df > 1] = 1
        df[df < -1] = -1
        return df

    def predict_proba(self, X):
        X = self.decision_function_raw(X)
        return np.exp(X)

    def decision_function_raw(self, X):
        wj = self._wj
        vj = self._vj
        hy = []
        prior = self._prior
        x = np.zeros(self.num_terms)
        for d in X:
            x.fill(0)
            index = d.indices
            if index.shape[0] > 0:
                x[index] = 1
            _ = ((x * wj) + (1 - x) * vj).sum(axis=1) + prior
            hy.append(_)
        hy = np.array(hy)
        hy = hy - np.atleast_2d(logsumexp(hy, axis=1)).T
        return hy


class Multinomial(Bernoulli):
    """Multinomial classifier"""

    def fit(self, X, klass):
        self._num_terms = X.shape[1]
        klasses = np.unique(klass)
        pr = np.zeros((klasses.shape[0], self.num_terms))
        for i, k in zip(X, klass):
            index = i.indices
            if index.shape[0] > 0:
                pr[k, index] += 1
        den = pr.sum(axis=1)
        self._log_xj = np.log((1 + pr) / np.atleast_2d(self.num_terms + den).T)
        return self

    def decision_function_raw(self, X):
        xj = self._log_xj
        hy = []
        for d in X:
            x = np.zeros(self.num_terms)
            index = d.indices
            if index.shape[0] > 0:
                x[index] = 1
            _ = (xj * x).sum(axis=1)
            hy.append(_)
        return np.array(hy)


class ThumbsUpDownEs(ThumbsUpDown):
    """Spanish thumbs up and down model"""

    def __init__(self, *args, **kwargs):
        super(ThumbsUpDownEs, self).__init__(lang=_SPANISH, stemming=False)

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([self.__getitem__(x) for x in X])


class ThumbsUpDownEn(ThumbsUpDown):
    """English thumbs up and down model"""

    def __init__(self, *args, **kwargs):
        super(ThumbsUpDownEn, self).__init__(lang=_ENGLISH, stemming=False)

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([self.__getitem__(x) for x in X])


class ThumbsUpDownAr(ThumbsUpDown):
    """Arabic thumbs up and down model"""

    def __init__(self, *args, **kwargs):
        super(ThumbsUpDownAr, self).__init__(lang=_ARABIC, stemming=False)

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([self.__getitem__(x) for x in X])


class Vec(BaseTextModel):
    """Read the key vec, useful to incorporate external knowledge as FastText print-sentence-vectors"""

    def __getitem__(self, x):
        return x['vec']

    def fit(self, X):
        return self


class TextModelInv(TextModel):
    """Extends :mod:`b4msa.textmodel.TextModel` in order to override tokenize method."""
    
    def __init__(self, is_by_character: bool=True, **kwargs):
        """
        :param is_by_character: if true, inverts the text character by character; if false, inverts word by word 
        :type is_by_character: bool
        """
        
        super(TextModelInv, self).__init__(**kwargs)
        self.is_by_character = is_by_character
    
    def tokenize(self, text: Union[str, dict]):
        """Inverts text

        :param text: Text
        :type text: str
        """
        if isinstance(text, dict):
            text = self.get_text(text)

        if self.is_by_character:
            result = text[::-1]
        else:
            result = text.split(' ')
            result.reverse()
            result = " ".join(result)            
            
        return super(TextModelInv, self).tokenize(result)


class GaussianBayes(object):
    """
    >>> from scipy.stats import multivariate_normal
    >>> from EvoMSA.model import GaussianBayes  
    >>> X_1 = multivariate_normal(mean=[5, 5], cov=[[4, 0], [0, 2]]).rvs(size=1000)
    >>> X_2 = multivariate_normal(mean=[1.5, -1.5], cov=[[2, 1], [1, 3]]).rvs(size=1000)
    >>> X_3 = multivariate_normal(mean=[12.5, -3.5], cov=[[2, 3], [3, 7]]).rvs(size=1000)
    >>> X = np.concatenate((X_1, X_2, X_3))
    >>> y = np.array([1] * 1000 + [2] * 1000 + [3] * 1000)
    >>> bayes = GaussianBayes().fit(X, y)
    >>> hy = bayes.predict(X)
    >>> bayes = GaussianBayes(naive=True).fit(X, y)
    >>> hy_naive = bayes.predict(X)
    """
    def __init__(self, naive=False) -> None:
        self._naive = naive

    @property
    def naive(self):
        return self._naive
    
    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, y):
        labels, counts = np.unique(y, return_counts=True)
        prior = counts / counts.sum()        
        self.labels = labels
        self._prior = np.log(prior)

    @property
    def likelihood(self):
        return self._likelihood

    @likelihood.setter
    def likelihood(self, D):
        X, y = D
        likelihood = []
        for k in self.labels:
            mask = y == k
            mu = np.mean(X[mask], axis=0)
            if self.naive:
                cov = np.var(X[mask], axis=0, ddof=1)
            else:
                cov = np.cov(X[mask], rowvar=False)
            _ = multivariate_normal(mean=mu, cov=cov, allow_singular=True)
            likelihood.append(_)
        self._likelihood = likelihood

    def fit(self, X, y):
        self.prior = y
        self.likelihood = (X, y)
        return self

    def predict_log_proba(self, X):
        log_likelihood = np.vstack([m.logpdf(X) for m in self.likelihood]).T
        prior = self.prior
        posterior = log_likelihood + prior
        evidence = np.atleast_2d(logsumexp(posterior, axis=1)).T
        return posterior - evidence

    def predict_proba(self, X):
        _ = self.predict_log_proba(X)
        return np.exp(_)

    def predict(self, X):
        hy = self.predict_log_proba(X)
        _ = np.argmax(hy, axis=1)
        return self.labels[_]