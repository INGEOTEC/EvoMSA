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
import numpy as np
from b4msa.textmodel import TextModel
from b4msa.utils import tweet_iterator
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC
from ConceptModelling.thumbs_up_down import ThumbsUpDown, _ARABIC, _ENGLISH, _SPANISH
import os
import pickle
import gzip


class BaseTextModel(object):
    """Base class for text model

    :param corpus: Text to build the text model
    :type corpus: list or dict
    """

    def __init__(self, corpus=None, **kwargs):
        pass

    @property
    def num_terms(self):
        """Dimension which is the number of terms of the corpus

        :rtype: int
        """

        try:
            return self._num_terms
        except AttributeError:
            self._num_terms = None
        return None

    def tonp(self, X):
        """Sparse representation to sparce matrix

        :param X: Sparse representation of matrix
        :type X: list
        :rtype: csr_matrix
        """

        data = []
        row = []
        col = []
        for r, x in enumerate(X):
            cc = [_[0] for _ in x if np.isfinite(_[1])]
            col += cc
            data += [_[1] for _ in x if np.isfinite(_[1])]
            _ = [r] * len(cc)
            row += _
        if self.num_terms is None:
            _ = csr_matrix((data, (row, col)))
            self._num_terms = _.shape[1]
            return _
        return csr_matrix((data, (row, col)), shape=(len(X), self.num_terms))

    def __getitem__(self, x):
        pass

    def tokenize(self, text):
        pass


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


class Identity(BaseTextModel, BaseClassifier):
    """Identity function used as either text model or classifier or regressor"""

    def tonp(self, x):
        return x

    def __getitem__(self, x):
        return x

    def decision_function(self, X):
        return X

    def predict_proba(self, X):
        return self.decision_function(X)


class B4MSATextModel(TextModel, BaseTextModel):
    """Text model based on B4MSA"""

    def __init__(self, *args, **kwargs):
        self._text = os.getenv('TEXT', default='text')
        TextModel.__init__(self, *args, **kwargs)

    def get_text(self, text):
        """Return self._text key from text

        :param text: Text
        :type text: dict
        """

        return text[self._text]

    def tokenize(self, text):
        """Tokenize a text

        :param text: Text
        :type text: dict or str
        """

        if isinstance(text, dict):
            text = self.get_text(text)
        if isinstance(text, (list, tuple)):
            tokens = []
            for _text in text:
                tokens.extend(TextModel.tokenize(self, _text))
            return tokens
        else:
            return TextModel.tokenize(self, text)


class HaSpace(object):
    """Text classifier based on a Humman Annotated dataset in Spanish"""

    def __init__(self, *args, **kwargs):
        self._model = self.get_model()
        self._text = os.getenv('TEXT', default='text')

    def fit(self, X, y):
        return self

    def decision_function(self, X):
        X = [self.get_text(x) for x in X]
        X = self._model.decision_function(X)
        X[~np.isfinite(X)] = 0
        return X

    def get_model(self):
        """Return the model"""

        import os
        from urllib import request
        fname = os.path.join(os.path.dirname(__file__), 'ha-es.model')
        if not os.path.isfile(fname):
            request.urlretrieve("http://ingeotec.mx/~mgraffg/models/ha-es.model",
                                fname)
        with gzip.open(fname) as fpt:
            _ = pickle.load(fpt)
        _.n_jobs = 1
        return _

    def get_text(self, text):
        """Return self._text key from text

        :param text: Text
        :type text: dict
        """

        key = self._text
        if isinstance(text, (list, tuple)):
            return " | ".join([x[key] for x in text])
        return text[key]


class HaSpaceEn(HaSpace):
    """Text classifier based on a Humman Annotated dataset in English"""

    def get_model(self):
        import os
        from urllib import request
        fname = os.path.join(os.path.dirname(__file__), 'ha-en.model')
        if not os.path.isfile(fname):
            request.urlretrieve("http://ingeotec.mx/~mgraffg/models/ha-en.model",
                                fname)
        with gzip.open(fname) as fpt:
            _ = pickle.load(fpt)
        _.n_jobs = 1
        return _


class HaSpaceAr(HaSpace):
    """Text classifier based on a Humman Annotated dataset in Arabic"""

    def get_model(self):
        import os
        from urllib import request
        fname = os.path.join(os.path.dirname(__file__), 'ha-ar.model')
        if not os.path.isfile(fname):
            request.urlretrieve("http://ingeotec.mx/~mgraffg/models/ha-ar.model",
                                fname)
        with gzip.open(fname) as fpt:
            _ = pickle.load(fpt)
        _.n_jobs = 1
        return _


class EmoSpace(BaseTextModel, BaseClassifier):
    """Spanish text model or classifier based on Emojis

    Let us describe the procedure to use EmoSpace to create a model using it as text model

    Read the dataset

    >>> from EvoMSA import base
    >>> from b4msa.utils import tweet_iterator
    >>> import os
    >>> tweets = os.path.join(os.path.dirname(base.__file__), 'tests', 'tweets.json')
    >>> D = [[x['text'], x['klass']] for x in tweet_iterator(tweets)]

    Once the dataset is loaded, it is time to import the models and the classifier

    >>> from EvoMSA.model import BaseTextModel, B4MSATextModel, EmoSpace
    >>> from sklearn.svm import LinearSVC

    The models one wishes to use are set in a list of lists, namely:

    >>> models = [[B4MSATextModel, LinearSVC], [EmoSpace, LinearSVC]]

    EvoMSA model is created using

    >>> from EvoMSA.base import EvoMSA
    >>> evo = EvoMSA(models=models).fit([dict(text=x[0]) for x in D], [x[1] for x in D])

    Predict a sentence in Spanish

    >>> evo.predict(['EvoMSA esta funcionando', 'EmoSpace esta funcionando'])
    """

    def __init__(self, *args, **kwargs):
        self._textModel, self._classifiers = self.get_model()
        self._text = os.getenv('TEXT', default='text')

    def fit(self, X, y):
        pass

    def get_model(self):
        import os
        from urllib import request
        fname = os.path.join(os.path.dirname(__file__), 'emo-es.b4msa')
        if not os.path.isfile(fname):
            request.urlretrieve("http://ingeotec.mx/~mgraffg/models/emo-es.b4msa",
                                fname)
        with gzip.open(fname) as fpt:
            return pickle.load(fpt)

    def get_text(self, text):
        key = self._text
        if isinstance(text, (list, tuple)):
            return " | ".join([x[key] for x in text])
        return text[key]

    def decision_function(self, X):
        tm = self._textModel
        _ = tm.tonp([tm[x] for x in X])
        return np.array([m.decision_function(_) for m in self._classifiers]).T

    def predict_proba(self, X):
        return self.decision_function(X)

    def transform(self, X):
        tm = self._textModel
        D = tm.tonp([tm[self.get_text(x)] for x in X])
        D = np.array([m.decision_function(D) for m in self._classifiers])
        return [[[k, v] for k, v in enumerate(_)] for _ in D.T]

    def __getitem__(self, x):
        tm = self._textModel
        _ = tm.tonp([tm[self.get_text(x)]])
        _ = np.array([m.decision_function(_) for m in self._classifiers]).flatten()
        return [[k, v] for k, v in enumerate(_)]


class EmoSpaceEs(EmoSpace):
    pass


class EmoSpaceEn(EmoSpace):
    """English text model or classifier based on a Emojis"""

    def get_model(self):
        import os
        from urllib import request
        fname = os.path.join(os.path.dirname(__file__), 'emo-en.b4msa')
        if not os.path.isfile(fname):
            request.urlretrieve("http://ingeotec.mx/~mgraffg/models/emo-en.b4msa",
                                fname)
        with gzip.open(fname) as fpt:
            return pickle.load(fpt)


class EmoSpaceAr(EmoSpace):
    """Arabic text model or classifier based on a Emojis"""

    def get_model(self):
        import os
        from urllib import request
        fname = os.path.join(os.path.dirname(__file__), 'emo-ar.b4msa')
        if not os.path.isfile(fname):
            request.urlretrieve("http://ingeotec.mx/~mgraffg/models/emo-ar.b4msa",
                                fname)
        with gzip.open(fname) as fpt:
            return pickle.load(fpt)


class Corpus(BaseTextModel):
    """Text model using only words"""

    def __init__(self, corpus, **kwargs):
        self._text = os.getenv('TEXT', default='text')
        self._m = {}
        self._num_terms = 0
        self._training = True
        self._textModel = TextModel([''], token_list=[-1])
        self.fit(corpus)

    def get_text(self, text):
        return text[self._text]

    def fit(self, c):
        r = [self.__getitem__(x) for x in c]
        self._training = False
        return r

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
        fname = os.path.join(os.path.dirname(__file__), 'conf', 'aggressiveness.ar')
        corpus = []
        for x in tweet_iterator(fname):
            corpus += x['words']
        super(AggressivenessAr, self).__init__(corpus)


class AggressivenessEn(Corpus):
    """English text model using an aggressive corpus"""

    def __init__(self, *args, **kwargs):
        fname = os.path.join(os.path.dirname(__file__), 'conf', 'aggressiveness.en')
        corpus = []
        for x in tweet_iterator(fname):
            corpus += x['words']
        super(AggressivenessEn, self).__init__(corpus)


class AggressivenessEs(Corpus):
    """Spanish text model using an aggressive corpus"""

    def __init__(self, *args, **kwargs):
        fname = os.path.join(os.path.dirname(__file__), 'conf', 'aggressiveness.es')
        corpus = []
        for x in tweet_iterator(fname):
            corpus += x['words']
        super(AggressivenessEs, self).__init__(corpus)


class Bernulli(BaseClassifier):
    """Bernulli classifier"""

    def __init__(self, random_state=0):
        self._num_terms = -1

    @property
    def num_terms(self):
        return self._num_terms

    def fit(self, X, klass):
        self._num_terms = X.shape[1]
        klasses = np.unique(klass)
        pr = np.zeros((klasses.shape[0], self.num_terms))
        for i, k in zip(X, klass):
            index = i.indices
            if index.shape[0] > 0:
                pr[k, index] += 1
        _ = np.atleast_2d(self.num_terms + np.array([(klass == _k).sum() for _k in klasses])).T
        pr = (1 + pr) / _
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
        pr = np.exp(X)
        den = pr.sum(axis=1)
        _ = pr / np.atleast_2d(den).T
        return _

    def decision_function_raw(self, X):
        wj = self._wj
        vj = self._vj
        hy = []
        for d in X:
            x = np.zeros(self.num_terms)
            index = d.indices
            if index.shape[0] > 0:
                x[index] = 1
            _ = ((x * wj) + (1 - x) * vj).sum(axis=1)
            hy.append(_)
        return np.array(hy)


class Multinomial(Bernulli):
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


class ThumbsUpDownEs(ThumbsUpDown, BaseTextModel):
    """Spanish thumbs up and down model"""

    def __init__(self, *args, **kwargs):
        super(ThumbsUpDownEs, self).__init__(lang=_SPANISH, stemming=False)

    def tonp(self, X):
        """Convert list to np

        :param X: list of tuples
        :type X: list
        :rtype: np.array
        """

        return np.array(X)


class ThumbsUpDownEn(ThumbsUpDown, BaseTextModel):
    """English thumbs up and down model"""

    def __init__(self, *args, **kwargs):
        super(ThumbsUpDownEn, self).__init__(lang=_ENGLISH, stemming=False)

    def tonp(self, X):
        """Convert list to np

        :param X: list of tuples
        :type X: list
        :rtype: np.array
        """

        return np.array(X)


class ThumbsUpDownAr(ThumbsUpDown, BaseTextModel):
    """Arabic thumbs up and down model"""

    def __init__(self, *args, **kwargs):
        super(ThumbsUpDownAr, self).__init__(lang=_ARABIC, stemming=False)

    def tonp(self, X):
        """Convert list to np

        :param X: list of tuples
        :type X: list
        :rtype: np.array
        """

        return np.array(X)


class Vec(BaseTextModel):
    """Read the key vec, useful to incorporate external knowledge as FastText print-sentence-vectors"""

    def __getitem__(self, x):
        return x['vec']

    def tonp(self, X):
        """Convert list to np

        :param X: list of tuples
        :type X: list
        :rtype: np.array
        """

        return np.array(X)
