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
from EvoMSA.thumbs_up_down import ThumbsUpDown, _ARABIC, _ENGLISH, _SPANISH
import os
import pickle
import gzip


class BaseTextModel(object):
    def __init__(self, corpus=None, **kwargs):
        pass

    @property
    def num_terms(self):
        try:
            return self._num_terms
        except AttributeError:
            self._num_terms = None
        return None

    def tonp(self, X):
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
    def __init__(self, random_state=0):
        pass
    
    def fit(self, X, y):
        return self

    def decision_function(self, X):
        pass


class Identity(BaseTextModel, BaseClassifier):
    def __getitem__(self, x):
        return x

    def decision_function(self, X):
        return X

    def predict_proba(self, X):
        return self.decision_function(X)


class B4MSATextModel(TextModel, BaseTextModel):
    def __init__(self, *args, **kwargs):
        self._text = os.getenv('TEXT', default='text')
        TextModel.__init__(self, *args, **kwargs)

    def get_text(self, text):
        return text[self._text]

    def tokenize(self, text):
        if isinstance(text, dict):
            text = self.get_text(text)
        if isinstance(text, (list, tuple)):
            tokens = []
            for _text in text:
                tokens.extend(TextModel.tokenize(self, _text))
            return tokens
        else:
            return TextModel.tokenize(self, text)


class EmoSpace(BaseTextModel, BaseClassifier):
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
        _ = [tm[self.get_text(x)] for x in X]
        return np.array([m.decision_function(_) for m in self._classifiers]).T

    def predict_proba(self, X):
        return self.decision_function(X)

    def __getitem__(self, x):
        tm = self._textModel
        _ = [tm[self.get_text(x)]]
        _ = np.array([m.decision_function(_) for m in self._classifiers]).flatten()
        return [[k, v] for k, v in enumerate(_)]


class EmoSpaceEn(EmoSpace):
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


class AffectiveAr(Corpus):
    def __init__(self, *args, **kwargs):
        fname = os.path.join(os.path.dirname(__file__), 'conf', 'ar.affective.words.json')
        corpus = []
        for x in tweet_iterator(fname):
            corpus += x['words']
        super(AffectiveAr, self).__init__(corpus)


class AffectiveEn(Corpus):
    def __init__(self, *args, **kwargs):
        fname = os.path.join(os.path.dirname(__file__), 'conf', 'en.affective.words.json')
        corpus = []
        for x in tweet_iterator(fname):
            corpus += x['words']
        super(AffectiveEn, self).__init__(corpus)


class AffectiveEs(Corpus):
    def __init__(self, *args, **kwargs):
        fname = os.path.join(os.path.dirname(__file__), 'conf', 'es.affective.words.json')
        corpus = []
        for x in tweet_iterator(fname):
            corpus += x['words']
        super(AffectiveEs, self).__init__(corpus)


class Bernulli(BaseClassifier):
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
    def __init__(self, *args, **kwargs):
        """
        Initializes the parameters for specific language
        """
        fname = os.path.join(os.path.dirname(__file__), 'conf', 'es.affective.words.json')
        super(ThumbsUpDownEs, self).__init__(file_name=fname, lang=_SPANISH, stemming=False)


class ThumbsUpDownEn(ThumbsUpDown):
    def __init__(self, *args, **kwargs):
        """
        Initializes the parameters for specific language
        """
        fname = os.path.join(os.path.dirname(__file__), 'conf', 'en.affective.words.json')
        super(ThumbsUpDownEn, self).__init__(file_name=fname, lang=_ENGLISH, stemming=False)


class ThumbsUpDownAr(ThumbsUpDown):
    def __init__(self, *args, **kwargs):
        """
        Initializes the parameters for specific language
        """
        fname = os.path.join(os.path.dirname(__file__), 'conf', 'ar.affective.words.json')
        super(ThumbsUpDownAr, self).__init__(file_name=fname, lang=_ARABIC, stemming=False)
