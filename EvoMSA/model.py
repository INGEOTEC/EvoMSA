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
from b4msa.classifier import SVC


class BaseTextModel(object):
    def __init__(self, corpus, **kwargs):
        pass

    def __getitem__(self, x):
        pass


class Identity(BaseTextModel):
    def __getitem__(self, x):
        return x


class BaseClassifier(object):
    def __init__(self, random_state=0):
        pass

    def fit(self, X, y):
        pass

    def decision_function(self, X):
        pass


class B4MSATextModel(TextModel, BaseTextModel):
    def __getitem__(self, x):
        if x is None:
            x = ''
        return TextModel.__getitem__(self, str(x))


class B4MSAClassifier(SVC, BaseClassifier):
    def __init__(self, random_state=0):
        SVC.__init__(self, model=None, random_state=random_state)

    def decision_function(self, X):
        _ = SVC.decision_function(self, X)
        _[_ > 1] = 1
        _[_ < -1] = -1
        return _


class Corpus(BaseTextModel):
    def __init__(self, corpus, **kwargs):
        self._m = {}
        self._num_terms = 0
        self._training = True
        self._textModel = TextModel([''], token_list=[-1])
        self.fit(corpus)

    def fit(self, c):
        r = [self.__getitem__(x) for x in c]
        self._training = False
        return r

    @property
    def num_terms(self):
        return self._num_terms

    def vec(self, d):
        v = np.zeros(self.num_terms)
        for i, x in self.__getitem__(d):
            v[i] = x
        return v

    def __getitem__(self, d):
        tokens = []
        for t in self._textModel.tokenize(d):
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


class Bernulli(BaseClassifier):
    def __init__(self, random_state=0):
        self._num_terms = -1

    @property
    def num_terms(self):
        return self._num_terms

    def fit(self, X, klass):
        self._num_terms = max([max([_[0] for _ in x]) for x in X]) + 1
        klasses = np.unique(klass)
        pr = np.zeros((klasses.shape[0], self.num_terms))
        for i, k in zip(X, klass):
            pr[k, np.array([_[0] for _ in i])] += 1
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
        if not isinstance(X, list):
            X = [X]
        hy = []
        for d in X:
            x = np.zeros(self.num_terms)
            index = [_[0] for _ in d if _[0] < self._num_terms]
            if len(index):
                x[np.array(index)] = 1
            _ = ((x * wj) + (1 - x) * vj).sum(axis=1)
            hy.append(_)
        return np.array(hy)


class Multinomial(Bernulli):
    def fit(self, X, klass):
        self._num_terms = max([max([_[0] for _ in x]) for x in X]) + 1
        klasses = np.unique(klass)
        pr = np.zeros((klasses.shape[0], self.num_terms))
        for i, k in zip(X, klass):
            pr[k, np.array([_[0] for _ in i])] += 1
        den = pr.sum(axis=1)
        self._log_xj = np.log((1 + pr) / np.atleast_2d(self.num_terms + den).T)
        return self
        
    def decision_function_raw(self, X):
        xj = self._log_xj
        if not isinstance(X, list):
            X = [X]
        hy = []
        for d in X:
            x = np.zeros(self.num_terms)
            index = [_[0] for _ in d if _[0] < self._num_terms]
            if len(index):
                x[np.array(index)] = 1
            _ = (xj * x).sum(axis=1)
            hy.append(_)
        return np.array(hy)
    
