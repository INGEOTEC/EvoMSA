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

from b4msa.utils import tweet_iterator
from b4msa.textmodel import TextModel
from sklearn.preprocessing import LabelEncoder
import numpy as np


class Corpus(object):
    def __init__(self):
        self._m = {}
        self._num_terms = 0
        self._training = True
        self._textModel = TextModel([''], token_list=[-1])

    def fit(self, c):
        r = [self.__getitem__(x) for x in c]
        self._training = False
        return r
            
    @property
    def num_terms(self):
        return self._num_terms

    def vec(self, d):
        v = np.zeros(self.num_terms)
        for x in self.__getitem__(d):
            v[x] = 1
        return v

    def __getitem__(self, d):
        tokens = []
        for t in self._textModel.tokenize(d):
            try:
                index = self._m[t]
            except KeyError:
                if not self._training:
                    continue
                index = self._num_terms
                self._m[t] = index
                self._num_terms += 1
            tokens.append(index)
        return tokens


class Bernulli(object):
    def __init__(self):
        self._corpus = Corpus()

    @property
    def corpus(self):
        return self._corpus

    def fit(self, text, klass):
        c = self.corpus
        r = c.fit(text)
        klasses = np.unique(klass)
        pr = np.zeros((klasses.shape[0], c.num_terms))
        for i, k in zip(r, klass):
            pr[k, np.array(i)] += 1
        _ = np.atleast_2d(c.num_terms + np.array([(klass == _k).sum() for _k in klasses])).T
        pr = (1 + pr) / _
        self._wj = np.log(pr)
        self._vj = np.log(1 - pr)
        return self

    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)

    def predict_proba(self, X):
        X = self.decision_function(X)
        pr = np.exp(X)
        den = pr.sum(axis=1)
        _ = pr / np.atleast_2d(den).T
        return _ * 2 - 1

    def decision_function(self, X):
        c = self._corpus
        wj = self._wj
        vj = self._vj
        if not isinstance(X, list):
            X = [X]
        hy = []
        for d in X:
            x = c.vec(d)
            _ = ((x * wj) + (1 - x) * vj).sum(axis=1)
            hy.append(_)
        return np.array(hy)
    

if __name__ == '__main__':
    import sys
    import json
    train = sys.argv[1]
    test = sys.argv[2]
    _ = [[x['klass'], x['text']] for x in tweet_iterator(train)]
    y = [d[0] for d in _]
    X = [d[1] for d in _]
    le = LabelEncoder().fit(y)
    y = le.transform(y)
    bernulli = Bernulli().fit(X, y)
    with open('%s.predict' % test, 'w') as fpt:
        for x in tweet_iterator(test):
            x['klass'] = str(le.inverse_transform(bernulli.predict(x['text']))[0])
            _ = json.dumps(x)
            fpt.write(_ + '\n')

