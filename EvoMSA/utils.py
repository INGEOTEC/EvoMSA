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
from sklearn.preprocessing import LabelEncoder
import os
from urllib import request
from microtc.utils import load_model
import numpy as np


class LabelEncoderWrapper(object):
    """Wrapper of LabelEncoder. The idea is to keep the order when the classes are numbers
    at some point this will help improve the performance in ordinary classification problems

    :param classifier: Specifies whether it is a classification problem
    :type classifier: bool
    """

    def __init__(self, classifier=True):
        self._m = {}
        self._classifier = classifier

    @property
    def classifier(self):
        """Whether EvoMSA is acting as classifier"""

        return self._classifier

    def fit(self, y):
        """Fit the label encoder

        :param y: Independent variables
        :type y: list or np.array
        :rtype: self
        """

        if not self.classifier:
            return self
        try:
            n = [int(x) for x in y]
        except ValueError:
            return LabelEncoder().fit(y)
        self.classes_ = np.unique(n)
        self._m = {v: k for k, v in enumerate(self.classes_)}
        self._inv = {v: k for k, v in self._m.items()}
        return self

    def transform(self, y):
        if not self.classifier:
            return np.array([float(_) for _ in y])
        return np.array([self._m[int(x)] for x in y])

    def inverse_transform(self, y):
        if not self.classifier:
            return y
        return np.array([self._inv[int(x)] for x in y])


def download(model_fname):
    if os.path.isfile(model_fname):
        return model_fname
    dirname = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    fname = os.path.join(dirname, model_fname)
    if not os.path.isfile(fname):
        request.urlretrieve("http://ingeotec.mx/~mgraffg/models/%s" % model_fname,
                            fname)
    return fname


def get_model(model_fname):
    fname = download(model_fname)
    return load_model(fname)


def linearSVC_array(classifiers):
    """Transform LinearSVC into weight stored in array.array

    :param classifers: List of LinearSVC where each element is binary
    :type classifers: list
    """

    import array
    intercept = array.array('d', [x.intercept_[0] for x in classifiers])
    coef = np.vstack([x.coef_[0] for x in classifiers])
    coef = array.array('d', coef.T.flatten())
    return coef, intercept


def compute_p(syss):
    from scipy.stats import wilcoxon
    p = []
    mu = syss.mean(axis=0)
    best = mu.argmax()
    for i in range(syss.shape[1]):
        if i == best:
            p.append(np.inf)
            continue
        try:
            pv = wilcoxon(syss[:, best], syss[:, i])[1]
            p.append(pv)
        except ValueError:
            p.append(np.inf)
    ps = np.argsort(p)
    alpha = [np.inf for _ in ps]
    m = ps.shape[0] - 1
    for r, i in enumerate(ps[:-1]):
        alpha_c = (0.05 / (m + 1 - (r + 1)))
        if p[i] > alpha_c:
            break
        alpha[i] = alpha_c
    return p, alpha
