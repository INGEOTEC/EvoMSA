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
from typing import List, Union, Dict, Tuple, Callable
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
from microtc.utils import load_model, tweet_iterator, Counter
from scipy.sparse import csr_matrix
from os.path import join, dirname, isdir, isfile
from urllib import request
from urllib.error import HTTPError
import numpy as np
import hashlib
import os
import gzip
try:
    USE_TQDM = True
    from tqdm import tqdm
except ImportError:
    USE_TQDM = False

MICROTC = '2.4.9'
MODEL_LANG = ['ar', 'ca', 'de', 'en', 'es', 'fr',
              'hi', 'in', 'it', 'ja', 'ko', 'nl',
              'pl', 'pt', 'ru', 'tl', 'tr', 'zh']


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


class Cache(object):
    """Store the output of the text models"""

    def __init__(self, basename):
        if basename is None:
            self._cache = None
        else:
            dirname = os.path.dirname(basename)
            if len(dirname) and not os.path.isdir(dirname):
                os.mkdir(dirname)
            self._cache = basename

    def __iter__(self):
        if self._cache is None:
            while True:
                yield None
        for i in self.textModels:
            yield i

    @property
    def textModels(self):
        try:
            return self._textModels
        except AttributeError:
            self._textModels = list()
        return self._textModels

    @property
    def ml(self):
        try:
            return self._classifiers
        except AttributeError:
            self._classifiers = list()
        return self._classifiers

    def ml_train(self):
        if self._cache is None or len(self.ml) == 0:
            while True:
                yield None
        for i in self.ml:
            yield i

    def ml_kfold(self):
        if self._cache is None or len(self.ml) == 0:
            while True:
                yield None
        for i in self.ml:
            yield i + '-K'

    @staticmethod
    def get_name(value):
        if isinstance(value, str):
            return hashlib.md5(value.encode()).hexdigest()
        else:
            try:
                vv = value.__name__
            except AttributeError:
                vv = value.__class__.__name__
            return vv

    def append(self, value, ml=None):
        if self._cache is None:
            return
        name = self._cache + "-%s" % self.get_name(value)
        if ml is not None:
            self.ml.append(name + '-' + self.get_name(ml))
        self.textModels.append(name)


def download(model_fname, force=False):
    if os.path.isfile(model_fname) and not force:
        return model_fname
    dirname = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    fname = os.path.join(dirname, model_fname)
    if not os.path.isfile(fname) or force:
        request.urlretrieve("https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/%s" % model_fname,
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


def bootstrap_confidence_interval(y: np.ndarray,
                                  hy: np.ndarray,
                                  metric: Callable[[float, float], float]=lambda y, hy: recall_score(y, hy,
                                                                                           average="macro"),
                                  alpha: float=0.05,
                                  nbootstrap: int=500) -> Tuple[float, float]:
    """Confidence interval from predictions"""
    alpha /= 2
    B = []
    for _ in range(nbootstrap):
        s = np.random.randint(hy.shape[0], size=hy.shape[0])
        _ = metric(y[s], hy[s])
        B.append(_)
    return (np.percentile(B, alpha * 100), np.percentile(B, (1 - alpha) * 100))


class ConfidenceInterval(object):
    """Estimate the confidence interval

    >>> from EvoMSA import base
    >>> from EvoMSA.utils import ConfidenceInterval
    >>> from microtc.utils import tweet_iterator
    >>> import os
    >>> tweets = os.path.join(os.path.dirname(base.__file__), 'tests', 'tweets.json')
    >>> D = list(tweet_iterator(tweets))
    >>> X = [x['text'] for x in D]
    >>> y = [x['klass'] for x in D]
    >>> kw = dict(stacked_method="sklearn.naive_bayes.GaussianNB") 
    >>> ci = ConfidenceInterval(X, y, evomsa_kwargs=kw)
    >>> result = ci.estimate()

    """

    def __init__(self, X: List[str], y: Union[np.ndarray, list],
                 Xtest: List[str]=None, y_test: Union[np.ndarray, list]=None,
                 evomsa_kwargs: Dict=dict(), 
                 folds: Union[None, BaseCrossValidator]=None, ) -> None:
        self._X = X
        self._y = np.atleast_1d(y)
        self._Xtest = Xtest
        self._y_test = y_test
        self._evomsa_kwargs = evomsa_kwargs
        self._folds = folds

    @property
    def gold(self):
        if self._y_test is not None:
            return self._y_test
        return self._y
    
    @property
    def hy(self):
        from .base import EvoMSA
        try:
            return self._hy
        except AttributeError:
            if self._Xtest is not None:
                m = EvoMSA(**self._evomsa_kwargs)
                m.fit(self._X, self._y)
                hy = m.predict(self._Xtest)
                self._hy = hy
                return hy
            folds = self._folds
            if folds is None:
                folds = StratifiedKFold(n_splits=5,
                                        shuffle=True, random_state=0)
            hy = np.empty_like(self._y)
            X, y = self._X, self._y
            for tr, ts in folds.split(X, y):
                m = EvoMSA(**self._evomsa_kwargs)
                m.fit([X[x] for x in tr], y[tr])
                hy[ts] = m.predict([X[x] for x in ts])
            self._hy = hy
            return self._hy

    def estimate(self, alpha: float=0.05,
                       metric: Callable[[float, float], float]=lambda y, hy: recall_score(y, hy,
                                                                                          average="macro"),
                       nbootstrap: int=500)->Tuple[float, float]:
        return bootstrap_confidence_interval(self.gold, self.hy,
                                             metric=metric,
                                             alpha=alpha,
                                             nbootstrap=nbootstrap)


class Download(object):
    def __init__(self, url, output='t.tmp') -> None:
        self._url = url
        self._output = output
        try:
            request.urlretrieve(url, output, reporthook=self.progress)
        except HTTPError:
            raise Exception(url)
        self.close()
    
    @property
    def tqdm(self):
        if not USE_TQDM:
            return None
        try:
            return self._tqdm
        except AttributeError:
            self._tqdm = tqdm(total=self._nblocks, leave=False)
        return self._tqdm
    
    def close(self):
        if USE_TQDM:
            self.tqdm.close()
        
    def update(self):
        if USE_TQDM:
            self.tqdm.update()

    def progress(self, nblocks, block_size, total):
        self._nblocks = total // block_size
        self.update()

        
def load_bow(lang='es', d=17, func='most_common_by_type', v1=False):
    def load(filename):
        try:
            with gzip.open(filename, 'rb') as fpt:
                return str(fpt.read(), encoding='utf-8')
        except Exception:
            os.unlink(filename)
            raise Exception(filename)

    lang = lang.lower().strip()
    assert lang in MODEL_LANG
    diroutput = join(dirname(__file__), 'models')
    if not isdir(diroutput):
        os.mkdir(diroutput)
    if v1:
        filename = f'{lang}_2.4.2.microtc'
        url = f'https://github.com/INGEOTEC/text_models/releases/download/models/{filename}'
        output = join(diroutput, filename)
        if not isfile(output):
            Download(url, output)
        return load_model(output)

    filename = f'{lang}_{MICROTC}_bow_{func}_{d}.json.gz'        
    url = f'https://github.com/INGEOTEC/text_models/releases/download/models/{filename}'
    output = join(diroutput, filename)
    if not isfile(output):
        Download(url, output)
    return Counter.fromjson(load(output))


class Linear(object):
    """
    >>> from EvoMSA.model import Linear
    >>> linear = Linear(coef=[12, 3], intercept=0.5, labels=[0, 'P'])
    >>> X = np.array([[2, -1]])
    >>> linear.decision_function(X)
    21.5
    >>> linear.predict(X)[0]
    'P'
    """

    def __init__(self, coef: Union[list, np.ndarray],
                 intercept: float=0,
                 labels: Union[list, np.ndarray, None]=None,
                 N: int=0) -> None:
        self._coef = np.atleast_1d(coef)
        self._intercept = intercept
        self._labels = np.atleast_1d(labels) if labels is not None else labels
        self._N = N
    
    @property
    def N(self):
        return self._N

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, v):
        self._labels = v

    def decision_function(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        if isinstance(X, np.ndarray):
            return np.dot(X, self._coef) + self._intercept
        return X.dot(self._coef) + self._intercept

    def predict(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        hy = self.decision_function(X)
        if self._labels is not None:
            return self._labels[np.where(hy > 0, 1, 0)]
        return np.where(hy > 0, 1, -1)

    
def _load_text_repr(lang='es', name='emojis', 
                    k=None,  d=17, func='most_common_by_type',
                    v1=False):
    import os
    from os.path import isdir, join, isfile, dirname
    from urllib.error import HTTPError
    def load(filename):
        try:
            return [Linear(**x) for x in tweet_iterator(filename)]
        except Exception:
            os.unlink(filename)
    lang = lang.lower().strip()
    assert lang in MODEL_LANG
    diroutput = join(dirname(__file__), 'models')
    if not isdir(diroutput):
        os.mkdir(diroutput)
    if v1:
        filename = f'{lang}_{name}_muTC2.4.2.json.gz'
    else:
        filename = f'{lang}_{MICROTC}_{name}_{func}_{d}.json.gz'
    url = f'https://github.com/INGEOTEC/text_models/releases/download/models/{filename}'
    output = join(diroutput, filename)
    if not isfile(output):
        Download(url, output)
    models = load(output)
    if k is None:
        return models
    return models[k]


def load_emoji(lang='es', emoji=None,
               d=17, func='most_common_by_type',
               v1=False):

    lang = lang.lower().strip()
    assert lang in MODEL_LANG
    return _load_text_repr(lang, 'emojis',
                           emoji, d=d, func=func,
                           v1=v1)


def load_keyword(lang='es', keyword=None,
                 d=17, func='most_common_by_type',
                 v1=False):

    lang = lang.lower().strip()
    assert lang in MODEL_LANG
    return _load_text_repr(lang, 'keywords', 
                           keyword, d=d, func=func,
                           v1=v1)    


def emoji_information(lang='es'):
    """
    Download and load the Emoji statistics

    :param lang: ['ar', 'zh', 'en', 'fr', 'pt', 'ru', 'es']
    :type lang: str

    >>> from EvoMSA.utils import emoji_information
    >>> info = emoji_information()
    >>> info['💧']
    {'recall': 0.10575916230366492, 'ratio': 0.0003977123419509893, 'number': 3905}
    """
    from os.path import join, dirname, isdir, isfile
    from urllib.error import HTTPError    

    lang = lang.lower().strip()
    assert lang in MODEL_LANG
    diroutput = join(dirname(__file__), 'models')
    if not isdir(diroutput):
        os.mkdir(diroutput)
    data = []
    for ext in ['info', 'perf']:
        fname = join(diroutput, f'{lang}_emo.{ext}')
        if not isfile(fname):
            path = f'https://github.com/INGEOTEC/text_models/releases/download/models/{lang}_emo.{ext}'
            try:
                request.urlretrieve(path, fname)
            except HTTPError:
                raise Exception(path)    
        data.append(load_model(fname))
    uno, dos = data
    [v.update(dict(number=uno[k])) for k, v in dos.items()]
    return dos
    

def load_dataset(lang='es', name='HA', 
                 k=None, d=17, func='most_common_by_type',
                 v1=False):
    """
    Download and load the Dataset representation

    :param lang: ['ar', 'zh', 'en', 'es']
    :type lang: str
    :param emoji: emoji identifier
    :type emoji: int

    >>> from EvoMSA.utils import load_dataset, load_bow
    >>> bow = load_bow(lang='en')
    >>> ds = load_dataset(lang='en', name='travel', k=0)
    >>> X = bow.transform(['this is funny'])
    >>> df = ds.decision_function(X)
    """
    lang = lang.lower().strip()
    assert lang in ['ar', 'zh', 'en', 'es']
    return _load_text_repr(lang, name, 
                           k, d=d, func=func,
                           v1=v1)    


def dataset_information(lang='es'):
    """
    Download and load datasets information

    :param lang: ['ar', 'zh', 'en', 'es']
    :type lang: str

    >>> from EvoMSA.utils import emoji_information
    >>> info = dataset_information()
    """
    from os.path import join, dirname, isdir, isfile
    from urllib.error import HTTPError   

    lang = lang.lower().strip()
    assert lang in ['ar', 'zh', 'en', 'es']
    diroutput = join(dirname(__file__), 'models')
    if not isdir(diroutput):
        os.mkdir(diroutput)
    data = []
    ext = 'info'
    fname = join(diroutput, f'{lang}_dataset.{ext}')
    if not isfile(fname):
        path = f'https://github.com/INGEOTEC/text_models/releases/download/models/{lang}_dataset.{ext}'
        try:
            request.urlretrieve(path, fname)
        except HTTPError:
            raise Exception(path)    
    data = load_model(fname)
    return data


def b4msa_params(lang, dim=15):
    from microtc.params import OPTION_DELETE, OPTION_NONE    
    assert lang in MODEL_LANG
    tm_kwargs=dict(num_option=OPTION_NONE,
                   usr_option=OPTION_DELETE,
                   url_option=OPTION_DELETE, 
                   emo_option=OPTION_NONE,
                   hashtag_option=OPTION_NONE,
                   ent_option=OPTION_NONE,
                   lc=True, 
                   del_dup=False,
                   del_punc=True,
                   del_diac=True,
                   select_ent=False,
                   select_suff=False,
                   select_conn=False,
                   max_dimension=True,
                   unit_vector=True, 
                   token_max_filter=2**dim)
    if lang == 'ja' or lang == 'zh':
        tm_kwargs['token_list'] = [1, 2, 3]
    else:
        tm_kwargs['token_list'] = [-2, -1, 2, 3, 4]
    return tm_kwargs
    

