# Copyright 2022 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from EvoMSA.base import DEFAULT_CL, DEFAULT_R
from EvoMSA.utils import MODEL_LANG
from EvoMSA.utils import load_bow, load_emoji, dataset_information, load_dataset, load_keyword, b4msa_params
from EvoMSA.model import GaussianBayes
from EvoMSA.model_selection import KruskalFS
from b4msa.textmodel import TextModel
from microtc.weighting import TFIDF
from joblib import Parallel, delayed
from typing import Union, List, Set, Callable, Tuple
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from scipy.sparse import csr_matrix
from tqdm import tqdm
import numpy as np


def config_regressor(instance):
    from sklearn.svm import LinearSVR
    from sklearn.model_selection import KFold

    instance.estimator_class = LinearSVR
    instance.estimator_kwargs = dict()
    instance.decision_function_name = 'predict'
    instance.kfold_class = KFold
    return instance


class BoW(object):
    """
    BoW is a bag-of-words text classifier. It is described in 
    "A Simple Approach to Multilingual Polarity Classification in Twitter. 
    Eric S. Tellez, Sabino Miranda-Jiménez, Mario Graff, 
    Daniela Moctezuma, Ranyart R. Suárez, Oscar S. Siordia. 
    Pattern Recognition Letters" and 
    "An Automated Text Categorization Framework based 
    on Hyperparameter Optimization. Eric S. Tellez, Daniela Moctezuma, 
    Sabino Miranda-Jímenez, Mario Graff. 
    Knowledge-Based Systems Volume 149, 1 June 2018." 

    BoW uses, by default, a pre-trained bag-of-words representation. The 
    representation was trained on 524,288 (:math:`2^{19}`) tweets
    randomly selected. The pre-trained representations are used
    when the parameters :attr:`lang` and :attr:`pretrain` are
    set; :attr:`pretrain` by default is set to True, and the default 
    language is Spanish (es). The available languages are:
    Arabic (ar), Catalan (ca), German (de), English (en), 
    Spanish (es), French (fr), Hindi (hi), Indonesian (in), 
    Italian (it), Japanese (ja), Korean (ko), Dutch (nl),
    Polish (pl), Portuguese (pt), Russian (ru), Tagalog (tl), 
    Turkish (tr), and Chinese (zh).
    

    :param lang: Language. (ar | ca | de | en | es | fr | hi | in | it | ja | ko | nl | pl | pt | ru | tl | tr | zh), default='es'.
    :type lang: str
    :param pretrain: Whether to use a pre-trained representation. default=True.
    :type pretrain: bool
    :param b4msa_kwargs: :py:class:`b4msa.textmodel.TextModel` keyword arguments used to train a bag-of-words representation. default=dict().
    :type b4msa_kwargs: dict
    :param estimator_class:
    :type estimator_class: class
    :param estimator_kwargs: 
    :type estimator_kwargs: dict
    :param key:
    :type key: Union[str, List[str]]
    :param label_key:
    :type label_key: str
    :param mixer_func:
    :type mixer_func: Callable[[List], csr_matrix]
    :param decision_function:
    :type decision_function: str
    :param kfold_class:
    :type kfold_class: class
    :param kfold_kwargs:
    :type kfold_kwargs: dict
    :param n_jobs: Number of jobs. default=1
    :type n_jobs: int

    >>> from microtc.utils import tweet_iterator
    >>> from EvoMSA.tests.test_base import TWEETS
    >>> from EvoMSA import BoW
    >>> bow = BoW(lang='es').fit(list(tweet_iterator(TWEETS)))
    >>> bow.predict(['Buenos dias']).tolist()
    ['P']
    """
    def __init__(self, lang: str='es',
                 voc_size_exponent: int=17,
                 voc_selection: str='most_common_by_type', 
                 key: Union[str, List[str]]='text',
                 label_key: str='klass',
                 mixer_func: Callable[[List], csr_matrix]=sum,
                 decision_function_name: str='decision_function',
                 estimator_class=LinearSVC,
                 estimator_kwargs=dict(),
                 pretrain=True,
                 b4msa_kwargs=dict(),
                 kfold_class=StratifiedKFold,
                 kfold_kwargs: dict=dict(random_state=0,
                                         shuffle=True),
                 v1: bool=False,
                 n_jobs: int=1) -> None:
        assert lang is None or lang in MODEL_LANG
        if lang in MODEL_LANG:
            assert voc_size_exponent >= 13 and voc_size_exponent <= 17
            assert voc_selection in ['most_common_by_type', 'most_common']
        self.voc_size_exponent = voc_size_exponent
        self.voc_selection = voc_selection
        self._n_jobs = n_jobs
        self._lang = lang
        self.key = key
        self.label_key = label_key
        self._mixer_func = mixer_func
        self.decision_function_name = decision_function_name
        self.estimator_class = estimator_class
        self.estimator_kwargs = estimator_kwargs
        self._b4msa_kwargs = b4msa_kwargs
        self._pretrain = pretrain
        self.kfold_class = kfold_class
        self.kfold_kwargs = kfold_kwargs
        self._b4msa_estimated = False
        self.v1 = v1

    @property
    def cache(self):
        try:
            return self._cache
        except AttributeError:
            return None
        
    @cache.setter
    def cache(self, value):
        self._cache = value

    @property
    def v1(self):
        return self._v1
    
    @v1.setter
    def v1(self, value):
        self._v1 = value

    @property
    def voc_selection(self):
        return self._voc_selection
    
    @voc_selection.setter
    def voc_selection(self, value):
        self._voc_selection = value

    @property
    def voc_size_exponent(self):
        return self._voc_size_exponent
    
    @voc_size_exponent.setter
    def voc_size_exponent(self, value):
        self._voc_size_exponent = value

    @property
    def label_key(self):
        return self._label_key

    @label_key.setter
    def label_key(self, value):
        self._label_key = value

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        self._key = value

    @property
    def decision_function_name(self):
        return self._decision_function

    @decision_function_name.setter
    def decision_function_name(self, value):
        self._decision_function = value

    @property
    def names(self):
        _names = [None] * len(self.bow.id2token)
        for k, v in self.bow.id2token.items():
            _names[k] = v
        return _names

    @property
    def weights(self):
        try:
            return self._weights
        except AttributeError:
            w = [None] * len(self.bow.token_weight)
            for k, v in self.bow.token_weight.items():
                w[k] = v
            self._weights = w
            return self._weights

    @property
    def pretrain(self):
        return self._pretrain

    @property
    def lang(self):
        return self._lang

    @property
    def kfold_class(self):
        return self._kfold_instance

    @kfold_class.setter
    def kfold_class(self, value):
        self._kfold_instance = value

    @property
    def kfold_kwargs(self):
        return self._kfold_kwargs

    @kfold_kwargs.setter
    def kfold_kwargs(self, value):
        self._kfold_kwargs = value

    @property
    def bow(self):
        try:
            bow = self._bow
        except AttributeError:
            if self.pretrain:
                if self.v1:
                    self._bow = load_bow(lang=self.lang, v1=self.v1)
                else:
                    freq = load_bow(lang=self.lang,
                                    d=self.voc_size_exponent, 
                                    func=self.voc_selection)
                    params = b4msa_params(lang=self.lang,
                                        dim=self._voc_size_exponent)
                    params.update(self._b4msa_kwargs)
                    bow = TextModel(**params)
                    tfidf = TFIDF()
                    tfidf.N = freq.update_calls
                    tfidf.word2id, tfidf.wordWeight = tfidf.counter2weight(freq)
                    bow.model = tfidf
                    self._bow = bow
            else:
                self._bow = TextModel(lang=self.lang,
                                      **self._b4msa_kwargs)
            bow = self._bow
        return bow

    @bow.setter
    def bow(self, value):
        self._bow = value

    @property
    def estimator_class(self):
        return self._estimator_class

    @estimator_class.setter
    def estimator_class(self, value):
        self._estimator_class = value

    @property
    def estimator_kwargs(self):
        return self._estimator_kwargs

    @estimator_kwargs.setter
    def estimator_kwargs(self, value):
        self._estimator_kwargs = value        

    def b4msa_fit(self, D):
        assert len(D)
        self._b4msa_estimated = True
        if self.key == 'text' or isinstance(D[0], str):
            return self.bow.fit(D)
        assert isinstance(D[0], dict)
        if isinstance(self.key, str):
            key = self.key
            return self.bow.fit([x[key] for x in D])
        _ = [[x[key] for key in self.key] for x in D]
        return self.bow.fit(_)

    def transform(self, D: List[Union[List, dict]], y=None) -> csr_matrix:
        assert len(D)
        if not self.pretrain:
            assert self._b4msa_estimated
        if self.pretrain and self.cache is not None:
            X = self.cache
            self.cache = None
            return X
        if self.key == 'text' or isinstance(D[0], str):
            return self.bow.transform(D)
        assert isinstance(D[0], dict)
        if isinstance(self.key, str):
            key = self.key
            return self.bow.transform([x[key] for x in D])
        Xs = [self.bow.transform([x[key] for x in D])
              for key in self.key]
        return self._mixer_func(Xs)

    def dependent_variable(self, D: List[Union[dict, list]], 
                           y: Union[np.ndarray, None]=None) -> np.ndarray:
        assert isinstance(D, list) and len(D)
        label_key = self.label_key
        if y is None:
            assert isinstance(D[0], dict)
            y = np.array([x[label_key] for x in D])
        assert isinstance(y, np.ndarray)
        return y

    def estimator(self):
        return self._estimator_class(**self._estimator_kwargs)

    @property
    def estimator_instance(self):
        return self._m

    @estimator_instance.setter
    def estimator_instance(self, m):
        self._m = m

    def train_predict_decision_function(self, D: List[Union[dict, list]], 
                                        y: Union[np.ndarray, None]=None) -> np.ndarray:
        def train_predict(tr, vs):
            m = self.estimator().fit(X[tr], y[tr])
            return getattr(m, self.decision_function_name)(X[vs])

        y = self.dependent_variable(D, y=y)
        kf = self.kfold_class(**self.kfold_kwargs)
        kfolds = [x for x in kf.split(D, y)]
        X = self.transform(D, y=y)
        hys = Parallel(n_jobs=self._n_jobs)(delayed(train_predict)(tr, vs)
                                            for tr, vs in kfolds)
        K = np.unique(y).shape[0]
        if hys[0].ndim == 1:
            hy = np.empty((y.shape[0], 1))
            hys = [np.atleast_2d(x).T for x in hys]
        else:
            hy = np.empty((y.shape[0], K))
        for (_, vs), pr in zip(kfolds, hys):
            hy[vs] = pr
        return hy

    def fit(self, D: List[Union[dict, list]], 
            y: Union[np.ndarray, None]=None) -> 'BoW':
        if not self.pretrain and not self._b4msa_estimated:
            self.b4msa_fit(D)
        y = self.dependent_variable(D, y=y)
        _ = self.transform(D, y=y)
        self.estimator_instance = self.estimator().fit(_, y)
        return self

    def predict(self, D: List[Union[dict, list]]) -> np.ndarray:
        _ = self.transform(D)
        return self.estimator_instance.predict(_)

    def decision_function(self, D: List[Union[dict, list]]) -> Union[list, np.ndarray]:
        _ = self.transform(D)
        hy = getattr(self.estimator_instance, self.decision_function_name)(_)
        if hy.ndim == 1:
            return np.atleast_2d(hy).T
        return hy


class TextRepresentations(BoW):
    """
    >>> from EvoMSA.evodag import TextRepresentations
    >>> from microtc.utils import tweet_iterator
    >>> from EvoMSA.tests.test_base import TWEETS    
    >>> text_repr =  TextRepresentations(lang='es')
    >>> text_repr.fit(list(tweet_iterator(TWEETS)))
    >>> text_repr.predict(['Buenos dias']).tolist()
    ['P']    
    """
    def __init__(self, 
                 emoji: bool=True,
                 dataset: bool=True,
                 keyword: bool=True,
                 skip_dataset: Set[str]=set(),
                 estimator_kwargs=dict(dual=False),
                 unit_vector=True,
                 **kwargs) -> None:
        super(TextRepresentations, self).__init__(estimator_kwargs=estimator_kwargs, **kwargs)
        self._skip_dataset = skip_dataset
        self._names = []
        self._text_representations = []
        self._unit_vector = unit_vector
        if emoji:
            self.load_emoji()
        if dataset:
            self.load_dataset()
        if keyword:
            self.load_keyword()

    @property
    def weights(self):
        try:
            return self._weights
        except AttributeError:
            w = np.array([x._coef for x in self.text_representations])
            self._weights = w
            return self._weights

    @property
    def bias(self):
        try:
            return self._bias
        except AttributeError:
            w = np.array([x._intercept for x in self.text_representations])
            self._bias = w
            return self._bias       

    @property
    def text_representations(self):
        return self._text_representations

    @text_representations.setter
    def text_representations(self, value):
        self._text_representations = value

    def text_representations_extend(self, value):
        names = set(self.names)
        for x in value:
            label = x.labels[-1]
            if label not in names:
                self.text_representations.append(x)
                self.names.append(label)
                names.add(label)

    def select(self, subset: Union[list, None]=None,
               D: List[Union[dict, list, None]]=None, 
               y: Union[np.ndarray, None]=None,
               feature_selection: Callable=KruskalFS,
               feature_selection_kwargs: dict=dict()) -> 'TextRepresentations':
        assert subset is not None or D is not None
        if subset is not None:
            tr = self.text_representations
            self.text_representations = [tr[i] for i in subset]
            names = self.names
            self.names = [names[i] for i in subset]
            return self
        y = self.dependent_variable(D, y=y)
        X = self.transform(D)
        feature_selection = feature_selection(**feature_selection_kwargs).fit(X, y=y)
        index = feature_selection.get_support(indices=True)
        return self.select(subset=index)

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        self._names = value

    def load_emoji(self) -> None:
        if self.v1:
            emojis = load_emoji(lang=self.lang, v1=self.v1)
            self.text_representations.extend(emojis)
            self.names.extend([x.labels[-1] for x in emojis])
        else:
            data = load_emoji(lang=self.lang,
                              d=self.voc_size_exponent, 
                              func=self.voc_selection)
            self.text_representations.extend(data)
            self.names.extend([x.labels[-1] for x in data])            

    def load_keyword(self) -> None:
        if self.v1:
            _ = load_keyword(lang=self.lang, v1=self.v1)
            self.text_representations.extend(_)
            self.names.extend([x.labels[-1] for x in _])
        else:       
            data = load_keyword(lang=self.lang,
                                d=self.voc_size_exponent, 
                                func=self.voc_selection)
            self.text_representations.extend(data)
            self.names.extend([x.labels[-1] for x in data])            

    def load_dataset(self) -> None:
        if self.v1:
            names = [name for name in dataset_information(lang=self.lang)
                    if name not in self._skip_dataset]
            _ = Parallel(n_jobs=self._n_jobs)(delayed(load_dataset)(lang=self.lang, name=name, v1=self.v1)
                                            for name in names)
            [self.text_representations.extend(k) for k in _]
            [self.names.extend([name] if len(k) == 1 else [f'{name}({i.labels[-1]})' for i in k])
            for k, name in zip(_, names)]
        else:
            data = load_dataset(lang=self.lang, name='datasets',
                                d=self.voc_size_exponent, 
                                func=self.voc_selection)
            self.text_representations.extend(data)
            self.names.extend([x.labels[-1] for x in data])

    def transform(self, D: List[Union[List, dict]], y=None) -> np.ndarray:
        if isinstance(self.key, str):
            X = super(TextRepresentations, self).transform(D, y=y)
            models = Parallel(n_jobs=self._n_jobs)(delayed(m.decision_function)(X)
                                                   for m in self.text_representations)
            _ = np.array(models).T
            if self._unit_vector:
                return _ / np.atleast_2d(np.linalg.norm(_, axis=1)).T
            else:
                return _
        assert len(D) and isinstance(D[0], dict)
        Xs = [super(TextRepresentations, self).transform([x[key] for x in D], y=y)
              for key in self.key]
        with Parallel(n_jobs=self._n_jobs) as parallel:
            models = []
            for X in Xs:
                _ = parallel(delayed(m.decision_function)(X)
                             for m in self.text_representations)
                models.append(np.array(_).T)
        _ = self._mixer_func(models)
        if self._unit_vector:
            return _ / np.atleast_2d(np.linalg.norm(_, axis=1)).T
        else:
            return _

class StackGeneralization(BoW):
    """
    >>> from EvoMSA.evodag import TextRepresentations, BoW, StackGeneralization
    >>> from microtc.utils import tweet_iterator
    >>> from EvoMSA.tests.test_base import TWEETS    
    >>> emoji =  TextRepresentations(lang='es', dataset=False)
    >>> dataset = TextRepresentations(lang='es', emoji=False)
    >>> bow = BoW(lang='es')
    >>> stacking = StackGeneralization(decision_function_models=[bow],
                                       transform_models=[dataset, emoji])
    >>> stacking.fit(list(tweet_iterator(TWEETS)))
    >>> stacking.predict(['Buenos dias']).tolist()
    ['P']
    """
    def __init__(self, decision_function_models: list=[], 
                 transform_models: list=[],
                 decision_function_name: str='predict_proba',
                 estimator_class=GaussianNB,
                 n_jobs: int=1,
                 **kwargs) -> None:
        assert len(decision_function_models) or len(transform_models)
        assert n_jobs == 1
        super(StackGeneralization, self).__init__(n_jobs=n_jobs,
                                                  decision_function_name=decision_function_name,
                                                  estimator_class=estimator_class,
                                                  **kwargs)
        self._decision_function_models = decision_function_models
        self._transform_models = transform_models
        self._estimated = False

    def fit(self, *args, **kwargs) -> 'StackGeneralization':
        super(StackGeneralization, self).fit(*args, **kwargs)
        self._estimated = True
        return self

    def transform(self, D: List[Union[List, dict]], y=None) -> np.ndarray:
        Xs = [text_repr.transform(D)
              for text_repr in self._transform_models]
        if not self._estimated:
            [text_repr.fit(D, y=y)
             for text_repr in self._decision_function_models]
            Xs += [text_repr.train_predict_decision_function(D, y=y)
                   for text_repr in self._decision_function_models]
            return np.concatenate(Xs, axis=1)
        Xs += [text_repr.decision_function(D)
               for text_repr in self._decision_function_models]
        return np.concatenate(Xs, axis=1)

    def train_predict_decision_function(self, *args, **kwargs) -> np.ndarray:
        assert not self._estimated
        return super(StackGeneralization, self).train_predict_decision_function(*args, **kwargs)


class EvoDAG(object):
    def __init__(self, n_estimators: int=30,
                 max_training_size: int=4000,
                 n_jobs = 1,
                 random_state: int=0,
                 **kwargs) -> None:
        self._st_kwargs = kwargs
        self._n_estimators = n_estimators
        self._max_training_size = max_training_size
        self._n_jobs = n_jobs
        self._random_state = random_state

    def estimator(self, **kwargs):
        from EvoDAG.model import EvoDAG as model
        _ = DEFAULT_CL.copy()
        _.update(**self._st_kwargs)
        _.update(**kwargs)
        return model(**_)

    @property
    def estimator_instance(self):
        return self._m

    @estimator_instance.setter
    def estimator_instance(self, m):
        self._m = m

    def dependent_variable(self, y: np.ndarray) -> np.ndarray:
        from EvoMSA.utils import LabelEncoderWrapper
        if not hasattr(self, '_le'):
            self._le = LabelEncoderWrapper().fit(y)
            return self._le.transform(y)
        return y

    @property
    def label_encoder(self):
        return self._le

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EvoDAG':
        def _fit(tr, **kwargs):
            return self.estimator(**kwargs).fit(X[tr], y[tr])

        y = self.dependent_variable(y) 
        if X.shape[0] > self._max_training_size:
            index = [tr for tr, _ in StratifiedShuffleSplit(n_splits=self._n_estimators,
                                                            random_state=self._random_state,
                                                            train_size=self._max_training_size).split(X, y)]
        else:
            index = [np.arange(X.shape[0])] * self._n_estimators
        _ = Parallel(n_jobs=self._n_jobs)(delayed(_fit)(tr,
                                                        seed=seed + self._random_state)
                                          for seed, tr in enumerate(index))
        self.estimator_instance = _
        return self

    def _decision_function(self, X: np.ndarray) -> list:
        def _decision_function(ins, X):
            return np.array([x.full_array() 
                             for x in ins.decision_function(X)]).T

        _ = Parallel(n_jobs=self._n_jobs)(delayed(_decision_function)(ins, X)
                                          for ins in self.estimator_instance)
        return _

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        _ = self._decision_function(X)
        return np.median(_, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        hy = df.argmax(axis=1)
        return self.label_encoder.inverse_transform(hy)

