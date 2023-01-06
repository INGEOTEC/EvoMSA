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
from EvoMSA.utils import load_bow, load_emoji, dataset_information, load_dataset, load_keyword
from EvoMSA.model import GaussianBayes
from EvoMSA.model_selection import KruskalFS
from b4msa.textmodel import TextModel
from joblib import Parallel, delayed
from typing import Union, List, Set, Callable, Tuple
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from scipy.sparse import csr_matrix
from tqdm import tqdm
import numpy as np


class BoW(object):
    """
    >>> from microtc.utils import tweet_iterator
    >>> from EvoMSA.tests.test_base import TWEETS
    >>> from EvoMSA.evodag import BoW
    >>> bow = BoW(lang='es').fit(list(tweet_iterator(TWEETS)))
    >>> bow.predict(['Buenos dias']).tolist()
    ['P']
    """
    def __init__(self, lang: str='es', 
                 key: Union[str, List[str]]='text',
                 label_key: str='klass',
                 mixer_func: Callable[[List], csr_matrix]=sum,
                 decision_function: str='decision_function',
                 estimator_class=LinearSVC,
                 estimator_kwargs=dict(),
                 pretrain=True,
                 b4msa_kwargs=dict(),
                 kfold_instance=StratifiedKFold,
                 kfold_kwargs: dict=dict(random_state=0,
                                         shuffle=True),
                 n_jobs: int=1) -> None:
        assert lang is None or lang in MODEL_LANG
        self._n_jobs = n_jobs
        self._lang = lang
        self._key = key
        self._label_key = label_key
        self._mixer_func = mixer_func
        self._decision_function = decision_function
        self._estimator_class = estimator_class
        self._estimator_kwargs = estimator_kwargs
        self._b4msa_kwargs = b4msa_kwargs
        self._pretrain = pretrain
        self._kfold_instance = kfold_instance
        self._kfold_kwargs = kfold_kwargs
        self._b4msa_estimated = False

    @property
    def names(self):
        _names = [None] * len(self.bow.id2token)
        for k, v in self.bow.id2token.items():
            _names[k] = v
        return _names   

    @property
    def pretrain(self):
        return self._pretrain

    @property
    def lang(self):
        return self._lang

    def b4msa_fit(self, D):
        assert len(D)
        self._b4msa_estimated = True
        if self._key == 'text' or isinstance(D[0], str):
            return self.bow.fit(D)
        assert isinstance(D[0], dict)
        if isinstance(self._key, str):
            key = self._key
            return self.bow.fit([x[key] for x in D])
        _ = [[x[key] for key in self._key] for x in D]
        return self.bow.fit(_)

    def transform(self, D: List[Union[List, dict]], y=None) -> csr_matrix:
        assert len(D)
        if not self.pretrain:
            assert self._b4msa_estimated
        if self._key == 'text' or isinstance(D[0], str):
            return self.bow.transform(D)
        assert isinstance(D[0], dict)
        if isinstance(self._key, str):
            key = self._key
            return self.bow.transform([x[key] for x in D])
        Xs = [self.bow.transform([x[key] for x in D])
              for key in self._key]
        return self._mixer_func(Xs)

    @property
    def bow(self):
        try:
            bow = self._bow
        except AttributeError:
            if self.pretrain:
                self._bow = load_bow(lang=self._lang)
            else:
                self._bow = TextModel(lang=self.lang,
                                      **self._b4msa_kwargs)
            bow = self._bow
        return bow

    @bow.setter
    def bow(self, value):
        self._bow = value

    def dependent_variable(self, D: List[Union[dict, list]], 
                           y: Union[np.ndarray, None]=None) -> np.ndarray:
        assert isinstance(D, list) and len(D)
        label_key = self._label_key
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
            return getattr(m, self._decision_function)(X[vs])

        y = self.dependent_variable(D, y=y)
        kf = self._kfold_instance(**self._kfold_kwargs)
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
        hy = getattr(self.estimator_instance, self._decision_function)(_)
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
        assert emoji or dataset or keyword
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
    def text_representations(self):
        return self._text_representations

    @text_representations.setter
    def text_representations(self, value):
        self._text_representations = value

    def select(self, subset: Union[list, None]=None,
               D: List[Union[dict, list, None]]=None, 
               y: Union[np.ndarray, None]=None,
               feature_selection: Callable=KruskalFS,
               feature_selection_kwargs: dict=dict()) -> None:
        assert subset is not None or D is not None
        if subset is not None:
            tr = self.text_representations
            self.text_representations = [tr[i] for i in subset]
            names = self.names
            self.names = [names[i] for i in subset]
            return
        y = self.dependent_variable(D, y=y)
        X = self.transform(D)
        feature_selection = feature_selection(**feature_selection_kwargs).fit(X, y=y)
        index = feature_selection.get_support(indices=True)
        self.select(subset=index)

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        self._names = value

    def load_emoji(self) -> None:
        emojis = load_emoji(lang=self._lang)
        self.text_representations.extend(emojis)
        self.names.extend([x.labels[-1] for x in emojis])

    def load_keyword(self) -> None:
        _ = load_keyword(lang=self._lang)
        self.text_representations.extend(_)
        self.names.extend([x.labels[-1] for x in _])        

    def load_dataset(self) -> None:
        names = [name for name in dataset_information(lang=self._lang)
                 if name not in self._skip_dataset]
        _ = Parallel(n_jobs=self._n_jobs)(delayed(load_dataset)(lang=self._lang, name=name)
                                          for name in names)
        [self.text_representations.extend(k) for k in _]
        [self.names.extend([name] if len(k) == 1 else [f'{name}({i.labels[-1]})' for i in k])
         for k, name in zip(_, names)]        

    def transform(self, D: List[Union[List, dict]], y=None) -> np.ndarray:
        if isinstance(self._key, str):
            X = super(TextRepresentations, self).transform(D, y=y)
            models = Parallel(n_jobs=self._n_jobs)(delayed(m.decision_function)(X)
                                                   for m in self.text_representations)
            _ = np.array(models).T
            if self._unit_vector:
                return _ / np.atleast_2d(np.linalg.norm(_)).T
            else:
                return _
        assert len(D) and isinstance(D[0], dict)
        Xs = [super(TextRepresentations, self).transform([x[key] for x in D], y=y)
              for key in self._key]
        with Parallel(n_jobs=self._n_jobs) as parallel:
            models = []
            for X in Xs:
                _ = parallel(delayed(m.decision_function)(X)
                             for m in self.text_representations)
                models.append(np.array(_).T)
        _ = self._mixer_func(models)
        if self._unit_vector:
            return _ / np.atleast_2d(np.linalg.norm(_)).T
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
    >>> stacking = StackGeneralization(decision_function_models=[bow], transform_models=[dataset, emoji])
    >>> stacking.fit(list(tweet_iterator(TWEETS)))
    >>> stacking.predict(['Buenos dias']).tolist()
    ['P']
    """
    def __init__(self, decision_function_models: list=[], 
                 transform_models: list=[],
                 decision_function: str='predict_proba',
                 estimator_class=GaussianNB,
                 n_jobs: int=1,
                 **kwargs) -> None:
        assert len(decision_function_models) or len(transform_models)
        assert n_jobs == 1
        super(StackGeneralization, self).__init__(n_jobs=n_jobs,
                                                  decision_function=decision_function,
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

