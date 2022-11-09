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
from EvoMSA.utils import load_bow, load_emoji, dataset_information, load_dataset
from joblib import Parallel, delayed
from typing import Union, List, Set, Callable
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from scipy.sparse import csr_matrix
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
                 random_state: int=0,
                 key: Union[str, List[str]]='text',
                 mixer_func: Callable[[List], csr_matrix]=sum,
                 decision_function: str='decision_function',
                 n_jobs: int=1) -> None:
        assert lang in MODEL_LANG
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._lang = lang
        self._key = key
        self._mixer_func = mixer_func
        self._decision_function = decision_function

    def transform(self, D: List[Union[List, dict]]) -> csr_matrix:
        assert len(D)
        if self._key == 'text':
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
            self._bow = load_bow(lang=self._lang)
            bow = self._bow
        return bow

    def dependent_variable(self, D: List[Union[dict, list]], 
                           y: Union[np.ndarray, None]=None) -> np.ndarray:
        assert isinstance(D, list) and len(D)
        if y is None:
            assert isinstance(D[0], dict)
            y = np.array([x['klass'] for x in D])
        assert isinstance(y, np.ndarray)
        return y

    def estimator(self):
        from sklearn.svm import LinearSVC
        return LinearSVC()

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
        self._y = y
        kf = StratifiedKFold(shuffle=True, random_state=self._random_state)
        kfolds = [x for x in kf.split(D, y)]
        X = self.transform(D)
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
        delattr(self, '_y')        
        return hy

    def fit(self, D: List[Union[dict, list]], 
            y: Union[np.ndarray, None]=None) -> 'BoW':
        y = self.dependent_variable(D, y=y)
        self._y = y
        _ = self.transform(D)
        self.estimator_instance = self.estimator().fit(_, y)
        delattr(self, '_y')
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
                 skip_dataset: Set[str]=set(),
                 decision_function='predict_proba',
                 **kwargs) -> None:
        super(TextRepresentations, self).__init__(decision_function=decision_function,
                                                  **kwargs)
        assert emoji or dataset
        self._skip_dataset = skip_dataset
        self._text_representations = []
        if emoji:
            self.load_emoji()
        if dataset:
            self.load_dataset()

    @property
    def text_representations(self):
        return self._text_representations

    def load_emoji(self) -> None:
        self._text_representations += load_emoji(lang=self._lang)

    def load_dataset(self) -> None:
        _ = Parallel(n_jobs=self._n_jobs)(delayed(load_dataset)(lang=self._lang, name=name)
                                          for name in dataset_information(lang=self._lang) if name not in self._skip_dataset)
        [self._text_representations.extend(k) for k in _]

    def estimator(self):
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()

    def transform(self, D: List[Union[List, dict]]) -> np.ndarray:
        if isinstance(self._key, str):
            X = super(TextRepresentations, self).transform(D)
            models = Parallel(n_jobs=self._n_jobs)(delayed(m.decision_function)(X)
                                                   for m in self.text_representations)
            return np.array(models).T
        assert len(D) and isinstance(D[0], dict)
        Xs = [self.bow.transform([x[key] for x in D])
              for key in self._key]
        with Parallel(n_jobs=self._n_jobs) as parallel:
            models = []
            for X in Xs:
                _ = parallel(delayed(m.decision_function)(X)
                             for m in self.text_representations)
                models.append(np.array(_).T)
        return self._mixer_func(models)                


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
                 n_jobs: int=1,
                 **kwargs) -> None:
        assert len(decision_function_models) or len(transform_models)
        assert n_jobs == 1
        super(StackGeneralization, self).__init__(n_jobs=n_jobs,
                                                  decision_function=decision_function,
                                                  **kwargs)
        self._decision_function_models = decision_function_models
        self._transform_models = transform_models
        self._estimated = False

    def estimator(self):
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()

    def fit(self, *args, **kwargs) -> 'StackGeneralization':
        super(StackGeneralization, self).fit(*args, **kwargs)
        self._estimated = True
        return self

    def transform(self, D: List[Union[List, dict]]) -> np.ndarray:
        Xs = [text_repr.transform(D)
              for text_repr in self._transform_models]
        if not self._estimated:
            [text_repr.fit(D, y=self._y)
             for text_repr in self._decision_function_models]
            Xs += [text_repr.train_predict_decision_function(D, y=self._y)
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

