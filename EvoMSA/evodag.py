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
from EvoMSA.utils import load_bow, load_emoji, emoji_information, dataset_information, load_dataset
from joblib import Parallel, delayed
from typing import Union, List, Set
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import numpy as np


class BoW(object):
    def __init__(self, lang='es', random_state=0, n_jobs=1) -> None:
        assert lang in MODEL_LANG
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._lang = lang

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
                                        y: Union[np.ndarray, None]=None) -> Union[List[np.ndarray], np.ndarray]:
        def train_predict(tr, vs):
            m = self.estimator().fit(X[tr], y[tr])
            return m.decision_function(X[vs])

        y = self.dependent_variable(D, y=y)
        kf = StratifiedKFold(shuffle=True, random_state=self._random_state)
        kfolds = [x for x in kf.split(D, y)]
        X = self.bow.transform(D)
        hys = Parallel(n_jobs=self._n_jobs)(delayed(train_predict)(tr, vs)
                                            for tr, vs in kfolds)
        K = np.unique(y).shape[0] 
        if K > 2:
            hy = np.empty((y.shape[0], K))
        else:
            hy = np.empty(y.shape[0])            
        for (_, vs), pr in zip(kfolds, hys):
            hy[vs] = pr
        if hy.ndim == 2:
            return [x.copy() for x in hy.T]
        return hy

    def fit(self, D: List[Union[dict, list]], 
            y: Union[np.ndarray, None]=None) -> "BoW":
        y = self.dependent_variable(D, y=y)
        _ = self.bow.transform(D)
        self.estimator_instance = self.estimator().fit(_, y)
        return self

    def predict(self, D: List[Union[dict, list]]) -> np.ndarray:
        _ = self.bow.transform(D)
        return self.estimator_instance.predict(_)

    def decision_function(self, D: List[Union[dict, list]]) -> Union[list, np.ndarray]:
        _ = self.bow.transform(D)
        hy = self.estimator_instance.decision_function(_)        
        if hy.ndim == 2:
            return [x.copy() for x in hy.T]
        return hy


class EvoDAG(BoW):
    def __init__(self, TR: bool=True, 
                 n_estimators: int=30,
                 max_training_size = 4000,
                 st_kwargs: dict=dict(),
                 skip_dataset: Set[str]=set(),
                 emoji: bool=True,
                 dataset: bool=True,
                 *args, **kwargs) -> None:
        super(EvoDAG, self).__init__(*args, **kwargs)
        assert emoji or dataset
        self._TR = TR
        self._st_kwargs = st_kwargs
        self._n_estimators = n_estimators
        self._max_training_size = max_training_size
        self._skip_dataset = skip_dataset
        self._models = []
        if emoji:
            self.load_emoji()
        if dataset:
            self.load_dataset()

    @property
    def models(self):
        return self._models

    def load_emoji(self) -> None:
        self._models += load_emoji(lang=self._lang)

    def load_dataset(self) -> None:
        _ = Parallel(n_jobs=self._n_jobs)(delayed(load_dataset)(lang=self._lang, name=name)
                                          for name in dataset_information(lang=self._lang) if name not in self._skip_dataset)
        [self._models.extend(k) for k in _]

    def stack_generalization(self, **kwargs):
        from EvoDAG.model import EvoDAG as model
        _ = DEFAULT_CL.copy()
        _.update(**self._st_kwargs)
        _.update(**kwargs)
        return model(**_)

    @property
    def stack_generalization_instance(self):
        return self._m_st

    @stack_generalization_instance.setter
    def stack_generalization_instance(self, v):
        self._m_st = v

    def dependent_variable(self, D: List[Union[dict, list]], 
                           y: Union[np.ndarray, None]=None) -> np.ndarray:
        from EvoMSA.utils import LabelEncoderWrapper
        y = super(EvoDAG, self).dependent_variable(D, y=y)
        if not hasattr(self, '_le'):
            self._le = LabelEncoderWrapper().fit(y)
            return self._le.transform(y)
        return y

    @property
    def label_encoder(self):
        return self._le

    def transform_models(self, D: List[Union[dict, list]]) -> List[np.ndarray]:
        X = self.bow.transform(D)
        models = Parallel(n_jobs=self._n_jobs)(delayed(m.decision_function)(X)
                                               for m in self.models)
        return models

    def _fit(self, X: np.ndarray, y: np.ndarray):
        def _fit(tr, **kwargs):
            return self.stack_generalization(**kwargs).fit(X[tr], y[tr])

        if X.shape[0] > self._max_training_size:
            index = [tr for tr, _ in StratifiedShuffleSplit(n_splits=self._n_estimators,
                                                            random_state=self._random_state,
                                                            train_size=self._max_training_size).split(X, y)]
        else:
            index = [np.arange(X.shape[0])] * self._n_estimators
        _ = Parallel(n_jobs=self._n_jobs)(delayed(_fit)(tr,
                                                        seed=seed + self._random_state)
                                          for seed, tr in enumerate(index))
        return _

    def fit(self, D: List[Union[dict, list]], 
            y: Union[np.ndarray, None]=None) -> 'EvoDAG':
            
        y = self.dependent_variable(D, y=y)
        super(EvoDAG, self).fit(D, y)
        models = self.transform_models(D)
        if self._TR:
            _ = self.train_predict_decision_function(D, y=y)
            if isinstance(_, list):
                models = _ + models
            else:
                models.insert(0, _)
        X = np.array(models).T
        self.stack_generalization_instance = self._fit(X, y)
        return self

    def transform(self, D: List[Union[dict, list]]) -> np.ndarray:
        if self._TR:
            _ = super(EvoDAG, self).decision_function(D)
            _ = _ if isinstance(_, list) else [ _ ]
        else:
            _ = []
        models = _ + self.transform_models(D)
        return np.array(models).T

    def _decision_function(self, D: List[Union[dict, list]]) -> list:
        def _decision_function(ins, X):
            return np.array([x.full_array() 
                             for x in ins.decision_function(X)]).T

        X = self.transform(D)
        _ = Parallel(n_jobs=self._n_jobs)(delayed(_decision_function)(ins, X)
                                          for ins in self._m_st)
        return _

    def decision_function(self, D: List[Union[dict, list]]) -> np.ndarray:
        _ = self._decision_function(D)
        return np.median(_, axis=0)

    def predict(self, D: List[Union[dict, list]]) -> np.ndarray:
        df = self.decision_function(D)
        hy = df.argmax(axis=1)
        return self.label_encoder.inverse_transform(hy)

