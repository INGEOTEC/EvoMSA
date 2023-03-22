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


from EvoMSA.base import DEFAULT_CL
from joblib import Parallel, delayed
from typing import List, Set
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


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

