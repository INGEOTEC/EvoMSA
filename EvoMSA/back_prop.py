# Copyright 2023 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union, List
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCSR
from sklearn.model_selection import StratifiedShuffleSplit
from IngeoML.utils import Batches
from IngeoML.optimizer import classifier
from EvoMSA.text_repr import BoW


@jax.jit
def bow_model(params, X):
    _ = X @ params['W'] + params['W0']
    # Y = _ / jnp.linalg.norm(_, axis=1, keepdims=True)
    # Y = Y @ params['W_cl'] + params['W0_cl']
    return _


class BoWBP(BoW):
    def __init__(self, voc_size_exponent: int=15,
                 estimator_kwargs=dict(dual=True, class_weight='balanced'),
                 deviation=None,
                 validation_set=None,
                 batches=None,
                 optimizer_kwargs: dict=dict(),
                 **kwargs):
        super(BoWBP, self).__init__(voc_size_exponent=voc_size_exponent,
                                    estimator_kwargs=estimator_kwargs)
        self.deviation = deviation
        self.validation_set = validation_set
        self.optimizer_kwargs = optimizer_kwargs
        self.batches = batches

    @property
    def batches(self):
        """Instance to create the batches"""
        return self._batches
    
    @batches.setter
    def batches(self, value):
        self._batches = value

    @property
    def optimizer_kwargs(self):
        """Arguments for the optimizer"""
        return self._optimizer_kwargs

    @optimizer_kwargs.setter
    def optimizer_kwargs(self, value):
        self._optimizer_kwargs = value

    @property
    def validation_set(self):
        """Validation set"""
        return self._validation_set
    
    @validation_set.setter
    def validation_set(self, value):
        if value is None:
            self._validation_set = value
            return
        if hasattr(value, 'split'):
            self._validation_set = value
            return
        assert isinstance(value, list) and len(value)
        if isinstance(value[0], dict):
            y = self.dependent_variable(value)
            X = self.transform(value)
            self._validation_set = [X, y]
        else:
            X, y = value
            self._validation_set = [self.transform(X), y]

    @property
    def deviation(self):
        """Function to measure the deviation between the true observations and the predictions."""
        return self._deviation
    
    @deviation.setter
    def deviation(self, value):
        self._deviation = value
        
    @property
    def parameters(self):
        """Parameter to optimize"""

        try:
            return self._parameters
        except AttributeError:
            W = jnp.array(self.estimator_instance.coef_.T)
            W0 = jnp.array(self.estimator_instance.intercept_)
            self._parameters = dict(W=W, W0=W0)
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        W = np.array(value['W'].T)
        W0 = np.array(value['W0'])
        self.estimator_instance.coef_ = W
        self.estimator_instance.intercept_ = W0

    def fit(self, D: List[Union[dict, list]], 
            y: Union[np.ndarray, None]=None) -> 'BoWBP':        
        super(BoWBP, self).fit(D, y=y)
        remainder = 'drop'
        if self.validation_set is None:
            if len(D) < 2048:
                self.validation_set = StratifiedShuffleSplit(n_splits=1,
                                                             test_size=0.2)
                remainder = 'fill'
        decoder = self.estimator_instance.classes_
        n_outputs = 1 if decoder.shape[0] == 2 else decoder.shape[0]
        if self.batches is None:
            self.batches = Batches(size=512 if len(D) >= 2048 else 256,
                                   random_state=0,
                                   remainder=remainder)
        optimizer_defaults = dict(array=BCSR.from_scipy_sparse,
                                  every_k_schedule=4,
                                  epochs=100, learning_rate=1e-4,
                                  n_iter_no_change=5)
        optimizer_defaults.update(self.optimizer_kwargs)
        texts = self.transform(D)
        labels = self.dependent_variable(D, y=y)
        p = classifier(self.parameters, bow_model,
                       texts, labels,
                       deviation=self.deviation, n_outputs=n_outputs,
                       validation=self.validation_set,
                       batches=self.batches, **optimizer_defaults)
        self.parameters = p
        return self        