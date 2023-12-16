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
from EvoMSA.text_repr import BoW, DenseBoW


@jax.jit
def bow_model(params, X):
    Y = X @ params['W_cl'] + params['W0_cl']
    return Y


@jax.jit
def dense_model(params, X):
    _ = X @ params['W'] + params['W0']
    Y = _ / jnp.linalg.norm(_, axis=1, keepdims=True)
    Y = Y @ params['W_cl'] + params['W0_cl']
    return Y


class BoWBP(BoW):
    def __init__(self, voc_size_exponent: int=15,
                 estimator_kwargs=dict(dual=True, class_weight='balanced'),
                 deviation=None,
                 validation_set=None,
                 batches=None,
                 optimizer_kwargs: dict=dict(),
                 **kwargs):
        super(BoWBP, self).__init__(voc_size_exponent=voc_size_exponent,
                                    estimator_kwargs=estimator_kwargs, **kwargs)
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
    
    def _transform(self, X):
        return super(BoWBP, self).transform(X)
    
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
            X = self._transform(value)
            self._validation_set = [X, y]
        else:
            X, y = value
            self._validation_set = [self._transform(X), y]

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
            self._parameters = dict(W_cl=W, W0_cl=W0)
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        self.estimator_instance.coef_ = np.array(value['W_cl'].T)
        self.estimator_instance.intercept_ = np.array(value['W0_cl'])

    @property
    def model(self):
        return bow_model

    def set_validation_set(self, D: List[Union[dict, list]], 
                           y: Union[np.ndarray, None]=None):
        """Procedure to create the validation set"""
        if self.validation_set is None:
            if len(D) < 2048:
                test_size = 0.2
            else:
                test_size = 512
            y = self.dependent_variable(D, y=y)
            _ = StratifiedShuffleSplit(n_splits=1,
                                       test_size=test_size).split(D, y)
            tr, vs = next(_)
            self.validation_set = [D[x] for x in vs]
            D = [D[x] for x in tr]
            y = y[tr]
        return D, y
    
    def combine_optimizer_kwargs(self):
        decoder = self.estimator_instance.classes_
        n_outputs = 1 if decoder.shape[0] == 2 else decoder.shape[0]
        optimizer_defaults = dict(array=BCSR.from_scipy_sparse,
                                  every_k_schedule=4, n_outputs=n_outputs,
                                  epochs=100, learning_rate=1e-4,
                                  n_iter_no_change=5)
        optimizer_defaults.update(self.optimizer_kwargs)        
        return optimizer_defaults

    def fit(self, D: List[Union[dict, list]], 
            y: Union[np.ndarray, None]=None) -> 'BoWBP':
        if self.batches is None:
            self.batches = Batches(size=512 if len(D) >= 2048 else 256,
                                   random_state=0)
        D, y = self.set_validation_set(D, y=y)
        super(BoWBP, self).fit(D, y=y)
        optimizer_kwargs = self.combine_optimizer_kwargs()
        texts = self._transform(D)
        labels = self.dependent_variable(D, y=y)
        p = classifier(self.parameters, self.model,
                       texts, labels,
                       deviation=self.deviation,
                       validation=self.validation_set,
                       batches=self.batches, **optimizer_kwargs)
        self.parameters = p
        return self


class DenseBoWBP(DenseBoW, BoWBP):
    def __init__(self, dataset=False, 
                 **kwargs):
        super(DenseBoWBP, self).__init__(dataset=dataset, **kwargs)

    @property
    def model(self):
        return dense_model
    
    def _transform(self, X):
        return self.bow.transform(X)

    @property
    def parameters(self):
        """Parameters to optimize"""
        super(DenseBoWBP, self).parameters
        self._parameters['W'] = jnp.array(self.weights.T)
        self._parameters['W0'] = jnp.array(self.bias)
        return self._parameters

    @property
    def weights(self):
        return np.array([x._coef for x in self.text_representations])
    
    @property
    def bias(self):
        return np.array([x._intercept for x in self.text_representations])

    @parameters.setter
    def parameters(self, value):
        self.estimator_instance.coef_ = np.array(value['W_cl'].T)
        self.estimator_instance.intercept_ = np.array(value['W0_cl'])
        for x, m in zip(np.array(value['W'].T),
                        self.text_representations):
            m._coef[:] = x[:]
        for x, m in zip(np.array(value['W0']),
                        self.text_representations):
            m._intercept = float(x)