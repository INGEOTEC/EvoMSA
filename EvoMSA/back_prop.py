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
from sklearn.base import clone
from IngeoML.optimizer import classifier
from EvoMSA.text_repr import BoW, DenseBoW


@jax.jit
def bow_model(params, X):
    """BoW model"""

    Y = X @ params['W_cl'] + params['W0_cl']
    return Y


@jax.jit
def dense_model(params, X):
    """DenseBoW model"""

    _ = X @ params['W'] + params['W0']
    Y = _ / jnp.linalg.norm(_, axis=1, keepdims=True)
    Y = Y @ params['W_cl'] + params['W0_cl']
    return Y


class BoWBP(BoW):
    """BoWBP is a :py:class:`~EvoMSA.text_repr.BoW` with the difference that the parameters are fine-tuned using jax
    
    >>> from microtc.utils import tweet_iterator
    >>> from EvoMSA.tests.test_base import TWEETS
    >>> from EvoMSA.back_prop import BoWBP
    >>> D = list(tweet_iterator(TWEETS))
    >>> bow = BoWBP(lang='es').fit(D)
    >>> bow.predict(['Buenos dias']).tolist()
    ['NONE']
    """

    def __init__(self, voc_size_exponent: int=15,
                 estimator_kwargs=dict(dual=True, class_weight='balanced'),
                 deviation=None,
                 validation_set=None,
                 optimizer_kwargs: dict=None,
                 **kwargs):
        super(BoWBP, self).__init__(voc_size_exponent=voc_size_exponent,
                                    estimator_kwargs=estimator_kwargs, **kwargs)
        self.deviation = deviation
        self.validation_set = validation_set
        self.optimizer_kwargs = optimizer_kwargs

    @property
    def evolution(self):
        """Evolution of the objective-function value"""
        return self._evolution

    @evolution.setter
    def evolution(self, value):
        self._evolution = value

    @property
    def optimizer_kwargs(self):
        """Arguments for the optimizer"""
        return self._optimizer_kwargs

    @optimizer_kwargs.setter
    def optimizer_kwargs(self, value):
        if value is None:
            value = {}
        self._optimizer_kwargs = value

    @property
    def validation_set(self):
        """Validation set"""
        return self._validation_set

    @validation_set.setter
    def validation_set(self, value):
        if value is None or value == 0:
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

        W = jnp.array(self.estimator_instance.coef_.T)
        W0 = jnp.array(self.estimator_instance.intercept_)
        return dict(W_cl=W, W0_cl=W0)

    @parameters.setter
    def parameters(self, value):
        self.estimator_instance.coef_ = np.array(value['W_cl'].T)
        self.estimator_instance.intercept_ = np.array(value['W0_cl'])

    @property
    def model(self):
        return bow_model

    def _transform(self, X):
        return super(BoWBP, self).transform(X)

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
        elif self.validation_set == 0:
            self.validation_set = None
        return D, y

    def _combine_optimizer_kwargs(self):
        decoder = self.estimator_instance.classes_
        n_outputs = 1 if decoder.shape[0] == 2 else decoder.shape[0]
        optimizer_defaults = dict(array=BCSR.from_scipy_sparse, n_outputs=n_outputs,
                                  return_evolution=True)
        optimizer_defaults.update(self.optimizer_kwargs)        
        return optimizer_defaults

    def initial_parameters(self, D, y=None):
        """Compute the initial parameters"""
        super(BoWBP, self).fit(D, y=y)

    def fit(self, D: List[Union[dict, list]], 
            y: Union[np.ndarray, None]=None) -> 'BoWBP':
        D, y = self.set_validation_set(D, y=y)
        self.initial_parameters(D, y=y)
        optimizer_kwargs = self._combine_optimizer_kwargs()
        texts = self._transform(D)
        labels = self.dependent_variable(D, y=y)
        p = classifier(self.parameters, self.model,
                       texts, labels,
                       deviation=self.deviation,
                       validation=self.validation_set,
                       **optimizer_kwargs)
        if optimizer_kwargs['return_evolution']:
            self.evolution = p[1]
            p = p[0]
        self.parameters = p
        return self


class DenseBoWBP(DenseBoW, BoWBP):
    """DenseBoWBP is a :py:class:`~EvoMSA.text_repr.DenseBoW` with the difference that the parameters are fine-tuned using jax
    
    >>> from microtc.utils import tweet_iterator
    >>> from EvoMSA.tests.test_base import TWEETS
    >>> from EvoMSA.back_prop import DenseBoWBP
    >>> D = list(tweet_iterator(TWEETS))
    >>> dense = DenseBoWBP(lang='es').fit(D)
    >>> dense.predict(['Buenos dias']).tolist()
    ['P']
    """

    def __init__(self, emoji: bool=True,
                 dataset: bool=False, keyword: bool=True,
                 estimator_kwargs=dict(dual='auto', class_weight='balanced'),
                 **kwargs):
        super(DenseBoWBP, self).__init__(emoji=emoji, dataset=dataset,
                                         keyword=keyword,
                                         estimator_kwargs=estimator_kwargs,
                                         **kwargs)

    @property
    def model(self):
        return dense_model

    def _transform(self, X):
        return self.bow.transform(X)

    @property
    def weights(self):
        return np.array([x.coef for x in self.text_representations])

    @property
    def bias(self):
        return np.array([x.intercept for x in self.text_representations])

    @property
    def parameters(self):
        """Parameters to optimize"""
        _parameters = super(DenseBoWBP, self).parameters
        _parameters['W'] = jnp.array(self.weights.T)
        _parameters['W0'] = jnp.array(self.bias)
        return _parameters

    @parameters.setter
    def parameters(self, value):
        self.estimator_instance.coef_ = np.array(value['W_cl'].T)
        self.estimator_instance.intercept_ = np.array(value['W0_cl'])
        for x, m in zip(np.array(value['W'].T),
                        self.text_representations):
            m.coef[:] = x[:]
        for x, m in zip(np.array(value['W0']),
                        self.text_representations):
            m.intercept = float(x)

    def __sklearn_clone__(self):
        ins = super(DenseBoWBP, self).__sklearn_clone__()
        _ = [clone(m) for m in self.text_representations]
        ins.text_representations = _
        return ins