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
from jax import nn
import jax.numpy as jnp
import numpy as np
from scipy.sparse import spmatrix
from jax.experimental.sparse import BCSR
from sklearn.model_selection import StratifiedShuffleSplit
from IngeoML.optimizer import classifier
from EvoMSA.text_repr import BoW, DenseBoW


@jax.jit
def bow_model(params, X):
    """BoWBP model"""

    Y = X @ params['W_cl'] + params['W0_cl']
    return Y


@jax.jit
def dense_model(params, X):
    """DenseBoWBP model"""

    _ = X @ params['W'] + params['W0']
    Y = _ / jnp.linalg.norm(_, axis=1, keepdims=True)
    Y = Y @ params['W_cl'] + params['W0_cl']
    return Y


@jax.jit
def stack_model(params, X, df):
    """StackBoWBP model"""

    _ = X @ params['W'] + params['W0']
    Y = _ / jnp.linalg.norm(_, axis=1, keepdims=True)
    Y = Y @ params['W_cl'] + params['W0_cl']
    Y = nn.softmax(Y, axis=1)
    pesos = nn.softmax(params['E'])
    return Y * pesos[0] + nn.softmax(df, axis=1) * pesos[1]


@jax.jit
def stack_model_binary(params, X, df):
    """StackBoWBP model"""

    _ = X @ params['W'] + params['W0']
    Y = _ / jnp.linalg.norm(_, axis=1, keepdims=True)
    Y = Y @ params['W_cl'] + params['W0_cl']
    Y = nn.sigmoid(Y)
    pesos = nn.softmax(params['E'])
    return Y * pesos[0] + nn.sigmoid(df) * pesos[1] - 0.5


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
                 deviation=None, fraction_initial_parameters=1,
                 optimizer_kwargs: dict=None,
                 **kwargs):
        super(BoWBP, self).__init__(voc_size_exponent=voc_size_exponent,
                                    estimator_kwargs=estimator_kwargs, **kwargs)
        self.deviation = deviation
        self.optimizer_kwargs = optimizer_kwargs
        self.fraction_initial_parameters = fraction_initial_parameters
        self.classes_ = None

    @property
    def fraction_initial_parameters(self):
        """Fraction of the training set to estimate the initial parameters"""
        return self._fraction_initial_parameters

    @fraction_initial_parameters.setter
    def fraction_initial_parameters(self, value):
        self._fraction_initial_parameters = value

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
    def deviation(self):
        """Function to measure the deviation between the true observations and the predictions."""
        return self._deviation

    @deviation.setter
    def deviation(self, value):
        self._deviation = value

    def initial_parameters(self, X, y):
        if y.ndim > 1:
            y = y.argmax(axis=1)
        train_size = self.fraction_initial_parameters
        if train_size == 1:
            tr = np.arange(X.shape[0])
        else:
            _ = StratifiedShuffleSplit(n_splits=1,
                                    train_size=train_size).split(X, y)
            tr, _ = next(_)
        m = self.estimator_class(**self.estimator_kwargs).fit(X[tr], y[tr])
        W = jnp.array(m.coef_.T)
        W0 = jnp.array(m.intercept_)
        return dict(W_cl=W, W0_cl=W0)

    @property
    def parameters(self):
        """Parameter to optimize"""
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    @property
    def model(self):
        return bow_model

    def _transform(self, X):
        return super(BoWBP, self).transform(X)

    def _combine_optimizer_kwargs(self):
        optimizer_defaults = dict(return_evolution=True)
        optimizer_defaults.update(self.optimizer_kwargs)        
        return optimizer_defaults

    def model_args(self, D: List[Union[dict, list]]):
        """Extra arguments pass to the model"""
        return None

    def fit(self, D: List[Union[dict, list]], 
            y: Union[np.ndarray, None]=None) -> 'BoWBP':
        optimizer_kwargs = self._combine_optimizer_kwargs()
        texts = self._transform(D)
        labels = self.dependent_variable(D, y=y)
        self.classes_ = np.unique(labels)
        p = classifier(self.initial_parameters, self.model,
                       texts, labels,
                       deviation=self.deviation,
                       model_args=self.model_args(D),
                       **optimizer_kwargs)
        if optimizer_kwargs['return_evolution']:
            self.evolution = p[1]
            p = p[0]
        self.parameters = p
        return self

    def decision_function(self, D: List[Union[dict, list]]) -> np.ndarray:
        X = self._transform(D)
        params = self.parameters
        args = self.model_args(D)
        if args is None:
            hy = self.model(params, BCSR.from_scipy_sparse(X))
        else:
            args = [self.array(x) for x in args]
            hy = self.model(params, BCSR.from_scipy_sparse(X), *args)
        return hy

    def predict(self, D: List[Union[dict, list]]) -> np.ndarray:
        df = self.decision_function(D)
        if df.shape[1] == 1:
            index = np.where(df.flatten() > 0, 1, 0)
        else:
            index = df.argmax(axis=1)
        return self.classes_[index]

    @staticmethod
    def array(data):
        """Encode data on jax"""

        if isinstance(data, spmatrix):
            return BCSR.from_scipy_sparse(data)
        return jnp.array(data)    


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

    def initial_parameters(self, X, y):
        if y.ndim > 1:
            y = y.argmax(axis=1)
        train_size = self.fraction_initial_parameters
        if train_size == 1:
            tr = np.arange(X.shape[0])
        else:
            _ = StratifiedShuffleSplit(n_splits=1,
                                    train_size=train_size).split(X, y)
            tr, _ = next(_)
        dense_w = self.weights.T
        dense_bias = self.bias
        _ = X[tr] @ dense_w + dense_bias
        _ = _ / np.linalg.norm(_, axis=1, keepdims=True)
        m = self.estimator_class(**self.estimator_kwargs).fit(_, y[tr])
        W = jnp.array(m.coef_.T)
        W0 = jnp.array(m.intercept_)
        return dict(W_cl=W, W0_cl=W0,
                    W=jnp.array(dense_w), W0=jnp.array(dense_bias)) 

    @property
    def weights(self):
        return np.array([x.coef for x in self.text_representations])

    @property
    def bias(self):
        return np.array([x.intercept for x in self.text_representations])

    # def __sklearn_clone__(self):
    #     ins = super(DenseBoWBP, self).__sklearn_clone__()
    #     _ = [clone(m) for m in self.text_representations]
    #     ins.text_representations = _
    #     return ins


class StackBoWBP(DenseBoWBP):
    @property
    def model(self):
        if self.classes_.shape[0] == 2:
            return stack_model_binary
        return stack_model

    def initial_parameters(self, X, y, df):
        params = super(StackBoWBP, self).initial_parameters(X, y)
        params['E'] = jnp.array([0.5, 0.5])
        return params

    def model_args(self, D: List[Union[dict, list]]):
        if not hasattr(self, '_bow_ins'):
            hy = BoW.train_predict_decision_function(self, D)
        else:
            X = super(StackBoWBP, self)._transform(D)
            hy = getattr(self._bow_ins, self.decision_function_name)(X)
        if hy.ndim == 1:
            hy = np.atleast_2d(hy).T
        return (hy, )

    def fit(self, D: List[Union[dict, list]], 
            y: Union[np.ndarray, None]=None) -> "StackBoWBP":
        super(StackBoWBP, self).fit(D, y=y)
        _ = self._transform(D)
        labels = self.dependent_variable(D, y=y)
        self._bow_ins = self.estimator_class(**self.estimator_kwargs).fit(_, labels)
        return self