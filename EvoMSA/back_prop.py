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
from jax.experimental.sparse import BCSR
import numpy as np
from scipy.special import expit, softmax
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import clone
from IngeoML.optimizer import classifier, array
from IngeoML.utils import soft_BER
from EvoMSA.text_repr import BoW, DenseBoW, StackGeneralization
from EvoMSA.utils import b4msa_params


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


def initial_parameters(hy_dense, df, y,
                       nclasses=2, score=None):
    """Estimate initial parameters :py:class:`~EvoMSA.back_prop.StackBoWBP`"""
    from sklearn.metrics import f1_score

    def f(x):
        hy = (x[0] * hy_dense + x[1] * df)
        if nclasses ==2:
            hy = np.where(hy > 0.5, 1, 0)
        else:
            hy = hy.argmax(axis=1)
        return score(y, hy)

    if score is None:
        score = lambda y, hy: f1_score(y, hy, average='macro')
    # df = softmax(df, axis=1)
    # df2 = softmax(df2, axis=1)
    value = np.linspace(0, 1, 100)
    _ = [f([v, 1-v]) for v in value]
    index = np.argmax(_)
    return jnp.array([value[index], 1 - value[index]])


@jax.jit
def stackbow(params, X, X2):
    mixer = nn.sigmoid(params['mixer'])
    frst = X * mixer
    scnd = X2 * (1  - mixer)
    return frst + scnd


@jax.jit
def stackbow_b_k(params, X):
    mixer = nn.softmax(params['mixer'])
    return X @ mixer


@jax.jit
def stackbow_m_k(params, X):
    mixer = nn.softmax(params['mixer'], axis=1)
    hy = X * mixer
    return hy.sum(axis=-1)


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
            args = [array(x) for x in args]
            hy = self.model(params, BCSR.from_scipy_sparse(X), *args)
        return hy

    def predict(self, D: List[Union[dict, list]]) -> np.ndarray:
        df = self.decision_function(D)
        if df.shape[1] == 1:
            index = np.where(df.flatten() > 0, 1, 0)
        else:
            index = df.argmax(axis=1)
        return self.classes_[index]


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


class StackBoW(StackGeneralization):
    """StackBoW"""

    def __init__(self, decision_function_models: list=None,
                 transform_models: list=[],
                 voc_size_exponent: int=15,
                 deviation=None, optimizer_kwargs: dict=None,
                 lang: str='es', **kwargs):
        if decision_function_models is None:
            estimator_kwargs = dict(dual='auto', class_weight='balanced')
            b4msa_kwargs = b4msa_params(lang=lang)
            if voc_size_exponent != 17:
                b4msa_kwargs['token_max_filter'] = 2**voc_size_exponent
            bow_np = BoW(lang=lang, pretrain=False,
                         b4msa_kwargs=b4msa_kwargs,
                         estimator_kwargs=estimator_kwargs)
            bow = BoW(lang=lang, voc_size_exponent=voc_size_exponent,
                      estimator_kwargs=estimator_kwargs)
            dense = DenseBoW(lang=lang,
                             dataset=False,
                             voc_size_exponent=voc_size_exponent,
                             estimator_kwargs=estimator_kwargs)
            decision_function_models = [bow_np, bow, dense]
        assert len(decision_function_models) > 1
        assert len(transform_models) == 0
        super().__init__(decision_function_models=decision_function_models,
                         **kwargs)
        self.deviation = deviation
        self.optimizer_kwargs = optimizer_kwargs
        self.classes_ = None
        self._mixer_value = None

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
    def mixer_value(self):
        """Contribution of each classifier to the prediction"""
        return self._mixer_value

    @property
    def deviation(self):
        """Function to measure the deviation between the true observations and the predictions."""
        return self._deviation

    @deviation.setter
    def deviation(self, value):
        self._deviation = value

    def _fit_bin_2(self, dfs, y):
        """Fit a binary problem with 2 algorithms"""
        X1 = jnp.c_[1 - dfs[0], dfs[0]]
        X2 = jnp.c_[1 - dfs[1], dfs[1]]
        h = {v: k for k, v in enumerate(self.classes_)}
        y_ = jnp.array([h[i] for i in y])
        y_ = np.c_[1 - y_, y_]
        if self.deviation is None:
            deviation = soft_BER
        else:
            deviation = self.deviation
        params = jnp.linspace(0, 1, 100)
        perf = [deviation(y_, p * X1 + (1 - p) * X2)
                for p in params]
        self._mixer_value = params[np.argmin(perf)]

    def _fit_bin_k(self, dfs, y):
        """Fit binary classification problems with k classifiers"""
        _ = np.ones(len(dfs))
        X = np.concatenate(dfs, axis=1)
        params = dict(mixer=jnp.array(_))
        p = classifier(params, stackbow_b_k, X, y,
                       validation=0, epochs=10000,
                       deviation=self.deviation,
                       distribution=True,
                       **self.optimizer_kwargs)
        self._mixer_value = softmax(p['mixer'])

    def _fit_mul_2(self, dfs, y):
        _ = np.zeros(dfs[0].shape[1])
        params = dict(mixer=jnp.array(_))
        p = classifier(params, stackbow, dfs[0], y,
                        model_args=(dfs[1], ),
                        validation=0, epochs=10000,
                        deviation=self.deviation,
                        distribution=True,
                        **self.optimizer_kwargs)
        self._mixer_value = expit(p['mixer'])

    def _fit_mul_k(self, dfs, y):
        X = np.array([x.T for x in dfs]).T
        _ = np.ones(X.shape[1:])
        params = dict(mixer=jnp.array(_))
        p = classifier(params, stackbow_m_k, X, y,
                       validation=0, epochs=10000,
                       deviation=self.deviation,
                       distribution=True,
                       **self.optimizer_kwargs)
        self._mixer_value = softmax(p['mixer'], axis=1)

    def fit(self, D: List[Union[dict, list]],
            y: Union[np.ndarray, None]=None) -> 'StackBoW':
        y = self.dependent_variable(D, y=y)
        self.classes_ = np.unique(y)            
        dfs = [ins.train_predict_decision_function(D, y=y)
               for ins in self._decision_function_models]
        if dfs[0].shape[1] > 1:
            dfs = [softmax(x, axis=1) for x in dfs]
            if len(dfs) == 2:
                self._fit_mul_2(dfs, y)
            else:
                self._fit_mul_k(dfs, y)
        else:
            dfs = [expit(x) for x in dfs]
            if len(dfs) == 2:
                self._fit_bin_2(dfs, y)
            else:
                self._fit_bin_k(dfs, y)
        _ = [clone(ins).fit(D, y=y) for ins in self._decision_function_models]
        self._decision_function_models = _
        return self
    
    def decision_function(self, D: List[Union[dict, list]]) -> np.ndarray:
        dfs = [ins.decision_function(D)
               for ins in self._decision_function_models]
        if dfs[0].shape[1] > 1:
            dfs = [softmax(x, axis=1) for x in dfs]
            mixer = self.mixer_value
            if len(dfs) == 2:
                frst = dfs[0] * mixer
                scnd = dfs[1] * (1  - mixer)
                return frst + scnd
            X = np.array([x.T for x in dfs]).T
            return (X * mixer).sum(axis=-1)
        else:
            dfs = [expit(x) for x in dfs]
            p = self.mixer_value
            if len(dfs) == 2:
                X1 = jnp.c_[1 - dfs[0], dfs[0]]
                X2 = jnp.c_[1 - dfs[1], dfs[1]]
                return p * X1 + (1 - p) * X2
            X = np.concatenate(dfs, axis=1)
            hy = X @ p
            return np.c_[1 - hy, hy]

    def predict(self, D: List[Union[dict, list]]) -> np.ndarray:
        df = self.decision_function(D)
        index = df.argmax(axis=1)
        return self.classes_[index]


class StackBoWBP(DenseBoWBP):
    @property
    def model(self):
        if self.classes_.shape[0] == 2:
            return stack_model_binary
        return stack_model

    def initial_parameters(self, X, y, df):
        params = super(StackBoWBP, self).initial_parameters(X, y)
        dense_w = self.weights.T
        dense_bias = self.bias
        Xd = X @ dense_w + dense_bias
        if self.classes_.shape[0] > 2:
            y = y.argmax(axis=1)
        hy_dense = self.train_predict_decision_function([1] * Xd.shape[0], y=y, X=Xd)
        if self.classes_.shape[0] > 2:
            df = expit(df)
            hy_dense = expit(hy_dense)
        else:
            df = softmax(df, axis=1)
            hy_dense = softmax(hy_dense, axis=1)
        params['E'] = initial_parameters(hy_dense, df, y,
                                         nclasses=self.classes_.shape[0])
        return params

    def model_args(self, D: List[Union[dict, list]]):
        if not hasattr(self, '_bow_ins'):
            X = self._transform(D)
            hy = self.train_predict_decision_function(D, X=X)
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