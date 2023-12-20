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


from sklearn.base import clone
from microtc.utils import tweet_iterator
from jax.experimental.sparse import BCSR
import numpy as np
from EvoMSA.back_prop import BoWBP, bow_model, DenseBoWBP
from EvoMSA.text_repr import BoW, DenseBoW
from EvoMSA.tests.test_base import TWEETS



def test_BoWBP():
    """Test BoWBP"""
    D = list(tweet_iterator(TWEETS))
    bow2 = BoW(lang='es',
               estimator_kwargs=dict(dual=True,
                                     random_state=0,
                                     class_weight='balanced'),
               voc_size_exponent=15).fit(D)
    bow2_coef = bow2.estimator_instance.coef_
    bow = BoWBP(lang='es',
                estimator_kwargs=dict(dual=True,
                                      random_state=0,
                                      class_weight='balanced')).fit(D)
    bow_coef = bow.estimator_instance.coef_
    diff = np.fabs(bow_coef - bow2_coef).sum()
    assert diff > 0


def test_bow_model():
    """Test bow_model"""
    D = list(tweet_iterator(TWEETS))
    bow = BoWBP(lang='es',
                optimizer_kwargs=dict(epochs=2),
                estimator_kwargs=dict(dual=True,
                                      random_state=0,
                                      class_weight='balanced')).fit(D)
    assert 'W_cl' in bow.parameters
    assert 'W0_cl' in bow.parameters
    X = BCSR.from_scipy_sparse(bow.transform(D))
    y = bow_model(bow.parameters, X)
    y2 = bow.decision_function(D)
    diff = np.fabs(y - y2)
    m = diff > 1e-5
    assert m.sum() == 0
    D = [x for x in D if x['klass'] in ['N', 'P']]
    bow = BoWBP(lang='es',
                optimizer_kwargs=dict(epochs=2),
                estimator_kwargs=dict(dual=True,
                                      random_state=0,
                                      class_weight='balanced')).fit(D)
    X = BCSR.from_scipy_sparse(bow.transform(D))
    y = bow_model(bow.parameters, X)
    y2 = bow.decision_function(D)
    diff = np.fabs(y - y2)
    m = diff > 1e-6
    assert m.sum() == 0


def test_BoWBP_validation_set():
    """Test the validation_set property"""
    D = list(tweet_iterator(TWEETS))
    bow = BoWBP(lang='es',
                optimizer_kwargs=dict(epochs=2),
                validation_set=D,
                estimator_kwargs=dict(dual=True,
                                      random_state=0,
                                      class_weight='balanced')).fit(D)    


def test_BoWBP_evolution():
    """Test the evolution feature"""
    D = list(tweet_iterator(TWEETS))
    bow = BoWBP(lang='es',
                optimizer_kwargs=dict(epochs=2),
                estimator_kwargs=dict(dual=True,
                                      random_state=0,
                                      class_weight='balanced')).fit(D)
    assert len(bow.evolution)
    bow = BoWBP(lang='es',
                optimizer_kwargs=dict(epochs=2, return_evolution=False),
                estimator_kwargs=dict(dual=True,
                                      random_state=0,
                                      class_weight='balanced')).fit(D)
    try:
        bow.evolution
        assert False
    except AttributeError:
        pass


def test_DenseBoWBP():
    """Test DenseBoWBP"""
    D = list(tweet_iterator(TWEETS))
    dense = DenseBoWBP(lang='es',
                       voc_size_exponent=13,
                       estimator_kwargs=dict(dual=True,
                                             random_state=0,
                                             class_weight='balanced')
                       ).fit(D)
    assert dense.voc_size_exponent == 13
    assert dense.bow.num_terms == 2**13
    params = dense.parameters
    for key in ['W', 'W0', 'W_cl', 'W0_cl']:
        assert key in params
    # assert not hasattr(dense, '_weights')
    hy = dense.predict(D)
    assert len(hy) == len(D)


def test_DenseBoWBP_zero_validation_set():
    """Test the option of no using a validation set"""
    D = list(tweet_iterator(TWEETS))
    D = D + D + D
    dense = DenseBoWBP(lang='es',
                       voc_size_exponent=13,
                       validation_set=0,
                       estimator_kwargs=dict(dual=True,
                                             random_state=0,
                                             class_weight='balanced')
                       ).fit(D)
    assert dense.validation_set is None


def test_DenseBoWBP_clone():
    """Test DenseBoWBP clone"""
    D = list(tweet_iterator(TWEETS))
    dense = DenseBoWBP(lang='es',
                       voc_size_exponent=13,
                       n_jobs=-1)
    dense2 = clone(dense)
    assert not dense.text_representations[0] is dense2.text_representations[0]