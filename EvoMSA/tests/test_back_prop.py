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


from microtc.utils import tweet_iterator
from jax.experimental.sparse import BCSR
import numpy as np
from EvoMSA.back_prop import BoWBP, bow_model, DenseBoWBP
from EvoMSA.text_repr import BoW
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
    m = diff > 1e-6
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


def test_DenseBoWBP():
    """Test DenseBoWBP"""
    D = list(tweet_iterator(TWEETS))
    dense = DenseBoWBP(lang='es',
                       voc_size_exponent=13,
                       estimator_kwargs=dict(dual=True,
                                             random_state=0,
                                             class_weight='balanced'))
    assert dense.voc_size_exponent == 13
    assert dense.bow.num_terms == 2**13
