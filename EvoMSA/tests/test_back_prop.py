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
from EvoMSA.back_prop import BoWBP, bow_model, DenseBoWBP, StackBoWBP
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
    bow2_coef = bow2.estimator_instance.coef_.T
    bow = BoWBP(lang='es', fraction_initial_parameters=0.6,
                estimator_kwargs=dict(dual=True,
                                      random_state=0,
                                      class_weight='balanced')).fit(D)
    # assert bow.parameters is None
    bow_coef = bow.parameters['W_cl']
    diff = np.fabs(bow_coef - bow2_coef).sum()
    assert diff > 0
    assert bow.predict(D) is not None


def test_binary():
    """Test BoWBP"""
    D = list(tweet_iterator(TWEETS))
    D = [x for x in D if x['klass'] in {'N', 'P'}]
    stack = StackBoWBP(lang='es',
                       voc_size_exponent=13).fit(D)
    hy = stack.predict(D)
    acc = (np.array([x['klass'] for x in D]) == hy).mean()
    assert acc > 0.95
    bow = BoWBP(lang='es').fit(D)
    hy = bow.predict(D)
    acc = (np.array([x['klass'] for x in D]) == hy).mean()
    assert acc > 0.9
    dense = DenseBoWBP(lang='es',
                       voc_size_exponent=13).fit(D)
    hy = dense.predict(D)
    acc = (np.array([x['klass'] for x in D]) == hy).mean()
    assert acc > 0.85



# def test_BoWBP_validation_set():
#     """Test the validation_set property"""
#     D = list(tweet_iterator(TWEETS))
#     bow = BoWBP(lang='es',
#                 optimizer_kwargs=dict(epochs=2),
#                 validation_set=D,
#                 estimator_kwargs=dict(dual=True,
#                                       random_state=0,
#                                       class_weight='balanced')).fit(D)    


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


def test_StackBoWBP():
    """Test StackBoWBP"""

    dataset = list(tweet_iterator(TWEETS))
    ins = StackBoWBP(lang='es', voc_size_exponent=13).fit(dataset)
    assert 'E' in ins.parameters
