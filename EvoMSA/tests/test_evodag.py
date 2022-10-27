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


from EvoMSA.tests.test_base import TWEETS
from microtc.utils import tweet_iterator
import numpy as np


def test_EvoDAG():
    from EvoMSA.evodag import EvoDAG
    try:
        m = EvoDAG(lang='none')
        assert False
    except AssertionError:
        pass
    m = EvoDAG(lang='es', n_jobs=1, emoji=False)
    assert isinstance(m.models, list)
    n_models = len(m.models)
    m = EvoDAG(lang='es', n_jobs=1, emoji=False,
               skip_dataset=set(['HA']))
    assert n_models > len(m.models)
    m = EvoDAG(lang='es', n_jobs=1, dataset=False)
    assert n_models < len(m.models)


def test_EvoDAG_fit():
    from EvoMSA.evodag import EvoDAG
    D = list(tweet_iterator(TWEETS))
    evodag = EvoDAG(lang='es', n_estimators=2,
                    max_training_size=100).fit(D)
    assert isinstance(evodag, EvoDAG)
    assert isinstance(evodag.stack_generalization_instance, list)
    nmodels = evodag.transform(D).shape[1]
    evodag = EvoDAG(lang='es', n_estimators=1,
                    max_training_size=len(D),
                    TR=False).fit(D)
    nmodels2 = evodag.transform(D).shape[1]
    assert nmodels >  nmodels2


def test_EvoDAG_decision_function():
    from EvoMSA.evodag import EvoDAG
    D = list(tweet_iterator(TWEETS))
    evodag = EvoDAG(lang='es', 
                    n_estimators=2,
                    max_training_size=100).fit(D)
    output = evodag._decision_function(D)
    assert len(output) == 2 and isinstance(output, list)
    hy = evodag.decision_function(D)    
    assert hy.shape[0] == 1000 and hy.shape[1] == 4


def test_EvoDAG_predict():
    from EvoMSA.evodag import EvoDAG
    D = list(tweet_iterator(TWEETS))
    evodag = EvoDAG(lang='es', 
                    n_estimators=2, 
                    max_training_size=100).fit(D)
    hy = evodag.predict(D)
    assert (hy == [x['klass'] for x in D]).mean() > 0.25
    

def test_BoW_train_predict_decision_function():
    from EvoMSA.evodag import BoW
    D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es')
    hy = bow.train_predict_decision_function(D)
    assert isinstance(hy, list)
    assert hy[0].shape[0] == len(D)
    bow = BoW(lang='es')
    hy = bow.train_predict_decision_function([x for x in D if x['klass'] in ['P', 'N']])
    assert isinstance(hy, np.ndarray)
    assert hy.shape[0] == len([x for x in D if x['klass'] in ['P', 'N']])


def test_BoW_predict():
    from EvoMSA.evodag import BoW
    D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es').fit(D)
    hy = bow.predict(D)
    assert hy.shape[0] == len(D)


def test_BoW_decision_function():
    from EvoMSA.evodag import BoW
    D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es').fit(D)
    hy = bow.decision_function(D)
    assert hy[0].shape[0] == len(D)
    bow = BoW(lang='es').fit([x for x in D if x['klass'] in ['P', 'N']])
    hy = bow.decision_function(D)
    assert hy.shape[0] == len(D)
    