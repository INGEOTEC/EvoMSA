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


def test_EvoDAG_decision_function():
    from EvoMSA.evodag import EvoDAG, TextRepresentations
    D = list(tweet_iterator(TWEETS))
    class _EvoDAG(TextRepresentations):
        def estimator(self):
            return EvoDAG(n_estimators=2, 
                          max_training_size=100)    
    evodag = _EvoDAG(decision_function='decision_function').fit(D)
    hy = evodag.decision_function(D)    
    assert hy.shape[0] == 1000 and hy.shape[1] == 4


def test_EvoDAG_predict():
    from EvoMSA.evodag import EvoDAG, TextRepresentations
    D = list(tweet_iterator(TWEETS))
    class _EvoDAG(TextRepresentations):
        def estimator(self):
            return EvoDAG(n_estimators=2, 
                          max_training_size=100)    
    evodag = _EvoDAG().fit(D)
    hy = evodag.predict(D)
    assert (hy == [x['klass'] for x in D]).mean() > 0.25
    

def test_BoW_train_predict_decision_function():
    from EvoMSA.evodag import BoW
    D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es')
    hy = bow.train_predict_decision_function(D)
    assert isinstance(hy, np.ndarray)
    assert hy.shape[0] == len(D)
    bow = BoW(lang='es')


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
    assert hy.shape[0] == len(D)
    bow = BoW(lang='es').fit([x for x in D if x['klass'] in ['P', 'N']])
    hy = bow.decision_function(D)
    assert hy.shape[0] == len(D)


def test_BoW_key():
    from EvoMSA.evodag import BoW
    D = list(tweet_iterator(TWEETS))
    O = BoW(lang='es').transform(D)    
    X = [dict(klass=x['klass'], premise=x['text']) for x in D]    
    bow = BoW(lang='es', key='premise')
    assert abs(bow.transform(X) - O).sum() == 0
    X = [dict(klass=x['klass'], premise=x['text'], conclusion=x['text']) for x in D]
    bow = BoW(lang='es', key=['premise', 'conclusion'])
    assert abs(bow.transform(X) - O * 2).sum() == 0


def test_TextRepresentations_transform():
    from EvoMSA.evodag import TextRepresentations
    D = list(tweet_iterator(TWEETS))
    text_repr = TextRepresentations(lang='es')
    X = text_repr.transform(D)
    assert X.shape[0] == len(D)
    assert len(text_repr.text_representations) == X.shape[1]


def test_TextRepresentations_fit():
    from EvoMSA.evodag import TextRepresentations
    D = list(tweet_iterator(TWEETS))
    text_repr = TextRepresentations(lang='es').fit(D)
    text_repr.predict(['Buen dia'])


def test_TextRepresentations_key():
    from EvoMSA.evodag import TextRepresentations
    D = list(tweet_iterator(TWEETS))
    O = TextRepresentations(lang='es').transform(D)    
    X = [dict(klass=x['klass'], premise=x['text'], conclusion=x['text']) for x in D]
    bow = TextRepresentations(lang='es', key=['premise', 'conclusion'])
    assert abs(bow.transform(X) - O * 2).sum() == 0    


def test_StackGeneralization():
    from EvoMSA.evodag import StackGeneralization, BoW, TextRepresentations
    D = list(tweet_iterator(TWEETS))
    text_repr = StackGeneralization(lang='es',
                                    decision_function_models=[BoW(lang='es')],
                                    transform_models=[TextRepresentations(lang='es')]).fit(D)
    assert text_repr.predict(['Buen dia'])[0] == 'P'


def test_StackGeneralization_train_predict_decision_function():
    from EvoMSA.evodag import StackGeneralization, BoW, TextRepresentations
    D = list(tweet_iterator(TWEETS))
    text_repr = StackGeneralization(lang='es',
                                    decision_function_models=[BoW(lang='es')],
                                    transform_models=[TextRepresentations(lang='es')])
    hy = text_repr.train_predict_decision_function(D)
    assert hy.shape[0] == len(D)
    D1 = [x for x in D if x['klass'] in ['P', 'N']]
    hy = text_repr.train_predict_decision_function(D1)
    assert hy.shape[1] == 2