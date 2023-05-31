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


def test_EvoDAG_decision_function():
    from EvoMSA.evodag import EvoDAG
    from EvoMSA.text_repr import DenseBoW
    D = list(tweet_iterator(TWEETS))
    class _EvoDAG(DenseBoW):
        def estimator(self):
            return EvoDAG(n_estimators=2, 
                          max_training_size=100)    
    evodag = _EvoDAG(keyword=False, emoji=False,
                     v1=True,
                     decision_function_name='decision_function').fit(D)
    hy = evodag.decision_function(D)    
    assert hy.shape[0] == 1000 and hy.shape[1] == 4


def test_EvoDAG_predict():
    from EvoMSA.evodag import EvoDAG
    from EvoMSA.text_repr import DenseBoW
    D = list(tweet_iterator(TWEETS))
    class _EvoDAG(DenseBoW):
        def estimator(self):
            return EvoDAG(n_estimators=2, 
                          max_training_size=100)    
    evodag = _EvoDAG(keyword=False, v1=True,
                     emoji=False).fit(D)
    hy = evodag.predict(D)
    assert (hy == [x['klass'] for x in D]).mean() > 0.25
    

