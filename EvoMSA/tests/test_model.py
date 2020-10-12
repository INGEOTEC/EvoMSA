# Copyright 2017 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from microtc.utils import tweet_iterator
from test_base import StoreDelete
TWEETS = os.path.join(os.path.dirname(__file__), 'tweets.json')


def test_corpus():
    from EvoMSA.model import Corpus
    c = Corpus([x['text'] for x in tweet_iterator(TWEETS)])
    a = c['hola hola mundo']
    assert len(a) == 3
    assert a[0] == a[1]


def test_bernoulli():
    import numpy as np
    from EvoMSA.model import Corpus, Bernoulli
    from sklearn.preprocessing import LabelEncoder
    c = Corpus([x['text'] for x in tweet_iterator(TWEETS)])
    X = c.transform([x['text'] for x in tweet_iterator(TWEETS)])
    y = [x['klass'] for x in tweet_iterator(TWEETS)]
    le = LabelEncoder().fit(y)
    y = le.transform(y)
    b = Bernoulli()
    b.fit(X, y)
    pr = b.decision_function(X)
    assert pr.shape[0] == 1000 and pr.shape[1] == 4
    assert np.all((pr <= 1) & (pr >= -1))


def test_multinomial():
    import numpy as np
    from EvoMSA.model import Corpus, Multinomial
    from sklearn.preprocessing import LabelEncoder
    c = Corpus([x['text'] for x in tweet_iterator(TWEETS)])
    X = c.transform([x['text'] for x in tweet_iterator(TWEETS)])
    y = [x['klass'] for x in tweet_iterator(TWEETS)]
    le = LabelEncoder().fit(y)
    y = le.transform(y)
    b = Multinomial()
    b.fit(X, y)
    pr = b.decision_function(X)
    print(pr.shape[0], pr, b.num_terms)
    assert pr.shape[0] == 1000 and pr.shape[1] == 4
    assert np.all((pr <= 1) & (pr >= -1))


def test_ThumbsUpDownEs():
    from EvoMSA.model import ThumbsUpDownEs
    thumbs = ThumbsUpDownEs()
    _ = thumbs['adoracion XxX fervor vergazo']
    assert _ == (2, 1)


def test_ThumbsUpDownEn():
    from EvoMSA.model import ThumbsUpDownEn
    aff = ThumbsUpDownEn()
    _ = aff['adorably XxX elation vergazo']
    assert (2, 0) == _
    print(_)


def test_ThumbsUpDownAr():
    from EvoMSA.model import ThumbsUpDownAr
    aff = ThumbsUpDownAr()
    _ = aff['adorably XxX elation vergazo']
    assert (0, 0) == _


def test_OutputClassifier():
    from EvoMSA.model import Corpus, OutputClassifier
    from sklearn.preprocessing import LabelEncoder
    c = Corpus([x['text'] for x in tweet_iterator(TWEETS)])
    X = c.transform([x['text'] for x in tweet_iterator(TWEETS)])
    y = [x['klass'] for x in tweet_iterator(TWEETS)]
    le = LabelEncoder().fit(y)
    y = le.transform(y)
    b = OutputClassifier(output='xx')
    assert b._output == 'xx'
    b.fit(X, y)
    assert os.path.isfile('xx_train.csv')
    pr = b.decision_function(X)
    assert os.path.isfile('xx_test.csv')
    assert len(open('xx_test.csv').readlines()) == pr.shape[0]
    os.unlink('xx_train.csv')
    os.unlink('xx_test.csv')


def test_AggressivenessEs():
    from EvoMSA.model import AggressivenessEs
    aff = AggressivenessEs()
    _ = aff['indio XxX fervor vergazo']
    assert len(_) == 1


def test_AggressivenessEn():
    from EvoMSA.model import AggressivenessEn
    aff = AggressivenessEn()
    _ = aff['cockhead cockjockey adoracion XxX fervor vergazo']
    print(_)
    assert len(_) == 2


def test_AggressivenessAr():
    from EvoMSA.model import AggressivenessAr
    aff = AggressivenessAr()
    _ = aff['adoracion XxX fervor vergazo']
    assert len(_) == 0


def test_Vec():
    from EvoMSA.model import Vec
    a = dict(vec=[1, 3, 1])
    vec = Vec()
    assert vec[a] == [1, 3, 1]


def test_LabeledDataSet():
    from EvoMSA.model import LabeledDataSet
    import os
    LabeledDataSet.create_space(TWEETS, 'lb.model')
    assert os.path.isfile('lb.model')
    os.unlink('lb.model')
