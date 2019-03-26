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
TWEETS = os.path.join(os.path.dirname(__file__), 'tweets.json')


def test_corpus():
    from EvoMSA.model import Corpus
    c = Corpus([x['text'] for x in tweet_iterator(TWEETS)])
    a = c['hola hola mundo']
    assert len(a) == 3
    assert a[0] == a[1]


def test_bernulli():
    import numpy as np
    from EvoMSA.model import Corpus, Bernulli
    from sklearn.preprocessing import LabelEncoder
    c = Corpus([x['text'] for x in tweet_iterator(TWEETS)])
    X = c.tonp([c[x['text']] for x in tweet_iterator(TWEETS)])
    y = [x['klass'] for x in tweet_iterator(TWEETS)]
    le = LabelEncoder().fit(y)
    y = le.transform(y)
    b = Bernulli()
    b.fit(X, y)
    pr = b.decision_function(X)
    assert pr.shape[0] == 1000 and pr.shape[1] == 4
    assert np.all((pr <= 1) & (pr >= -1))


def test_multinomial():
    import numpy as np
    from EvoMSA.model import Corpus, Multinomial
    from sklearn.preprocessing import LabelEncoder
    c = Corpus([x['text'] for x in tweet_iterator(TWEETS)])
    X = c.tonp([c[x['text']] for x in tweet_iterator(TWEETS)])
    y = [x['klass'] for x in tweet_iterator(TWEETS)]
    le = LabelEncoder().fit(y)
    y = le.transform(y)
    b = Multinomial()
    b.fit(X, y)
    pr = b.decision_function(X)
    print(pr.shape[0], pr, b.num_terms)
    assert pr.shape[0] == 1000 and pr.shape[1] == 4
    assert np.all((pr <= 1) & (pr >= -1))


def test_EmoSpace():
    from EvoMSA.model import EmoSpaceEs

    class EmoTest(EmoSpaceEs):
        @staticmethod
        def model_fname():
            return 'test.evoemo'

    X = [x for x in tweet_iterator(TWEETS)]
    emo = EmoTest()
    Xs = [emo[x] for x in X]
    assert len(Xs) == len(X) and len(Xs[0]) == 4
    assert emo.decision_function(X).shape[1] == 4
    # assert emo.model_fname() == 'emo-v%s-es.evoemo' % EvoMSA.__version__


def test_EmoSpace_create_space():
    from EvoMSA.model import EmoSpaceEs
    dirname = os.path.join(EmoSpaceEs.DIRNAME(), 'models')
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    output = os.path.join(dirname, EmoSpaceEs.model_fname())
    if not os.path.isfile(output):
        EmoSpaceEs.create_space(TWEETS, output=output)
    assert os.path.isfile(output)
    print(output)
    X = [x for x in tweet_iterator(TWEETS)]
    emo = EmoSpaceEs()
    Xs = [emo[x] for x in X]
    assert len(Xs) == len(X) and len(Xs[0]) == 4
    assert emo.decision_function(X).shape[1] == 4


def test_EmoSpaceEs():
    import EvoMSA
    from EvoMSA.model import EmoSpaceEs
    emo = EmoSpaceEs
    assert emo.model_fname() == 'emo-v%s-es.evoemo' % EvoMSA.__version__


def test_EmoSpace_transform():
    from EvoMSA.model import EmoSpaceEs

    class EmoTest(EmoSpaceEs):
        @staticmethod
        def model_fname():
            return 'test.evoemo'

    X = [x for x in tweet_iterator(TWEETS)]
    emo = EmoTest()
    r = emo.transform(X)
    print(len(r), len(X), len(r[0]), 4)
    assert len(r) == len(X) and len(r[0]) == 4


def test_EmoSpaceEn():
    from EvoMSA.model import EmoSpaceEn
    import EvoMSA
    cls = EmoSpaceEn
    assert cls
    assert cls.model_fname() == 'emo-v%s-en.evoemo' % EvoMSA.__version__


def test_EmoSpaceAr():
    from EvoMSA.model import EmoSpaceAr
    import EvoMSA
    cls = EmoSpaceAr
    assert cls
    assert cls.model_fname() == 'emo-v%s-ar.evoemo' % EvoMSA.__version__


def test_tonp():
    from EvoMSA.model import B4MSATextModel
    c = B4MSATextModel([x for x in tweet_iterator(TWEETS)])
    X = [c[x] for x in tweet_iterator(TWEETS)]
    Xp = c.tonp(X)
    assert Xp.shape[0] == len(X) and Xp.shape[1] == c.num_terms


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
    X = c.tonp([c[x['text']] for x in tweet_iterator(TWEETS)])
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


def test_semantic_token():
    from EvoMSA.model import SemanticTokenEs, EmoSpaceEs

    class EmoTest(EmoSpaceEs):
        @staticmethod
        def model_fname():
            return 'test.evoemo'

    class STest(SemanticTokenEs):
        @property
        def semantic_space(self):
            """Semantic space

            :rtype: instance
            """

            try:
                return self._semantic_space
            except AttributeError:
                self._semantic_space = EmoTest()
            return self._semantic_space

    corpus = [x for x in tweet_iterator(TWEETS)]
    semantic = STest(corpus)
    print(semantic._weight.shape[0])
    assert semantic._weight.shape[0] == 999
    tr = semantic.transform([dict(text='buenos dias')])[0]
    print(tr)
    assert len(tr) == 3
    print([semantic.id2token[x[0]] for x in tr])


def test_semantic_affective_es():
    from EvoMSA.model import SemanticAffectiveEs, EmoSpaceEs

    class EmoTest(EmoSpaceEs):
        @staticmethod
        def model_fname():
            return 'test.evoemo'

    class STest(SemanticAffectiveEs):
        @property
        def semantic_space(self):
            """Semantic space

            :rtype: instance
            """

            try:
                return self._semantic_space
            except AttributeError:
                self._semantic_space = EmoTest()
            return self._semantic_space
    
    corpus = [x for x in tweet_iterator(TWEETS)]
    semantic = STest(corpus)
    tokens = semantic.tokens(None)
    assert tokens
    print(semantic._weight.shape[0])
    assert semantic._weight.shape[0] == 1386


def test_HA():
    from EvoMSA.model import HA
    from EvoMSA.base import EvoMSA
    import os
    X = [x for x in tweet_iterator(TWEETS)]
    HA.create_space(TWEETS, 'ha.model')
    EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10,
                            n_estimators=3),
           models=[['ha.model', 'sklearn.svm.LinearSVC']],
           n_jobs=2).fit(X, [x['klass'] for x in X])
    os.unlink('ha.model')


def test_EmoSpace_model_cl():
    from EvoMSA.model import EmoSpace
    emo = EmoSpace(model_cl=EmoSpace._create_space(TWEETS))
    r = emo[dict(text='buenos dias')]
    print(r)
    assert len(r) == 4


# def test_emo_array():
#     import array
#     import numpy as np
#     from EvoMSA.model import EmoSpace
#     from EvoMSA.cython_utils import TextModelPredict
#     import math
#     emo = EmoSpace(model_cl=EmoSpace._create_space(TWEETS))
#     tm = emo._textModel
#     intercept = array.array('d', [x.intercept_[0] for x in emo._classifiers])
#     coef = np.vstack([x.coef_[0] for x in emo._classifiers])
#     coef = array.array('d', coef.T.flatten())
#     ee = TextModelPredict(tm, coef, array.array('d', intercept))
#     output = []
#     ee.transform(['buenos dias', 'cabron'], output)
#     for k, v in tm['buenos dias']:
#         init = len(intercept) * k
#         for j in range(len(intercept)):
#             intercept[j] += coef[init + j] * v
#     for a, b in zip(output[0], intercept):
#         print(a, b, a-b)
#         assert math.fabs(a - b) < 1e-6
#     for a, b in zip(output[1], intercept):
#         print(a, b, a-b)
#         assert math.fabs(a - b) > 1e-6
#     print('***')
#     for a, b in zip(ee['buenos dias'], intercept):
#         print(a, b, a-b)
#         assert math.fabs(a - b) < 1e-6

