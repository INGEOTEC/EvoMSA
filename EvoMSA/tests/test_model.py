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
TWEETS = os.path.join(os.path.dirname(__file__), 'tweets.json')


def test_corpus():
    from EvoMSA.model import Corpus
    from b4msa.utils import tweet_iterator
    c = Corpus([x['text'] for x in tweet_iterator(TWEETS)])
    a = c['hola hola mundo']
    assert len(a) == 3
    assert a[0] == a[1]


def test_bernulli():
    import numpy as np
    from EvoMSA.model import Corpus, Bernulli
    from b4msa.utils import tweet_iterator
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
    from b4msa.utils import tweet_iterator
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
    import EvoMSA
    from EvoMSA.model import EmoSpaceEs
    from b4msa.utils import tweet_iterator
    X = [x for x in tweet_iterator(TWEETS)]
    emo = EmoSpaceEs()
    Xs = [emo[x] for x in X]
    assert len(Xs) == len(X) and len(Xs[0]) == 64
    assert emo.decision_function(X).shape[1] == 64
    assert emo.model_fname() == 'emo-v%s-es.evoemo' % EvoMSA.__version__


def test_EmoSpace_transform():
    from EvoMSA.model import EmoSpace
    from b4msa.utils import tweet_iterator
    X = [x for x in tweet_iterator(TWEETS)]
    emo = EmoSpace()
    r = emo.transform(X)
    print(len(r), len(X), len(r[0]), 64)
    assert len(r) == len(X) and len(r[0]) == 64


def test_EmoSpaceEn():
    from EvoMSA.model import EmoSpaceEn
    import EvoMSA
    cls = EmoSpaceEn()
    assert cls
    assert cls.model_fname() == 'emo-v%s-en.evoemo' % EvoMSA.__version__


def test_EmoSpaceAr():
    from EvoMSA.model import EmoSpaceAr
    import EvoMSA
    cls = EmoSpaceAr()
    assert cls
    assert cls.model_fname() == 'emo-v%s-ar.evoemo' % EvoMSA.__version__


def test_tonp():
    from EvoMSA.model import B4MSATextModel
    from b4msa.utils import tweet_iterator
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
    from b4msa.utils import tweet_iterator
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


# def test_HaSpace():
#     from EvoMSA.model import HaSpace
#     from b4msa.utils import tweet_iterator
#     X = [x for x in tweet_iterator(TWEETS)]
#     y = [x['klass'] for x in X]
#     emo = HaSpace().fit(X, y)
#     Xs = emo.decision_function(X)
#     print(Xs)
#     assert len(Xs) == len(X) and Xs.shape[1] == 3


# def test_HaSpaceEn():
#     from EvoMSA.model import HaSpaceEn
#     from b4msa.utils import tweet_iterator
#     X = [x for x in tweet_iterator(TWEETS)]
#     y = [x['klass'] for x in X]
#     emo = HaSpaceEn().fit(X, y)
#     Xs = emo.decision_function(X)
#     print(Xs)
#     assert len(Xs) == len(X) and Xs.shape[1] == 3


# def test_HaSpaceAr():
#     from EvoMSA.model import HaSpaceAr
#     from b4msa.utils import tweet_iterator
#     X = [x for x in tweet_iterator(TWEETS)]
#     y = [x['klass'] for x in X]
#     emo = HaSpaceAr().fit(X, y)
#     Xs = emo.decision_function(X)
#     print(Xs)
#     assert len(Xs) == len(X) and Xs.shape[1] == 3


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


def test_semantic_token_es():
    from EvoMSA.model import SemanticTokenEs
    from b4msa.utils import tweet_iterator
    corpus = [x for x in tweet_iterator(TWEETS)]
    semantic = SemanticTokenEs(corpus)
    print(semantic._weight.shape[0])
    assert semantic._weight.shape[0] == 998
    tr = semantic.transform([dict(text='buenos dias')])[0]
    print(tr)
    assert len(tr) == 3
    print([semantic.id2token[x[0]] for x in tr])


def test_semantic_token_en():
    from EvoMSA.model import SemanticTokenEn, EmoSpaceEn
    from b4msa.utils import tweet_iterator
    corpus = [x for x in tweet_iterator(TWEETS)]
    semantic = SemanticTokenEn(corpus)
    isinstance(semantic.semantic_space, EmoSpaceEn)


def test_semantic_token_ar():
    from EvoMSA.model import SemanticTokenAr, EmoSpaceAr
    from b4msa.utils import tweet_iterator
    corpus = [x for x in tweet_iterator(TWEETS)]
    semantic = SemanticTokenAr(corpus)
    isinstance(semantic.semantic_space, EmoSpaceAr)


def test_semantic_affective_es():
    from EvoMSA.model import SemanticAffectiveEs
    from b4msa.utils import tweet_iterator
    corpus = [x for x in tweet_iterator(TWEETS)]
    semantic = SemanticAffectiveEs(corpus)
    tokens = semantic.tokens(None)
    assert tokens
    print(semantic._weight.shape[0])
    assert semantic._weight.shape[0] == 1124


def test_semantic_affective_ar():
    from EvoMSA.model import SemanticAffectiveAr
    from b4msa.utils import tweet_iterator
    corpus = [x for x in tweet_iterator(TWEETS)]
    semantic = SemanticAffectiveAr(corpus)
    tokens = semantic.tokens(None)
    assert len(tokens) == 4073


def test_semantic_affective_en():
    from EvoMSA.model import SemanticAffectiveEn
    from b4msa.utils import tweet_iterator
    corpus = [x for x in tweet_iterator(TWEETS)]
    semantic = SemanticAffectiveEn(corpus)
    tokens = semantic.tokens(None)
    print(len(tokens))
    assert len(tokens) == 4102


def test_EmoSpace_create_space():
    from EvoMSA.model import EmoSpaceEs
    import os
    EmoSpaceEs.create_space(TWEETS, output='t.model')
    assert os.path.isfile('t.model')
    os.unlink('t.model')


def test_HA():
    from EvoMSA.model import HA
    from EvoMSA.base import EvoMSA
    from b4msa.utils import tweet_iterator
    import os
    X = [x for x in tweet_iterator(TWEETS)]
    HA.create_space(TWEETS, 'ha.model')
    EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10,
                            n_estimators=3),
           models=[['ha.model', 'sklearn.svm.LinearSVC']],
           n_jobs=2).fit(X, [x['klass'] for x in X])
    os.unlink('ha.model')


def test_emospace_model_cl():
    from EvoMSA.model import EmoSpace
    tm, cl = EmoSpace._create_space(TWEETS)
    emo = EmoSpace(model_cl=[tm, cl])
    r = emo[dict(text='buenos dias')]
    print(r)
    assert len(r) == 4
    
