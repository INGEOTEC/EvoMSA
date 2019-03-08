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
from EvoMSA.base import EvoMSA
from microtc.utils import tweet_iterator
import os
TWEETS = os.path.join(os.path.dirname(__file__), 'tweets.json')


def get_data():
    D = [[x['text'], x['klass']] for x in tweet_iterator(TWEETS)]
    X = [x[0] for x in D]
    y = [x[1] for x in D]
    return X, y


def test_TextModel():
    from b4msa.textmodel import TextModel
    X, y = get_data()
    evo = EvoMSA()
    evo.model(X)
    assert isinstance(evo._textModel, list)
    assert isinstance(evo._textModel[0], TextModel)
    assert len(evo._textModel) == 1
    evo.model([X, X])
    assert isinstance(evo._textModel, list)
    assert len(evo._textModel) == 2
    for x in evo._textModel:
        assert isinstance(x, TextModel)


def test_vector_space():
    X, y = get_data()
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10, n_estimators=3),
                 models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernulli']])
    evo.model(X)
    nrows = len(X)
    X = evo.vector_space(X)
    assert X[0].shape[0] == nrows


def test_EvoMSA_kfold_decision_function():
    from sklearn.preprocessing import LabelEncoder
    X, y = get_data()
    le = LabelEncoder().fit(y)
    y = le.transform(y)
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10, n_estimators=3),
                 models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernulli']])
    evo.model(X)
    X = evo.vector_space(X)
    cl = evo.models[1][1]
    D = evo.kfold_decision_function(cl, X[1], y)
    assert len(D[0]) == 4
    assert isinstance(D[0], list)


def test_EvoMSA_fit():
    from EvoMSA.model import Bernulli
    from EvoDAG.model import EvoDAGE
    from microtc.utils import load_model, save_model
    X, y = get_data()
    print('iniciando')
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10, time_limit=5,
                                  n_estimators=5),
                 models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernulli']],
                 n_jobs=1).fit(X, y)
    print("Termine fit")
    assert evo
    assert isinstance(evo._svc_models[1], Bernulli)
    assert isinstance(evo._evodag_model, EvoDAGE)
    save_model(evo, 'test.evomodel')
    print("Guarde modelo")
    evo = load_model('test.evomodel')
    print("Cargue modelo")
    assert isinstance(evo._svc_models[1], Bernulli)
    assert isinstance(evo._evodag_model, EvoDAGE)
    os.unlink('test.evomodel')


def test_EvoMSA_fit2():
    X, y = get_data()
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10, time_limit=5,
                                  n_estimators=5),
                 n_jobs=2).fit([X, [x for x, y0 in zip(X, y) if y0 in ['P', 'N']]],
                               [y, [x for x in y if x in ['P', 'N']]])
    assert evo
    D = evo.transform(X, y)
    assert len(D[0]) == 5


def test_EvoMSA_evodag_args():
    X, y = get_data()
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10, time_limit=5,
                                  n_estimators=5),
                 n_jobs=2).fit([X, [x for x, y0 in zip(X, y) if y0 in ['P', 'N']]],
                               [y, [x for x in y if x in ['P', 'N']]])
    assert evo
    D = evo.transform(X, y)
    assert len(D[0]) == 5
    assert len(D) == 1000


def test_EvoMSA_predict():
    import numpy as np
    X, y = get_data()
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10, time_limit=15, n_estimators=10),
                 models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernulli']],
                 n_jobs=1).fit([X, [x for x, y0 in zip(X, y) if y0 in ['P', 'N']]],
                               [y, [x for x in y if x in ['P', 'N']]])
    hy = evo.predict(X)
    assert len(hy) == 1000
    print((np.array(y) == hy).mean(), hy)
    print(evo.predict_proba(X))
    assert (np.array(y) == hy).mean() > 0.8


def test_EvoMSA_bernulli_predict():
    import numpy as np
    X, y = get_data()
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10, time_limit=15, n_estimators=10),
                 models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernulli']], TR=False,
                 n_jobs=1).fit([X, [x for x, y0 in zip(X, y) if y0 in ['P', 'N']]],
                               [y, [x for x in y if x in ['P', 'N']]])
    hy = evo.predict(X)
    assert len(hy) == 1000
    print((np.array(y) == hy).mean(), hy)
    print(evo.predict_proba(X))
    assert (np.array(y) == hy).mean() > 0.8


def test_EvoMSA_predict_proba():
    X, y = get_data()
    evo = EvoMSA(evodag_args=dict(popsize=100, early_stopping_rounds=100, time_limit=5,
                                  n_estimators=5),
                 n_jobs=2).fit([X, [x for x, y0 in zip(X, y) if y0 in ['P', 'N']]],
                               [y, [x for x in y if x in ['P', 'N']]])
    hy = evo.predict_proba(X)
    assert len(hy) == 1000
    assert hy.min() >= 0 and hy.max() <= 1


def test_binary_labels_json():
    import json
    X, y = get_data()
    h = dict(NONE=0, N=0, NEU=0, P=1)
    y = [h[x] for x in y]
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10, time_limit=5,
                                  n_estimators=5),
                 n_jobs=2).fit(X, y)
    hy = evo.predict(X)
    for x in hy:
        print(type(x), str(x))
        _ = json.dumps(dict(klass=str(x)))
    print(_)


def test_EvoMSA_exogenous_model():
    X, y = get_data()
    model = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10),
                   n_jobs=2).fit(X, y)
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10, time_limit=5,
                                  n_estimators=5),
                 n_jobs=2)
    evo.exogenous_model = model
    evo.fit(X, y)
    D = evo.transform(X)
    assert D.shape[1] == 8


def test_EvoMSA_model():
    X, y = get_data()
    model = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10,
                                    n_estimators=3),
                   models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernulli']],
                   n_jobs=2)
    assert len(model.models) == 2
    model.model(X)
    assert len(model._textModel) == 2
    print(model._textModel)


def test_EvoMSA_fit_svm():
    from sklearn.preprocessing import LabelEncoder
    X, y = get_data()
    from sklearn.svm import LinearSVC
    from EvoMSA.model import Bernulli
    model = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10,
                                    n_estimators=3),
                   models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernulli']],
                   n_jobs=2)
    le = LabelEncoder().fit(y)
    y = le.transform(y)
    model.fit_svm(X, y)
    print(model._svc_models)
    assert len(model._svc_models) == 2
    for ins, klass in zip(model._svc_models, [LinearSVC, Bernulli]):
        assert isinstance(ins, klass)


def test_EvoMSA_transform():
    from sklearn.preprocessing import LabelEncoder
    X, y = get_data()
    Xn = [X, [x for x, y0 in zip(X, y) if y0 in ['P', 'N']]]
    Y = [y, [x for x in y if x in ['P', 'N']]]
    Yn = []
    for y0 in Y:
        _ = LabelEncoder().fit(y0)
        Yn.append(_.transform(y0).tolist())
    X = Xn
    y = Yn
    for m, shape, TR in zip([[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernulli']],
                             [['EvoMSA.model.Corpus', 'EvoMSA.model.Bernulli']]], [11, 6],
                            [True, False]):
        evo = EvoMSA(evodag_args=dict(popsize=10,
                                      early_stopping_rounds=10,
                                      time_limit=15, n_estimators=10), TR=TR,
                     models=m,
                     n_jobs=1)
        evo.fit_svm(X, y)
        D = evo.transform(X[0], y[0])
        D.shape[1] == shape


def test_EvoMSA_evodag_class():
    from sklearn.neighbors import NearestCentroid
    import numpy as np
    X, y = get_data()
    model = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10,
                                    n_estimators=3),
                   models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernulli']],
                   evodag_class="sklearn.neighbors.NearestCentroid", TR=False,
                   n_jobs=2).fit(X, y)
    assert isinstance(model._evodag_model, NearestCentroid)
    cl = model.predict(X)
    hy = model.predict_proba(X)
    cl2 = model._le.inverse_transform(hy.argmax(axis=1))
    print(cl, cl2)
    assert np.all(cl == cl2)


def test_EvoMSA_multinomial():
    from EvoMSA.model import Multinomial
    X, y = get_data()
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10, time_limit=5,
                                  n_estimators=5),
                 models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Multinomial']], TR=False,
                 n_jobs=1).fit(X, y)
    assert evo
    assert isinstance(evo._svc_models[0], Multinomial)


def test_EvoMSA_empty_string():
    from EvoMSA.model import Multinomial
    X, y = get_data()
    X.append("")
    y.append("NONE")
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10, time_limit=5,
                                  n_estimators=5),
                 models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Multinomial']], TR=False,
                 n_jobs=1).fit(X, y)
    assert evo
    assert isinstance(evo._svc_models[0], Multinomial)


def test_label_encoder():
    import numpy as np
    from EvoMSA.base import LabelEncoderWrapper
    y = [2, 2, -1, 0, 0]
    l = LabelEncoderWrapper().fit(y)
    print(l._m)
    yy = l.transform(y)
    assert np.all(np.array([2, 2, 0, 1, 1]) == yy)
    y = [x['klass'] for x in tweet_iterator(TWEETS)]
    l = LabelEncoderWrapper().fit(y)


def test_label_encoder_kwargs():
    from EvoMSA.base import LabelEncoderWrapper
    y = [2, 2, -1, 0.3, 0]
    l = LabelEncoderWrapper(classifier=False).fit(y)
    print(l._m)
    yy = l.transform(y)
    assert yy[-2] == 0.3
    yy = l.inverse_transform(y)
    assert yy[-2] == 0.3
    assert not l.classifier


def test_EvoMSA_regression():
    from EvoMSA.base import LabelEncoderWrapper
    from EvoMSA.model import EmoSpaceEs
    import os
    dirname = os.path.join(EmoSpaceEs.DIRNAME(), 'models')
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    output = os.path.join(dirname, EmoSpaceEs.model_fname())
    if not os.path.isfile(output):
        EmoSpaceEs.create_space(TWEETS, output=output)
    X, y = get_data()
    X = [dict(text=x) for x in X]
    l = LabelEncoderWrapper().fit(y)
    y = l.transform(y) - 1.5
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10,
                                  time_limit=5, n_estimators=2),
                 classifier=False,
                 models=[['EvoMSA.model.Identity', 'EvoMSA.model.EmoSpaceEs']], TR=False,
                 n_jobs=1).fit(X, y)
    assert evo
    df = evo.decision_function(X)
    print(df.shape, df.ndim)
    assert df.shape[0] == len(X) and df.ndim == 1


def test_EvoMSA_identity():
    from EvoMSA.model import Identity
    import numpy as np
    X, y = get_data()
    model = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10,
                                    n_estimators=3),
                   models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernulli']], TR=False,
                   evodag_class="EvoMSA.model.Identity",
                   n_jobs=2).fit(X, y)
    assert isinstance(model._evodag_model, Identity)
    cl = model.predict(X)
    hy = model.predict_proba(X)
    cl2 = model._le.inverse_transform(hy.argmax(axis=1))
    print(cl, cl2)
    assert np.all(cl == cl2)
    

def test_EvoMSA_param_TR():
    from EvoMSA.base import EvoMSA
    from b4msa.textmodel import TextModel
    X, y = get_data()
    model = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10,
                                    n_estimators=3),
                   TR=False, n_jobs=2)
    assert len(model.models) == 0
    model = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10,
                                    n_estimators=3),
                   n_jobs=2)
    assert len(model.models) == 1
    print(model.models[0])
    assert model.models[0][0] == TextModel


def test_EvoMSA_param_Emo():
    from EvoMSA.model import EmoSpaceEs, EmoSpaceEn, EmoSpaceAr
    from EvoMSA.base import EvoMSA

    X, y = get_data()
    for cl, lang in zip([EmoSpaceAr, EmoSpaceEn, EmoSpaceEs],
                        ['ar', 'en', 'es']):
        model = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10,
                                        n_estimators=3),
                       TR=False, lang=lang, Emo=True, n_jobs=2)
        assert len(model.models) == 1
        assert model.models[0][0] == cl


def test_EvoMSA_param_TH():
    from EvoMSA.model import ThumbsUpDownAr, ThumbsUpDownEn, ThumbsUpDownEs
    from EvoMSA.base import EvoMSA

    X, y = get_data()
    for cl, lang in zip([ThumbsUpDownAr, ThumbsUpDownEn, ThumbsUpDownEs],
                        ['ar', 'en', 'es']):
        model = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10,
                                        n_estimators=3),
                       TR=False, lang=lang, TH=True, n_jobs=2)
        assert len(model.models) == 1
        assert model.models[0][0] == cl


def test_EvoMSA_param_HA():
    from EvoMSA.model import ThumbsUpDownAr, ThumbsUpDownEn, ThumbsUpDownEs
    from EvoMSA.model import EmoSpace, HA
    from EvoMSA.base import EvoMSA
    from b4msa.lang_dependency import get_lang
    import os
    X, y = get_data()
    dirname = os.path.join(EmoSpace.DIRNAME(), 'models')
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    for lang in ['ar', 'en', 'es']:
        l = get_lang(lang)
        model_fname = "%s.evoha" % l
        HA.create_space(TWEETS, os.path.join(dirname, model_fname))
    for cl, lang in zip([ThumbsUpDownAr, ThumbsUpDownEn, ThumbsUpDownEs],
                        ['ar', 'en', 'es']):
        model = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10,
                                        n_estimators=3),
                       TR=False, lang=lang, HA=True, n_jobs=2)
        assert len(model.models) == 1
        print(model.models[0][0])
        assert os.path.isfile(model.models[0][0])
        
