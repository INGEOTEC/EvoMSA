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
try:
    from mock import MagicMock
except ImportError:
    from unittest.mock import MagicMock


class StoreDelete(object):
    def __init__(self, func, data, output):
        self._func = func
        self._data = data
        self._output = output
        self._delete = False

    def __enter__(self):
        dirname = os.path.join(get_dirname(), 'models')
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        self._output = os.path.join(dirname, self._output)
        if not os.path.isfile(self._output):
            self._func(self._data, output=self._output)
            self._delete = True
        return self

    def __exit__(self, *args):
        if self._delete:
            os.unlink(self._output)


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


def test_vector_space():
    X, y = get_data()
    evo = EvoMSA(stacked_method_args=dict(popsize=10, early_stopping_rounds=10,
                                          n_estimators=3),
                 models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernoulli']])
    evo.model(X)
    nrows = len(X)
    X = evo.vector_space(X)
    assert X[0].shape[0] == nrows


def test_EvoMSA_kfold_decision_function():
    from sklearn.preprocessing import LabelEncoder
    X, y = get_data()
    le = LabelEncoder().fit(y)
    y = le.transform(y)
    evo = EvoMSA(stacked_method_args=dict(popsize=10, early_stopping_rounds=10,
                                          n_estimators=3),
                 models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernoulli']])
    evo.model(X)
    X = evo.vector_space(X)
    cl = evo.models[1][1]
    D = evo.kfold_decision_function(cl, X[1], y)
    assert len(D[0]) == 4
    assert isinstance(D[0], list)


def test_EvoMSA_lang_missing():
    from EvoMSA import base
    try:
        base.EvoMSA(Emo=True)
    except AssertionError:
        return
    assert False


def test_EvoMSA_fit():
    from EvoMSA.model import Bernoulli
    from EvoDAG.model import EvoDAGE
    from microtc.utils import load_model, save_model
    X, y = get_data()
    print('iniciando')
    evo = EvoMSA(stacked_method_args=dict(popsize=10, early_stopping_rounds=10,
                                          time_limit=5,
                                          n_estimators=5),
                 models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernoulli']],
                 n_jobs=1).fit(X, y)
    print("Termine fit")
    assert evo
    assert isinstance(evo._svc_models[1], Bernoulli)
    assert isinstance(evo._evodag_model, EvoDAGE)
    save_model(evo, 'test.evomodel')
    print("Guarde modelo")
    evo = load_model('test.evomodel')
    print("Cargue modelo")
    assert isinstance(evo._svc_models[1], Bernoulli)
    assert isinstance(evo._evodag_model, EvoDAGE)
    os.unlink('test.evomodel')


def test_EvoMSA_predict():
    import numpy as np
    X, y = get_data()
    evo = EvoMSA(stacked_method_args=dict(popsize=10, early_stopping_rounds=10,
                                          time_limit=15, n_estimators=10),
                 models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernoulli']],
                 n_jobs=1).fit(X, y)
    hy = evo.predict(X)
    assert len(hy) == 1000
    print((np.array(y) == hy).mean(), hy)
    print(evo.predict_proba(X))
    assert (np.array(y) == hy).mean() > 0.8


def test_EvoMSA_predict_proba():
    X, y = get_data()
    evo = EvoMSA(stacked_method_args=dict(popsize=100,
                                          early_stopping_rounds=100,
                                          time_limit=5,
                                          n_estimators=5),
                 n_jobs=2).fit(X, y)
    hy = evo.predict_proba(X)
    assert len(hy) == 1000
    assert hy.min() >= 0 and hy.max() <= 1


def test_binary_labels_json():
    import json
    X, y = get_data()
    h = dict(NONE=0, N=0, NEU=0, P=1)
    y = [h[x] for x in y]
    evo = EvoMSA(stacked_method_args=dict(popsize=10, early_stopping_rounds=10,
                                          time_limit=5,
                                          n_estimators=5),
                 n_jobs=2).fit(X, y)
    hy = evo.predict(X)
    for x in hy:
        print(type(x), str(x))
        _ = json.dumps(dict(klass=str(x)))
    print(_)


def test_EvoMSA_model():
    X, y = get_data()
    model = EvoMSA(stacked_method_args=dict(popsize=10,
                                            early_stopping_rounds=10,
                                            n_estimators=3),
                   models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernoulli']],
                   n_jobs=2)
    assert len(model.models) == 2
    model.model(X)
    assert len(model._textModel) == 2
    print(model._textModel)


def test_EvoMSA_fit_svm():
    from sklearn.preprocessing import LabelEncoder
    X, y = get_data()
    from sklearn.svm import LinearSVC
    from EvoMSA.model import Bernoulli
    model = EvoMSA(stacked_method_args=dict(popsize=10,
                                            early_stopping_rounds=10,
                                            n_estimators=3),
                   models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernoulli']],
                   n_jobs=2)
    le = LabelEncoder().fit(y)
    y = le.transform(y)
    model.model(X)
    Xvs = model.vector_space(X)
    model.fit_svm(Xvs, y)
    print(model._svc_models)
    assert len(model._svc_models) == 2
    for ins, klass in zip(model._svc_models, [LinearSVC, Bernoulli]):
        assert isinstance(ins, klass)


def test_EvoMSA_evodag_class():
    from sklearn.neighbors import NearestCentroid
    import numpy as np
    X, y = get_data()
    model = EvoMSA(models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernoulli']],
                   stacked_method="sklearn.neighbors.NearestCentroid",
                   TR=False, n_jobs=2).fit(X, y)
    assert isinstance(model._evodag_model, NearestCentroid)
    cl = model.predict(X)
    hy = model.predict_proba(X)
    cl2 = model._le.inverse_transform(hy.argmax(axis=1))
    print(cl, cl2)
    assert np.all(cl == cl2)


def test_EvoMSA_multinomial():
    from EvoMSA.model import Multinomial
    X, y = get_data()
    evo = EvoMSA(stacked_method_args=dict(popsize=10, early_stopping_rounds=10,
                                          time_limit=5,
                                          n_estimators=5),
                 models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Multinomial']],
                 TR=False,
                 n_jobs=1).fit(X, y)
    assert evo
    assert isinstance(evo._svc_models[0], Multinomial)


def test_EvoMSA_empty_string():
    from EvoMSA.model import Multinomial
    X, y = get_data()
    X.append("")
    y.append("NONE")
    evo = EvoMSA(stacked_method_args=dict(popsize=10, early_stopping_rounds=10,
                                          time_limit=5,
                                          n_estimators=5),
                 models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Multinomial']],
                 TR=False,
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


def get_dirname():
    from EvoMSA import base
    import os
    return os.path.dirname(base.__file__)


def test_EvoMSA_regression():
    from EvoMSA.base import LabelEncoderWrapper
    from EvoMSA.utils import download
    X, y = get_data()
    X = [dict(text=x) for x in X]
    l = LabelEncoderWrapper().fit(y)
    y = l.transform(y) - 1.5
    evo = EvoMSA(stacked_method_args=dict(popsize=10,
                                          early_stopping_rounds=10,
                                          time_limit=5,
                                          n_estimators=2),
                 classifier=False,
                 models=[[download("emo_Es.tm"),
                          'EvoMSA.model.Identity']], TR=False,
                 n_jobs=1).fit(X, y)
    assert evo
    df = evo.decision_function(X)
    print(df.shape, df.ndim)
    assert df.shape[0] == len(X) and df.ndim == 1
    df = evo.predict(X)
    assert df.shape[0] == len(X) and df.ndim == 1


def test_EvoMSA_identity():
    from EvoMSA.model import Identity
    import numpy as np
    X, y = get_data()
    model = EvoMSA(stacked_method_args=dict(popsize=10,
                                            early_stopping_rounds=10,
                                            n_estimators=3),
                   models=[['EvoMSA.model.Corpus', 'EvoMSA.model.Bernoulli']],
                   TR=False,
                   stacked_method="EvoMSA.model.Identity",
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
    model = EvoMSA(stacked_method_args=dict(popsize=10,
                                            early_stopping_rounds=10,
                                            n_estimators=3),
                   TR=False, n_jobs=2)
    assert len(model.models) == 0
    model = EvoMSA(stacked_method_args=dict(popsize=10,
                                            early_stopping_rounds=10,
                                            n_estimators=3),
                   n_jobs=2)
    assert len(model.models) == 1
    print(model.models[0])
    assert model.models[0][0] == TextModel


def test_EvoMSA_param_Emo():
    from EvoMSA.base import EvoMSA
    X, y = get_data()
    for lang in ['ar', 'en', 'es']:
        model = EvoMSA(stacked_method_args=dict(popsize=10,
                                                early_stopping_rounds=10,
                                                n_estimators=3),
                       TR=False, lang=lang, Emo=True, n_jobs=2)
        assert len(model.models) == 1


def test_EvoMSA_param_TH():
    from EvoMSA.model import ThumbsUpDownAr, ThumbsUpDownEn, ThumbsUpDownEs
    from EvoMSA.base import EvoMSA

    X, y = get_data()
    for cl, lang in zip([ThumbsUpDownAr, ThumbsUpDownEn, ThumbsUpDownEs],
                        ['ar', 'en', 'es']):
        model = EvoMSA(stacked_method_args=dict(popsize=10,
                                                early_stopping_rounds=10,
                                                n_estimators=3),
                       TR=False, lang=lang, TH=True, n_jobs=2)
        assert len(model.models) == 1
        assert model.models[0][0] == cl


def test_EvoMSA_param_HA():
    from EvoMSA.base import EvoMSA
    X, y = get_data()
    for lang in ['ar', 'en', 'es']:
        model = EvoMSA(stacked_method_args=dict(popsize=10,
                                                early_stopping_rounds=10,
                                                n_estimators=3),
                       TR=False, lang=lang, HA=True, n_jobs=2)
        assert len(model.models) == 1


def test_EvoMSA_param_HA():
    from EvoMSA.base import EvoMSA
    X, y = get_data()
    for lang in ['ar', 'en', 'es']:
        model = EvoMSA(TR=False, lang=lang, Aggress=True)
        assert len(model.models) == 1


def test_EvoMSA_cpu_count():
    from EvoMSA.base import EvoMSA
    from multiprocessing import cpu_count
    X, y = get_data()
    model = EvoMSA(stacked_method_args=dict(popsize=10, early_stopping_rounds=10,
                                    n_estimators=3),
                   TR=False, n_jobs=-1)
    print(model.n_jobs, cpu_count())
    assert model.n_jobs == cpu_count()


def test_evomsa_wrapper():
    from microtc.utils import save_model
    from EvoMSA.base import EvoMSA
    from test_base import get_data
    X, y = get_data()
    model = EvoMSA(stacked_method="sklearn.naive_bayes.GaussianNB",
                   n_jobs=2).fit(X, y)
    save_model(model, 'tmp.evomsa')
    assert os.path.isfile('tmp.evomsa')
    evo = EvoMSA(models=[["tmp.evomsa", "EvoMSA.model.Identity"]],
                 stacked_method="sklearn.naive_bayes.GaussianNB",
                 n_jobs=2).fit(X, y)
    assert evo
    os.unlink("tmp.evomsa")


def test_tm_njobs():
    X, y = get_data()
    evo = EvoMSA(tm_n_jobs=2, n_jobs=1, TH=True, lang="es",
                 stacked_method="sklearn.svm.LinearSVC").fit(X, y)
    evo.predict(X)
    assert evo.n_jobs == 1
    assert evo.tm_n_jobs == 2


def test_sklearn_kfold():
    import numpy as np
    evo = EvoMSA(tm_n_jobs=2, n_jobs=1, TH=True, lang="es",
                 n_splits=3, stacked_method="sklearn.svm.LinearSVC")
    D = np.array([0, 1, 1, 1, 2, 2, 2])
    res = evo.sklearn_kfold(None, D, D)
    for _, _, _, tr, ts, _ in res:
        print(tr, ts)
        assert np.unique(D[tr]).shape[0] == 3


def test_model_instance():
    from microtc.textmodel import TextModel
    X, y = get_data()
    tm = TextModel().fit(X)
    evo = EvoMSA(tm_n_jobs=1, n_jobs=1, TR=False, lang="es",
                 models=[[tm, "sklearn.svm.LinearSVC"]],
                 stacked_method="sklearn.svm.LinearSVC").fit(X, y)
    assert evo.models[0][0] == tm


def test_cache():
    import hashlib

    def func(data, output):
        from b4msa.textmodel import TextModel
        from microtc.utils import tweet_iterator, save_model

        tm = TextModel().fit(list(tweet_iterator(data)))
        save_model(tm, output)

    with StoreDelete(func, TWEETS, "textmodel_cache.tm") as sd:
        cache = os.path.join("tm", "train.json")
        evo = EvoMSA(models=[[sd._output, "sklearn.svm.LinearSVC"]],
                     cache=cache)
        assert os.path.isdir("tm")
        output = hashlib.md5(sd._output.encode()).hexdigest()
        output = cache + "-%s" % output
        print(evo.cache.textModels)
        assert evo.cache.textModels[1] == output
        X, y = get_data()
        evo.first_stage(X, y)
        assert os.path.isfile(output)
        ML = list(evo.cache.ml_train())
        ML_K = list(evo.cache.ml_kfold())
        evo = EvoMSA(models=[[sd._output, "sklearn.svm.LinearSVC"]],
                     stacked_method_args=dict(popsize=10,
                                              early_stopping_rounds=10,
                                              n_estimators=3),
                     cache=cache).fit(X, y, test_set=X[:30])
        hy = evo.predict(X[:10])
        print(len(hy), hy)
        assert len(hy) == 10
        for k in ML:
            print(k)
            assert os.path.isfile(k)
        for k in ML_K:
            print(k)
            assert os.path.isfile(k)
        cache = os.path.join("tm", "test")
        evo.predict(X, cache=cache)
        output = cache + '-' + output.split("-")[1]
        print(output)
        assert os.path.isfile(output)


def test_lazy_loading():

    def func(data, output):
        from b4msa.textmodel import TextModel
        from microtc.utils import tweet_iterator, save_model

        tm = TextModel().fit(list(tweet_iterator(data)))
        save_model(tm, output)

    with StoreDelete(func, TWEETS, "textmodel_cache.tm") as sd:
        cache = os.path.join("tm", "train.json")
        X, y = get_data()
        evo = EvoMSA(models=[[sd._output, "sklearn.svm.LinearSVC"]],
                     cache=cache, TR=False,
                     stacked_method="sklearn.naive_bayes.GaussianNB").fit(X, y)
        evo = EvoMSA(models=[[sd._output, "sklearn.svm.LinearSVC"]],
                     cache=cache, TR=False,
                     stacked_method="sklearn.naive_bayes.GaussianNB").fit(X, y)
        print(evo._textModel[0], sd._output)
        assert evo._textModel[0] == sd._output
        evo.stacked_method


        
