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
import os
TWEETS = os.path.join(os.path.dirname(__file__), 'tweets.json')


def get_data():
    from b4msa.utils import tweet_iterator
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
    evo = EvoMSA()
    evo.model(X)
    X = evo.vector_space(X)
    assert len(X[0]) == 1000
    assert len(X[0][0][0]) == 2


def test_EvoMSA_kfold_decision_function():
    X, y = get_data()
    evo = EvoMSA()
    evo.model(X)
    X = evo.vector_space(X)
    D = evo.kfold_decision_function(X[0], y)
    assert len(D[0]) == 4
    assert isinstance(D[0], list)


def test_EvoMSA_fit():
    from b4msa.classifier import SVC
    from EvoDAG.model import Ensemble
    X, y = get_data()
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10),
                 n_jobs=2).fit(X, y)
    assert evo
    assert isinstance(evo._svc_models[0], SVC)
    assert isinstance(evo._evodag_model, Ensemble)


def test_EvoMSA_fit2():
    X, y = get_data()
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10),
                 n_jobs=4).fit([X, [x for x, y0 in zip(X, y) if y0 in ['P', 'N']]],
                               [y, [x for x in y if x in ['P', 'N']]])
    assert evo
    D = evo.transform(X, y)
    assert len(D[0]) == 5


def test_EvoMSA_evodag_args():
    X, y = get_data()
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10),
                 n_jobs=4).fit([X, [x for x, y0 in zip(X, y) if y0 in ['P', 'N']]],
                               [y, [x for x in y if x in ['P', 'N']]])
    assert evo
    D = evo.transform(X, y)
    assert len(D[0]) == 5
    assert len(D) == 1000


def test_EvoMSA_predict():
    import numpy as np
    X, y = get_data()
    evo = EvoMSA(evodag_args=dict(popsize=100, early_stopping_rounds=100),
                 n_jobs=4).fit([X, [x for x, y0 in zip(X, y) if y0 in ['P', 'N']]],
                               [y, [x for x in y if x in ['P', 'N']]])
    hy = evo.predict(X)
    assert len(hy) == 1000
    print((np.array(y) == hy).mean(), hy)
    print(evo.predict_proba(X))
    assert (np.array(y) == hy).mean() > 0.9


def test_EvoMSA_fit3():
    X, y = get_data()
    evo = EvoMSA(use_ts=False, evodag_args=dict(popsize=10, early_stopping_rounds=10),
                 n_jobs=4).fit([X, [x for x, y0 in zip(X, y) if y0 in ['P', 'N']]],
                               [y, [x for x in y if x in ['P', 'N']]])
    assert evo
    D = evo.transform(X, y)
    assert len(D[0]) == 1


def test_EvoMSA_predict_proba():
    X, y = get_data()
    evo = EvoMSA(evodag_args=dict(popsize=100, early_stopping_rounds=100),
                 n_jobs=4).fit([X, [x for x, y0 in zip(X, y) if y0 in ['P', 'N']]],
                               [y, [x for x in y if x in ['P', 'N']]])
    hy = evo.predict_proba(X)
    assert len(hy) == 1000
    assert hy.min() >= 0 and hy.max() <= 1


def test_binary_labels_json():
    import json
    X, y = get_data()
    h = dict(NONE=0, N=0, NEU=0, P=1)
    y = [h[x] for x in y]
    evo = EvoMSA(evodag_args=dict(popsize=10, early_stopping_rounds=10),
                 n_jobs=4).fit(X, y)
    hy = evo.predict(X)
    for x in hy:
        print(type(x), str(x))
        _ = json.dumps(dict(klass=str(x)))
    print(_)


def test_EvoMSA_predict_proba_logistic_regression():
    X, y = get_data()
    evo = EvoMSA(logistic_regression=True,
                 evodag_args=dict(popsize=100, early_stopping_rounds=100),
                 n_jobs=4).fit([X, [x for x, y0 in zip(X, y) if y0 in ['P', 'N']]],
                               [y, [x for x in y if x in ['P', 'N']]])
    hy = evo.predict_proba(X)
    assert len(hy) == 1000
    assert hy.min() >= 0 and hy.max() <= 1
    print(hy)


def test_EvoMSA_logistic_regression_params():
    X, y = get_data()
    evo = EvoMSA(logistic_regression=True, logistic_regression_args=dict(C=10),
                 evodag_args=dict(popsize=100, early_stopping_rounds=100),
                 n_jobs=4)
    assert evo._logistic_regression.C == 10

