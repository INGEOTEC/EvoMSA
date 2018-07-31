# Copyright 2018 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from test_base import get_data
from EvoMSA.base import EvoMSA as evomsa
from nose.tools import assert_almost_equals
import numpy as np


def test_calibration_predict():
    X, y = get_data()
    y = np.array(y)
    evo = evomsa(evodag_args=dict(n_estimators=3, popsize=10, time_limit=15,
                                  early_stopping_rounds=10), seed=0,
                 n_jobs=1, probability_calibration=True).fit(X, y)
    pr = evo.predict_proba(X)
    evo._evodag_model._m.models[0]._probability_calibration = None
    pr2 = evo.predict_proba(X)
    assert np.fabs(pr - pr2).sum() > 0


def test_calibration_predict_2classes():
    X, y = get_data()
    h = dict(NONE='N', NEU='P', N='N', P='P')
    y = [h[x] for x in y]
    y = np.array(y)
    evo = evomsa(evodag_args=dict(n_estimators=3, popsize=10, time_limit=15,
                                  early_stopping_rounds=10), seed=0,
                 n_jobs=1, probability_calibration=True).fit(X, y)
    pr = evo.predict_proba(X)
    [assert_almost_equals(x, 1) for x in pr.sum(axis=1)]
    evo._evodag_model._m.models[0]._probability_calibration = None
    pr2 = evo.predict_proba(X)
    assert np.fabs(pr - pr2).sum() > 0
    print(pr2[:3])
    print(pr[:3])
    assert pr.shape[1] == 2


def test_calibration_predict_2classes_single():
    from EvoMSA.calibration import Calibration
    X, y = get_data()
    h = dict(NONE='N', NEU='P', N='N', P='P')
    y = [h[x] for x in y]
    y = np.array(y)
    evo = evomsa(evodag_args=dict(n_estimators=3, popsize=10, time_limit=15,
                                  early_stopping_rounds=10), seed=0,
                 n_jobs=1).fit(X, y)
    X = evo.transform(X)
    df = evo._evodag_model._m._decision_function_raw(X)
    y = evo._le.transform(y)
    c = Calibration().fit(df[0], y)
    proba = c.predict_proba(df[0])
    assert proba.shape[1] == 2
    [assert_almost_equals(x, 1) for x in proba.sum(axis=1)]


def test_calibration_predict_single():
    from EvoMSA.calibration import Calibration
    X, y = get_data()
    y = np.array(y)
    evo = evomsa(evodag_args=dict(n_estimators=3, popsize=10, time_limit=15,
                                  early_stopping_rounds=10), seed=0,
                 n_jobs=1).fit(X, y)
    X = evo.transform(X)
    df = evo._evodag_model._m._decision_function_raw(X)
    y = evo._le.transform(y)
    c = Calibration().fit(df[0], y)
    proba = c.predict_proba(df[0])
    assert proba.shape[1] == 4
    [assert_almost_equals(x, 1) for x in proba.sum(axis=1)]
    

def test_calibration_weight():
    from EvoMSA.calibration import Calibration
    y = np.concatenate((np.zeros(10), np.ones(20)))
    c = Calibration()
    w = c.weight(y)
    assert (w[y == 0]).sum() == (w[y == 1]).sum()
    
    
