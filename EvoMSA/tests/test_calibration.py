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
import numpy as np


def test_calibration_coef():
    X, y = get_data()
    evo = evomsa(evodag_args=dict(popsize=10, early_stopping_rounds=10, n_estimators=2),
                 n_jobs=2, probability_calibration=True).fit(X, y)
    assert evo
    assert len(evo._calibration_coef) == 2
    for x in evo._calibration_coef:
        assert len(x) == 4


def test_calibration_predict():
    X, y = get_data()
    y = np.array(y)
    evo = evomsa(evodag_args=dict(n_estimators=3, popsize=10,
                                  early_stopping_rounds=10), seed=0,
                 n_jobs=2, probability_calibration=True).fit(X, y)
    pr = evo.predict_proba(X)
    evo._probability_calibration = False
    pr2 = evo.predict_proba(X)
    assert np.fabs(pr - pr2).sum() > 0
