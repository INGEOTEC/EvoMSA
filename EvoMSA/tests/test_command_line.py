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
from EvoMSA.command_line import train, utils
from EvoMSA.command_line import predict
import sys
import gzip
import pickle
import os
from test_base import TWEETS
from nose.tools import assert_almost_equals


def test_train():
    from EvoMSA.base import EvoMSA
    sys.argv = ['EvoMSA', '-ot.model', '-n4',
                '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10}',
                TWEETS]
    train(output=True)
    with gzip.open('t.model', 'r') as fpt:
        evo = pickle.load(fpt)
    assert isinstance(evo, EvoMSA)
    os.unlink('t.model')
    assert evo._use_ts


def test_evo_kwargs():
    from EvoMSA.base import EvoMSA
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10}',
                '-ot.model', '--b4msa-kw={"del_dup1":false}',
                '-n4', TWEETS, TWEETS]
    train(output=True)
    with gzip.open('t.model', 'r') as fpt:
        evo = pickle.load(fpt)
    assert isinstance(evo, EvoMSA)
    os.unlink('t.model')


def test_predict():
    from b4msa.utils import tweet_iterator
    import numpy as np
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10}',
                '-ot.model', '-n4', TWEETS, TWEETS]
    train(output=True)
    sys.argv = ['EvoMSA', '-mt.model', '-ot1.json', TWEETS]
    predict()
    hy = np.array([x['klass'] for x in tweet_iterator('t1.json')])
    [x['decision_function'] for x in tweet_iterator('t1.json')]
    y = np.array([x['klass'] for x in tweet_iterator(TWEETS)])
    acc = (y == hy).mean()
    print(acc)
    assert acc <= 1 and acc > 0.8
    os.unlink('t1.json')
    os.unlink('t.model')


def test_evo_test_set():
    from EvoMSA.base import EvoMSA
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10}',
                '-ot.model', '--test_set', TWEETS, '-n4', TWEETS]
    train(output=True)
    with gzip.open('t.model', 'r') as fpt:
        evo = pickle.load(fpt)
    assert isinstance(evo, EvoMSA)
    os.unlink('t.model')


def test_evo_parameters():
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10}',
                '-ot.model', '-Pnada.json', '-n4', TWEETS]
    try:
        train(output=True)
    except FileNotFoundError:
        return
    assert False


def test_utils_b4msa_df():
    from EvoMSA.command_line import utils
    from b4msa.utils import tweet_iterator
    import shutil
    sys.argv = ['EvoMSA', '--b4msa-kw={"del_dup1":false}',
                '-omodel.json', '--b4msa-df', TWEETS]
    utils(output=True)
    assert os.path.isfile('model.json')
    sys.argv = ['EvoMSA', '-omodel', '--b4msa-df', '--test_set', TWEETS, TWEETS]
    utils(output=True)
    assert os.path.isdir('model')
    dos = os.path.join('model', 'train.json')
    for a, b in zip(tweet_iterator('model.json'), tweet_iterator(dos)):
        for v, w in zip(a['vec'], b['vec']):
            print(v, w)
            assert_almost_equals(v, w, places=3)
    shutil.rmtree('model')
    os.unlink('model.json')


def test_train_no_use_ts():
    from EvoMSA.base import EvoMSA
    sys.argv = ['EvoMSA', '--no-use-ts', '-ot.model', '-n4',
                '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10}',
                TWEETS, TWEETS]
    train(output=True)
    with gzip.open('t.model', 'r') as fpt:
        evo = pickle.load(fpt)
    assert isinstance(evo, EvoMSA)
    os.unlink('t.model')
    assert not evo._use_ts


def test_train_kw():
    from EvoMSA.base import EvoMSA
    sys.argv = ['EvoMSA', '-ot.model', '-n4',
                '--kw={"logistic_regression": true}',
                '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10}',
                TWEETS]
    train(output=True)
    with gzip.open('t.model', 'r') as fpt:
        evo = pickle.load(fpt)
    assert isinstance(evo, EvoMSA)
    os.unlink('t.model')
    assert evo._logistic_regression is not None


def test_train_exogenous():
    from EvoMSA.base import EvoMSA
    from b4msa.utils import tweet_iterator
    import json
    with open('ex.json', 'w') as fpt:
        for x in tweet_iterator(TWEETS):
            x['decision_function'] = x['q_voc_ratio']
            fpt.write(json.dumps(x) + '\n')
    sys.argv = ['EvoMSA', '-ot.model', '-n4',
                '--kw={"logistic_regression": true}',
                '--exogenous', 'ex.json', 'ex.json',
                '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10}',
                TWEETS]
    train(output=True)
    with gzip.open('t.model', 'r') as fpt:
        evo = pickle.load(fpt)
    assert isinstance(evo, EvoMSA)
    os.unlink('t.model')
    assert evo._logistic_regression is not None
    m = evo._evodag_model.models[0]
    os.unlink('ex.json')
    print(m.nvar)
    assert m.nvar == 6
    assert evo._n_jobs == 4


def test_logistic_regression_params():
    from EvoMSA.base import EvoMSA
    sys.argv = ['EvoMSA', '-ot.model', '-n1',
                '--kw={"logistic_regression": true}',
                '--logistic-regression-kw={"C": 10}',
                '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10}',
                TWEETS]
    train(output=True)
    with gzip.open('t.model', 'r') as fpt:
        evo = pickle.load(fpt)
    assert isinstance(evo, EvoMSA)
    assert evo._logistic_regression.C == 10
    os.unlink('t.model')


def test_utils_transform():
    from b4msa.utils import tweet_iterator
    import json
    with open('ex.json', 'w') as fpt:
        for x in tweet_iterator(TWEETS):
            x['decision_function'] = x['q_voc_ratio']
            fpt.write(json.dumps(x) + '\n')
    sys.argv = ['EvoMSA', '-ot.model', '-n4', '--no-use-ts',
                '--kw={"logistic_regression": true}',
                '--exogenous', 'ex.json', 'ex.json',
                '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10}',
                TWEETS, TWEETS]
    train(output=True)

    sys.argv = ['EvoMSA', '-mt.model', '-ot.json', '--exogenous', 'ex.json', 'ex.json',
                '--transform', TWEETS]
    utils()
    os.unlink('t.model')
    vec = [x['vec'] for x in tweet_iterator('t.json')]
    os.unlink('t.json')
    assert len(vec[0]) == 6


def test_raw_outputs():
    from b4msa.utils import tweet_iterator
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10}',
                '-ot.model', '-n4', TWEETS, TWEETS]
    train(output=True)
    sys.argv = ['EvoMSA', '--raw-outputs', '-mt.model', '-ot1.json', TWEETS]
    predict()
    df = [x['decision_function'] for x in tweet_iterator('t1.json')]
    assert len(df[0]) == 30 * 4
    os.unlink('t1.json')
    os.unlink('t.model')


def test_decision_function():
    from b4msa.utils import tweet_iterator
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10}',
                '-ot.model', '-n4', TWEETS, TWEETS]
    train(output=True)
    sys.argv = ['EvoMSA', '--decision-function', '-mt.model', '-ot1.json', TWEETS]
    predict()
    df = [x['decision_function'] for x in tweet_iterator('t1.json')]
    assert len(df[0]) == 4
    os.unlink('t1.json')
    os.unlink('t.model')


def test_fitness():
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10}',
                '-ot.model', '-n2', TWEETS, TWEETS]
    train(output=True)
    sys.argv = ['EvoMSA', '--fitness', 't.model']
    utils()
    os.unlink('t.model')
    
