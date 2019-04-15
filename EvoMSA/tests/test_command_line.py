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
from microtc.utils import load_model, tweet_iterator
import sys
import os
from test_base import TWEETS
from nose.tools import assert_almost_equals


def test_train():
    from EvoMSA.base import EvoMSA
    sys.argv = ['EvoMSA', '-ot.model', '-n2',
                '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "time_limit": 5, "n_estimators": 5}',
                TWEETS]
    train(output=True)
    evo = load_model('t.model')
    assert isinstance(evo, EvoMSA)
    os.unlink('t.model')


def test_evo_kwargs():
    from EvoMSA.base import EvoMSA
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "time_limit": 5, "n_estimators": 5}',
                '-ot.model', '--b4msa-kw={"del_dup":false}',
                '-n2', TWEETS, TWEETS]
    train(output=True)
    evo = load_model('t.model')
    assert isinstance(evo, EvoMSA)
    os.unlink('t.model')


def test_predict():
    import numpy as np
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "time_limit": 5, "n_estimators": 5}',
                '-ot.model', '-n2', TWEETS, TWEETS]
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
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "time_limit": 5, "n_estimators": 5}',
                '-ot.model', '--test_set', TWEETS, '-n2', TWEETS]
    train(output=True)
    evo = load_model('t.model')
    assert isinstance(evo, EvoMSA)
    os.unlink('t.model')


def test_utils_b4msa_df():
    from EvoMSA.command_line import utils
    import shutil
    sys.argv = ['EvoMSA', '--kw={"seed": 1}', '-omodel.json', '--b4msa-df', TWEETS]
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


def test_utils_transform():
    import json
    with open('ex.json', 'w') as fpt:
        for x in tweet_iterator(TWEETS):
            x['decision_function'] = x['q_voc_ratio']
            fpt.write(json.dumps(x) + '\n')
    sys.argv = ['EvoMSA', '-ot.model', '-n2',
                '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "time_limit": 5, "n_estimators": 5}',
                TWEETS, TWEETS]
    train(output=True)

    sys.argv = ['EvoMSA', '-mt.model', '-ot.json', '--transform', TWEETS]
    utils()
    os.unlink('t.model')
    vec = [x['vec'] for x in tweet_iterator('t.json')]
    os.unlink('t.json')
    print(len(vec[0]))
    assert len(vec[0]) == 8


def test_raw_outputs():
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "time_limit": 60, "n_estimators": 30}',
                '-ot.model', '-n2', TWEETS, TWEETS]
    train(output=True)
    sys.argv = ['EvoMSA', '--raw-outputs', '-mt.model', '-ot1.json', TWEETS]
    predict()
    df = [x['decision_function'] for x in tweet_iterator('t1.json')]
    assert len(df[0]) == 30 * 4
    os.unlink('t1.json')
    os.unlink('t.model')


def test_decision_function():
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "time_limit": 5, "n_estimators": 5}',
                '-ot.model', '-n2', TWEETS, TWEETS]
    train(output=True)
    sys.argv = ['EvoMSA', '--decision-function', '-mt.model', '-ot1.json', TWEETS]
    predict()
    df = [x['decision_function'] for x in tweet_iterator('t1.json')]
    assert len(df[0]) == 4
    os.unlink('t1.json')
    os.unlink('t.model')


def test_fitness():
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "time_limit": 5, "n_estimators": 5}',
                '-ot.model', '-n2', TWEETS, TWEETS]
    train(output=True)
    sys.argv = ['EvoMSA', '--fitness', 't.model']
    utils()
    os.unlink('t.model')


def test_max_lines():
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "time_limit": 5, "n_estimators": 5}', '-ot.model', '-n2', TWEETS]
    train()
    sys.argv = ['EvoMSA', '-mt.model', '--max-lines', '500', '-ot.json', TWEETS]
    predict()
    os.unlink('t.model')
    os.unlink('t.json')


def test_evo_test_set_shuffle():
    from EvoMSA.base import EvoMSA
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "time_limit": 5, "n_estimators": 5}',
                '-ot.model', '--test_set', 'shuffle', '-n2', TWEETS]
    train(output=True)
    evo = load_model('t.model')
    assert isinstance(evo, EvoMSA)
    os.unlink('t.model')


def test_predict_numbers():
    from sklearn.preprocessing import LabelEncoder
    import json
    d = [x for x in tweet_iterator(TWEETS)]
    le = LabelEncoder().fit([x['klass'] for x in d])
    y = le.transform([x['klass'] for x in d]).tolist()
    with open('ex.json', 'w') as fpt:
        for x, y0 in zip(d, y):
            x['klass'] = y0
            fpt.write(json.dumps(x) + '\n')
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "time_limit": 5, "n_estimators": 5}',
                '--kw={"models": [["EvoMSA.model.Corpus", "EvoMSA.model.Bernulli"]], "TR": false}',
                '-ot.model', '-n1', 'ex.json']
    train(output=True)
    sys.argv = ['EvoMSA', '-mt.model', '-ot1.json', TWEETS]
    predict()


def test_predict_NearestCentroid():
    from sklearn.preprocessing import LabelEncoder
    import json
    d = [x for x in tweet_iterator(TWEETS)]
    le = LabelEncoder().fit([x['klass'] for x in d])
    y = le.transform([x['klass'] for x in d]).tolist()
    with open('ex.json', 'w') as fpt:
        for x, y0 in zip(d, y):
            x['klass'] = y0
            fpt.write(json.dumps(x) + '\n')
    sys.argv = ['EvoMSA',
                '--kw={"evodag_class": "sklearn.neighbors.NearestCentroid", "TR": false, "models": [["EvoMSA.model.Corpus", "EvoMSA.model.Bernulli"]]}',
                '-ot.model', '-n1', 'ex.json']
    train(output=True)
    sys.argv = ['EvoMSA', '-mt.model', '-ot1.json', TWEETS]
    predict()


def test_performance_validation_set():
    # import os
    from EvoMSA.command_line import performance, fitness_vs
    for seed in range(3):
        # if os.path.isfile('t-%s.model' % seed):
        #     continue
        sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "n_estimators": 5}',
                    '--kw={"seed": %s}' % seed, '-ot-%s.model' % seed, '-n1', TWEETS]
        train()
    sys.argv = ['EvoMSA', '-m'] + ['t-%s.model' % seed for seed in range(3)]
    print(fitness_vs((0, 't-0.model')))
    m = performance(output=True)
    assert len(m._p) == 3
    # assert False


def test_performance_validation_set2():
    import os
    from EvoMSA.command_line import performance
    for seed in range(4):
        print('haciendo', seed)
        if os.path.isfile('t-%s.model' % seed):
            continue
        sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "time_limit": 5, "n_estimators": 3}',
                    '--kw={"seed": %s}' % seed, '-ot-%s.model' % seed, '-n1', TWEETS]
        train()
    sys.argv = ['EvoMSA', '-n2', '-m'] + ['t-%s.model' % seed for seed in range(2)] + ['-'] + ['t-%s.model' % seed for seed in range(2, 4)]
    m = performance(output=True)
    assert len(m._p) == 2


def test_performance_public_set():
    import os
    from EvoMSA.command_line import performance
    for seed in range(4):
        if os.path.isfile('t-%s.model' % seed):
            continue
        sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "time_limit": 5, "n_estimators": 3}',
                    '--kw={"seed": %s}' % seed, '-ot-%s.model' % seed, '-n1', TWEETS]
        train(output=True)
    for seed in range(4):
        if os.path.isfile('t-%s.predict' % seed):
            continue
        sys.argv = ['EvoMSA', '-mt-%s.model' % seed, '-ot-%s.predict' % seed, TWEETS]
        predict()
    for score in ['macroF1', 'macroRecall', 'macroPrecision', 'accuracy']:
        sys.argv = ['EvoMSA', '--score', score, '-n2', '-y', TWEETS] + ['t-%s.predict' % seed for seed in range(2)] + ['-'] + ['t-%s.predict' % seed for seed in range(2, 4)]
        m = performance(output=True)
        assert len(m._p) == 2


def test_list_of_text():
    import os
    import json
    from EvoMSA.command_line import train
    with open('t.json', 'w') as fpt:
        for x in tweet_iterator(TWEETS):
            x['text'] = [x['text'], x['text']]
            fpt.write(json.dumps(x) + '\n')
    sys.argv = ['EvoMSA', '-ot.model', '-n2',
                '--kw={"models": [["EvoMSA.model.Corpus", "EvoMSA.model.Bernulli"]], "lang": "es", "TR": false}',
                '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10, "time_limit": 5, "n_estimators": 5}',
                't.json']
    train()
    os.unlink('t.json')

