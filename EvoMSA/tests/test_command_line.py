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
from EvoMSA.command_line import train
from EvoMSA.command_line import predict
import sys
import gzip
import pickle
import os
from test_base import TWEETS


def test_train():
    from EvoMSA.base import EvoMSA
    sys.argv = ['EvoMSA', '-ot.model', '-n4', TWEETS]
    train(output=True)
    with gzip.open('t.model', 'r') as fpt:
        evo = pickle.load(fpt)
    assert isinstance(evo, EvoMSA)
    os.unlink('t.model')


def test_evo_kwargs():
    from EvoMSA.base import EvoMSA
    sys.argv = ['EvoMSA', '--evodag-kw={"popsize": 10, "early_stopping_rounds": 10}',
                '-ot.model', '-n4', TWEETS, TWEETS]
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
    y = np.array([x['klass'] for x in tweet_iterator(TWEETS)])
    acc = (y == hy).mean()
    print(acc)
    assert acc < 1 and acc > 0.8
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


