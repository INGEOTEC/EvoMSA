# Copyright 2020 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_cache():
    from EvoMSA.utils import Cache
    import hashlib

    cache = Cache("hola")
    cache.append("bb")
    cache.append(Cache)
    cache.append(Cache(None))
    ll = [hashlib.md5("bb".encode()).hexdigest(), "Cache", "Cache"]
    for i, j in zip(cache, ll):
        print(i)
        assert i == "hola-%s" % j


def test_cache_cl():
    from EvoMSA.utils import Cache
    from sklearn.svm import LinearSVC
    import hashlib

    cache = Cache("hola")
    cache.append("bb", LinearSVC)
    cache.append(Cache, LinearSVC)
    cache.append(Cache(None), LinearSVC)
    ll = [hashlib.md5("bb".encode()).hexdigest(), "Cache", "Cache"]
    for i, j in zip(cache.ml_train(), ll):
        print(i)
        assert i == "hola-%s-%s" % (j, "LinearSVC")

    for i, j in zip(cache.ml_kfold(), ll):
        print(i)
        assert i == "hola-%s-%s-K" % (j, "LinearSVC")


def test_confidence_interval():
    from EvoMSA.utils import ConfidenceInterval
    from EvoMSA.tests.test_base import get_data

    X, y = get_data()
    kw = dict(stacked_method="sklearn.naive_bayes.GaussianNB") 
    ci = ConfidenceInterval(X, y, evomsa_kwargs=kw)
    result = ci.estimate()
    assert len(result) == 2


def test_download():
    from EvoMSA.utils import download
    _ = download("b4msa_Es.tm", force=True)


def test_load_bow():
    from EvoMSA.utils import load_bow
    bow = load_bow(lang='en')
    repr = bow['hi']
    assert len(repr) == 7


def test_emoji_information():
    from EvoMSA.utils import emoji_information
    info = emoji_information()
    assert info['💧']['number'] == 3905


def test_load_emoji():
    from EvoMSA.utils import load_emoji, emoji_information
    info = emoji_information()
    emojis = load_emoji(lang='es')
    assert isinstance(emojis, list)
    assert len(emojis) == len(info)


def test_dataset_information():
    from EvoMSA.utils import dataset_information
    info = dataset_information(lang='es')
    assert len(info) >= 21


def test_load_dataset():
    from EvoMSA.utils import load_dataset, load_bow
    import numpy as np
    bow = load_bow(lang='en')
    ds = load_dataset(lang='en', name='HA', k=0)
    X = bow.transform(['this is funny'])
    df = ds.decision_function(X)    
    np.testing.assert_almost_equal(df[0], -0.389922806003241)
    ds.labels = None        