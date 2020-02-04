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
    
