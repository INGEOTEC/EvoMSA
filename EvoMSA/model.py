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
from b4msa.textmodel import TextModel
from b4msa.classifier import SVC


class BaseTextModel(object):
    def __init__(self, corpus, **kwargs):
        pass

    def __getitem__(self, x):
        pass


class Identity(BaseTextModel):
    def __getitem__(self, x):
        return x


class BaseClassifier(object):
    def __init__(self, random_state=0):
        pass

    def fit(self, X, y):
        pass

    def decision_function(self, X):
        pass


class B4MSATextModel(TextModel, BaseTextModel):
    def __getitem__(self, x):
        if x is None:
            x = ''
        return TextModel.__getitem__(self, str(x))


class B4MSAClassifier(SVC, BaseClassifier):
    def __init__(self, random_state=0):
        SVC.__init__(self, model=None, random_state=random_state)

    def decision_function(self, X):
        _ = SVC.decision_function(self, X)
        _[_ > 1] = 1
        _[_ < -1] = -1
        return _




