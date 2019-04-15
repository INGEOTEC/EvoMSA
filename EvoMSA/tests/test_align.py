# Copyright 2019 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from test_base import StoreDelete, TWEETS
from EvoMSA.model import LabeledDataSet


def test_read_words():
    from EvoMSA.align import _read_words
    for lang in ['ar', 'en', 'es']:
        assert len(_read_words(lang))


def labeledDataSet(*args, output=None):
    return LabeledDataSet.create_space(*args, output=output,
                                       emo_option='delete',
                                       token_list=[-3, -1, 4])


def test_projection():
    from EvoMSA.align import projection
    with StoreDelete(LabeledDataSet.create_space, TWEETS, 'emo-static-es.evoemo') as sd2:
        with StoreDelete(labeledDataSet, TWEETS, 'emo-static-en.evoemo') as sd1:
            res = projection('es', 'en')
            assert res.ndim == 2
            assert res.shape[0] == res.shape[1]
