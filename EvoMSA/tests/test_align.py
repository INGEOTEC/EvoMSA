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


def labeledDataSet(*args, output=None):
    return LabeledDataSet.create_space(*args, output=output,
                                       emo_option='delete',
                                       token_list=[-3, -1, 4])


def _read_words(lang):
    """Read the words from our aggressive lexicon

    :param lang: Language
    :type lang: str [ar|en|es]
    """

    import os
    from microtc.utils import tweet_iterator
    from EvoMSA import base
    fname = os.path.join(os.path.dirname(base.__file__), 'conf', 'aggressiveness.%s' % lang)
    corpus = []
    for x in tweet_iterator(fname):
        corpus += x['words']
    return corpus


def test_projection():
    from EvoMSA.align import projection
    from EvoMSA.utils import download
    with StoreDelete(LabeledDataSet.create_space, TWEETS, 'emo-static-es.evoemo') as sd2:
        with StoreDelete(labeledDataSet, TWEETS, 'emo-static-en.evoemo') as sd1:
            res = projection(download('emo-static-es.evoemo'),
                             download('emo-static-en.evoemo'),
                             _read_words('es'),
                             _read_words('en'))
            assert res.ndim == 2
            assert res.shape[0] == res.shape[1]
