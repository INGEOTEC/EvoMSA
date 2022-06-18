# -*- coding: utf-8 -*-
# Copyright 2018 Sabino Miranda with collaboration of Eric S. Tellez

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from EvoMSA.ConceptModelling.thumbs_up_down import ThumbsUpDown
from EvoMSA.ConceptModelling.thumbs_up_down import _SPANISH
from EvoMSA.ConceptModelling.text_preprocessing import TextPreprocessing, _OPTION_DELETE, _OPTION_GROUP


def test_ThumbsUpDown():
    thumbs = ThumbsUpDown(lang=_SPANISH, stemming=False)
    _ = thumbs['adoracion XxX fervor vergazo']
    assert _ == (2, 1)


def test_ThumbsUpDown_nolang():
    try:
        thumbs = ThumbsUpDown(lang='XX', stemming=False)
    except:
        return
    assert False


def test_ThumbsUpDown_stemming():
    thumbs = ThumbsUpDown(lang=_SPANISH, stemming=False)
    text = thumbs.get_text(dict(text='jugando'))
    thumbs.stemming(text)
    output = thumbs.stemming('@mgraffg')
    assert output == []
    assert thumbs.generate_ar_words('xx') == ['xx']


def test_ThumbsUpDown_getitem():
    thumbs = ThumbsUpDown(lang=_SPANISH, stemming=False)
    assert thumbs[dict(text='hola')] == (0, 0)
    assert thumbs[('hola', ':)')] == (0, 0)


def test_TextPreprocessing():
    ins = TextPreprocessing()
    output = ins.remove_stopwords('hola a todos')
    assert output == 'hola'
    output = ins.text_transform('hola http://a.b', url=_OPTION_DELETE)
    assert output == 'hola'
    output = ins.text_transform('hola http://a.b', url=_OPTION_GROUP)
    assert output == 'hola _url'
    output = ins.text_transform('hola http://a.b', stemming_comp=True)
    assert output == ''
    output = ins.escapeText('(hola)')
    assert output == '\\(hola\\)'



