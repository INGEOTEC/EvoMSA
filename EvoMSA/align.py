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


from microtc.utils import tweet_iterator
import os


def _read_words(lang):
    """Read the words from our aggressive lexicon

    :param lang: Language
    :type lang: str [ar|en|es]
    """
    fname = os.path.join(os.path.dirname(__file__), 'conf', 'aggressiveness.%s' % lang)
    corpus = []
    for x in tweet_iterator(fname):
        corpus += x['words']
    return corpus


def projection(lang_from, lang_to):
    """
    Compute the coefficients to project the output of a Emoji Space in the origin language to the objetive language

    :param lang_from: Origin language
    :type lang_from: str [ar|en|es]
    :param lang_to: Objective language
    :type lang_to: str [ar|en|es]
    """

    model_from = os.path.join(os.path.dirname(__file__), 'models', 'emo-static-%s.evoemo' % lang_from)
    model_to = os.path.join(os.path.dirname(__file__), 'models', 'emo-static-%s.evoemo' % lang_to)
