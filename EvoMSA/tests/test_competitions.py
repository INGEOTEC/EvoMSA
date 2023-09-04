# Copyright 2023 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from EvoMSA.tests.test_base import TWEETS
from EvoMSA.competitions import Comp2023
from EvoMSA.tests.test_base import get_data
import numpy as np


def instance():
    return Comp2023(lang='es', voc_size_exponent=13)


def test_bow():
    D, y = get_data()
    comp2023 = instance()
    comp2023.bow().fit(D, y)


def test_bow_voc_selection():
    D, y = get_data()
    comp2023 = instance()
    comp2023.bow_voc_selection().fit(D, y)


def test_bow_training_set():
    D, y = get_data()
    comp2023 = instance()
    comp2023.bow_training_set().fit(D, y)


def test_stack_bow_keywords_emojis():
    D, y = get_data()
    comp2023 = instance()
    comp2023.stack_bow_keywords_emojis(D, y).fit(D, y)


def test_stack_bow_keywords_emojis_voc_selection():
    D, y = get_data()
    comp2023 = instance()
    comp2023.stack_bow_keywords_emojis_voc_selection(D, y).fit(D, y) 


def test_stack_bow():
    D, y = get_data()
    comp2023 = instance()
    comp2023.stack_bows()


def test_stack_2_bow_keywords():
    D, y = get_data()
    comp2023 = instance()
    comp2023.stack_2_bow_keywords(D, y)


def test_stack_2_bow_tailored_keywords():
    D, y = get_data()
    comp2023 = instance()
    comp2023.tailored = 'es_2.4.9_emojis_most_common_by_type_13.json.gz'
    comp2023.stack_2_bow_tailored_keywords(D, y)


def test_stack_2_bow_all_keywords():
    D, y = get_data()
    comp2023 = instance()
    comp2023.stack_2_bow_all_keywords(D, y)


def test_stack_2_bow_tailored_all_keywords():
    D, y = get_data()
    comp2023 = instance()
    comp2023.tailored = 'es_2.4.9_emojis_most_common_by_type_13.json.gz'
    comp2023.stack_2_bow_tailored_all_keywords(D, y)


def test_stack_3_bows():
    D, y = get_data()
    comp2023 = instance()
    comp2023.stack_3_bows()


def test_stack_3_bows_tailored_keywords():
    D, y = get_data()
    comp2023 = instance()
    comp2023.tailored = 'es_2.4.9_emojis_most_common_by_type_13.json.gz'
    comp2023.stack_3_bows_tailored_keywords(D, y)


def test_stack_3_bows_tailored_all_keywords():
    D, y = get_data()
    comp2023 = instance()
    comp2023.tailored = 'es_2.4.9_emojis_most_common_by_type_13.json.gz'
    comp2023.stack_3_bow_tailored_all_keywords(D, y)


def test_Comp2023_iter():
    cnt = 0
    assert sum([1 for x in Comp2023(lang='es', tailored='X')]) == 13
    assert sum([1 for x in Comp2023(lang='it', tailored='X')]) == 10
    assert sum([1 for x in Comp2023(lang='es')]) == 9
    assert sum([1 for x in Comp2023(lang='it')]) == 8