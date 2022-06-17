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


def test_ThumbsUpDown():
    from EvoMSA.ConceptModelling.thumbs_up_down import ThumbsUpDown
    from EvoMSA.ConceptModelling.thumbs_up_down import _SPANISH
    thumbs = ThumbsUpDown(lang=_SPANISH, stemming=False)
    _ = thumbs['adoracion XxX fervor vergazo']
    assert _ == (2, 1)
