# Copyright 2024 Mario Graff Guerrero

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
import numpy as np
from EvoMSA.tests.test_base import TWEETS


def test_unique():
    """Test unique function"""
    from EvoMSA.unique import unique

    D = ['hola buenos dias', 'peticion', 'la vida', 'peticion',
         'adios colegas', 'la vida', 'comiendo en el salón', 'adios colegas',
         'nuevo', 'la vida', 'nuevo', 'hola buenos dias']
    actual = np.array([0, 1, 2, 4, 6, 8])
    index = unique(D, lang='es', return_index=True, batch_size=4)
    assert np.all(index == actual)
    index = unique(D, G=['hola buenos dias'],
                   lang='es', return_index=True, batch_size=4)
    assert index[0] != 0
    index = unique(D, lang='es', return_index=False, batch_size=4)
    assert isinstance(index, list) and len(index) == actual.shape[0]
    D = list(tweet_iterator(TWEETS))
    D[-1] = D[0]
    index = unique(D, lang='es', batch_size=11, return_index=True)
    assert index[-1] == 998


    def test_main():
        from EvoMSA.unique import main
        from os.path import isfile
        import os

        class A:
            pass
        
        a = A()
        a.file = [TWEETS]
        a.output = 't.json'
        a.lang = 'es'
        main(a)
        assert isfile('t.json')
        os.remove('j.json')

        