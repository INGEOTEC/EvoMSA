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
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def projection(model_from, model_to, text_from, text_to):
    """
    Compute the coefficients to project the output of a Emoji Space in the origin language to the objetive language

    :param lang_from: Origin model
    :type lang_from: str
    :param lang_to: Objective model
    :type lang_to: str [ar|en|es]
    :param text_from: Text in the origin language
    :type text_from: list
    :param text_from: Text in the objective language
    :type text_from: list
    """

    from microtc.utils import load_model
    import numpy as np
    from sklearn.neighbors import KDTree
    model_from = load_model(model_from)
    model_to = load_model(model_to)
    vec_from = model_from.transform(text_from)
    vec_to = model_to.transform(text_to)
    done = set()
    output = []
    X = []
    kdtree = KDTree(vec_to, metric='manhattan')
    ss = kdtree.query(vec_from)[1].flatten()
    for k, j in tqdm(enumerate(ss)):
        if j in done:
            continue
        X.append(vec_from[k])
        output.append(vec_to[j])
        done.add(j)
    output = np.stack(output)
    X = np.stack(X)
    return np.linalg.lstsq(X, output, rcond=None)[0]

