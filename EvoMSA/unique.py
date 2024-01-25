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
import argparse
from typing import Union, List
import numpy as np
import sys
from microtc.utils import tweet_iterator
import EvoMSA
from EvoMSA.text_repr import BoW
import json


def unique(D: List[Union[dict, list]],
           G: List[Union[dict, list]]=None,
           lang: str='es',
           return_index: bool=False,
           alpha: float=0.95,
           batch_size: int=1024,
           bow_params: dict=None,
           transform=None):
    """Compute the unique elements in a set using :py:class:`~EvoMSA.text_repr.BoW`

    :param D: Texts; in the case, it is a list of dictionaries the text is on the key :py:attr:`BoW.key`
    :type D: List of texts or dictionaries.
    :param G: D must be different than G.
    :type G: List of texts or dictionaries.
    :param lang: Language.
    :type lang: str
    :param return_index: Return the indexes.
    :type return_index: bool
    :param alpha: Value to assert similarity.
    :type alpha: float
    :param batch_size: Batch size
    :type batch_size: int
    :param bow_params: :py:class:`~EvoMSA.text_repr.BoW` params.
    :type bow_params: dict
    :param transform: Function to transform the text, default :py:class:`~EvoMSA.text_repr.BoW`

    >>> from EvoMSA.text_repr import unique
    >>> D = ['hi', 'bye', 'HI.']
    >>> unique(D, lang='en')
    ['hi', 'bye']
    """
    from tqdm import tqdm

    def unique_self(elementos):
        _ = X[elementos]
        sim = np.dot(_, _.T) >= alpha
        indptr = sim.indptr
        indices = sim.indices
        remove = []
        index = np.where(np.diff(indptr) > 1)[0]
        init = indptr[index]
        index = index[index == indices[init]]
        for i, j in zip(index, index+1):
            remove.extend(indices[indptr[i]:indptr[j]][1:].tolist())
        remove = set(remove)
        _ = [i for k, i in enumerate(elementos)
             if k not in remove]
        return np.array(_)
    

    def unique_rest(frst, rest):
        sim = np.dot(X[rest], X[frst].T).max(axis=1).toarray()
        mask = sim.flatten() < alpha
        return rest[mask]

    bow_params = dict() if bow_params is None else bow_params
    bow_params.update(lang=lang)
    if transform is None:
        transform = BoW(**bow_params).transform
    X = transform(D)
    init = 0
    pool = np.arange(len(D))
    with tqdm(total=pool.shape[0]) as _tqdm:
        while init < pool.shape[0]:
            past = pool[:init]    
            elementos = pool[init:init + batch_size]
            ele_size = elementos.shape[0]
            rest = pool[init + batch_size:]
            frst = unique_self(elementos)
            r_size = rest.shape[0]
            rest = unique_rest(frst, rest)
            _tqdm.update(ele_size + r_size - rest.shape[0])
            init = init + frst.shape[0]
            pool = np.r_[past, frst, rest]
    if G is not None:
        G = transform(G)
        sim = np.dot(X[pool], G.T).max(axis=1).toarray()
        mask = sim.flatten() < alpha
        pool = pool[mask]
    if return_index:
        return pool
    return [D[x] for x in pool]


def main(args):
    def output_json(D, file):
        for x in D:
            print(json.dumps(x), file=file)

    filename  = args.file[0]
    lang = args.lang
    D = list(tweet_iterator(filename))
    D = unique(D, lang=lang)
    if args.output is None:
        output_json(D, file=sys.stdout)
    else:
        with open(args.output, 'w') as fpt:
            output_json(D, file=fpt)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EvoMSA unique feature',
                                     prog='EvoMSA.unique')
    parser.add_argument('-v', '--version', action='version',
                        version=f'EvoMSA {EvoMSA.__version__}')
    parser.add_argument('-o', '--output',
                        help='output filename',
                        dest='output', default=None, type=str)
    parser.add_argument('--lang', help='Language (ar | ca | de | en | es | fr | hi | in | it | ja | ko | nl | pl | pt | ru | tl | tr | zh)',
                        type=str, default='es')
    parser.add_argument('file',
                        help='input filename',
                        nargs=1, type=str)
    args = parser.parse_args()
    main(args)
