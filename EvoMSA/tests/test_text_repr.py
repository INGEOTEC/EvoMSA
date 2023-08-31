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
from microtc.utils import tweet_iterator
import numpy as np


def test_BoW_train_predict_decision_function():
    from EvoMSA.text_repr import BoW
    D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es')
    hy = bow.train_predict_decision_function(D)
    assert isinstance(hy, np.ndarray)
    assert hy.shape[0] == len(D)
    bow = BoW(lang='es')


def test_BoW_predict():
    from EvoMSA.text_repr import BoW
    D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es').fit(D)
    hy = bow.predict(D)
    assert hy.shape[0] == len(D)
    bow = BoW(lang='es', v1=True).fit(D)
    hy = bow.predict(D)
    assert hy.shape[0] == len(D)


def test_BoW_decision_function():
    from EvoMSA.text_repr import BoW
    D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es').fit(D)
    hy = bow.decision_function(D)
    assert hy.shape[0] == len(D)
    bow = BoW(lang='es').fit([x for x in D if x['klass'] in ['P', 'N']])
    hy = bow.decision_function(D)
    assert hy.shape[0] == len(D)


def test_BoW_key():
    from EvoMSA.text_repr import BoW
    D = list(tweet_iterator(TWEETS))
    O = BoW(lang='es').transform(D)    
    X = [dict(klass=x['klass'], premise=x['text']) for x in D]    
    bow = BoW(lang='es', key='premise')
    assert abs(bow.transform(X) - O).sum() == 0
    X = [dict(klass=x['klass'], premise=x['text'], conclusion=x['text']) for x in D]
    bow = BoW(lang='es', key=['premise', 'conclusion'])
    assert abs(bow.transform(X) - O * 2).sum() == 0


def test_DenseBoW_transform():
    from EvoMSA.text_repr import TextRepresentations
    D = list(tweet_iterator(TWEETS))
    text_repr = TextRepresentations(lang='es', 
                                    voc_size_exponent=13,
                                    keyword=False, 
                                    emoji=False)
    X = text_repr.transform(D)
    assert X.shape[0] == len(D)
    assert len(text_repr.text_representations) == X.shape[1]


def test_DenseBoW_fit():
    from EvoMSA.text_repr import DenseBoW
    D = list(tweet_iterator(TWEETS))
    text_repr = DenseBoW(lang='es', 
                         voc_size_exponent=13,
                         keyword=False, 
                         emoji=False).fit(D)
    text_repr.predict(['Buen dia'])


def test_DenseBoW_key():
    from EvoMSA.text_repr import DenseBoW
    D = list(tweet_iterator(TWEETS))
    O = DenseBoW(lang='es', unit_vector=False,
                 voc_size_exponent=13,
                 keyword=False, emoji=False).transform(D)    
    X = [dict(klass=x['klass'], premise=x['text'], conclusion=x['text']) for x in D]
    bow = DenseBoW(lang='es', unit_vector=False,
                   voc_size_exponent=13,
                   keyword=False, emoji=False, 
                   key=['premise', 'conclusion'])
    assert abs(bow.transform(X) - O * 2).sum() == 0    


def test_StackGeneralization():
    from EvoMSA.text_repr import StackGeneralization, BoW, DenseBoW
    D = list(tweet_iterator(TWEETS))
    text_repr = StackGeneralization(lang='es',
                                    decision_function_models=[BoW(lang='es')],
                                    transform_models=[DenseBoW(lang='es', voc_size_exponent=13,
                                    keyword=False, 
                                    emoji=False)]).fit(D)
    assert text_repr.predict(['Buen dia'])[0] == 'P'


def test_StackGeneralization_train_predict_decision_function():
    from EvoMSA.text_repr import StackGeneralization, BoW, DenseBoW
    D = list(tweet_iterator(TWEETS))
    text_repr = StackGeneralization(lang='es',
                                    decision_function_models=[BoW(lang='es')],
                                    transform_models=[DenseBoW(lang='es', keyword=False, 
                                    voc_size_exponent=13,
                                    emoji=False)])
    hy = text_repr.train_predict_decision_function(D)
    assert hy.shape[0] == len(D)
    D1 = [x for x in D if x['klass'] in ['P', 'N']]
    hy = text_repr.train_predict_decision_function(D1)
    assert hy.shape[1] == 2


def test_DenseBoW_tr_setter():
    from EvoMSA.text_repr import DenseBoW
    D = list(tweet_iterator(TWEETS))
    text_repr = DenseBoW(lang='es', 
                         keyword=False, 
                         voc_size_exponent=13,
                         emoji=False)
    tr = text_repr.text_representations
    text_repr.text_representations = tr[:3]
    assert text_repr.transform(['Buen dia']).shape[1] == 3

def test_BoW_setter():
    from EvoMSA.text_repr import BoW
    D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es')
    bow.bow = True
    assert bow._bow


def test_DenseBoW_names():
    from EvoMSA.text_repr import DenseBoW
    text_repr = DenseBoW(lang='es', 
                         voc_size_exponent=13,
                         keyword=False, 
                         emoji=False)
    X = text_repr.transform(['buenos dias'])
    assert X.shape[1] == len(text_repr.names)


def test_DenseBoW_select():
    from EvoMSA.text_repr import DenseBoW
    text_repr = DenseBoW(lang='es', 
                         voc_size_exponent=13,
                         keyword=False, 
                         emoji=False)
    text_repr.select([1, 10, 11])
    X = text_repr.transform(['buenos dias'])
    assert X.shape[1] == len(text_repr.names)


def test_BoW_pretrain_False():
    from EvoMSA.text_repr import BoW
    D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es', pretrain=False,
              b4msa_kwargs=dict(max_dimension=True,
                                token_max_filter=2**10)).fit(D)
    X = bow.transform(D)
    assert X.shape[1] == 2**10
    _ = [dict(tt=x['text'], klass=x['klass']) for x in D]
    bow = BoW(lang='es', pretrain=False, key='tt',
              b4msa_kwargs=dict(max_dimension=True,
                                token_max_filter=2**10)).fit(_)
    X = bow.transform(_)
    assert X.shape[1] == 2**10
    _ = [dict(tt=x['text'], klass=x['klass'], tt2=x['text']) for x in D]
    bow = BoW(lang='es', pretrain=False, key=['tt', 'tt2'],
              b4msa_kwargs=dict(max_dimension=True,
                                token_max_filter=2**10)).fit(_)
    X = bow.transform(_)
    assert X.shape[1] == 2**10


def test_DenseBoW_keyword():
    from EvoMSA.text_repr import DenseBoW
    text_repr = DenseBoW(lang='es', keyword=True,
                         v1=True,
                         emoji=False, dataset=False)
    X = text_repr.transform(['hola'])
    assert 2113 == X.shape[1]


def test_BoW_label_key():
    from EvoMSA.text_repr import BoW
    D = list(tweet_iterator(TWEETS))
    _ = [dict(tt=x['text'], label=x['klass']) for x in D]
    bow = BoW(lang='es', pretrain=False, key='tt',
              label_key='label',  
              b4msa_kwargs=dict(max_dimension=True,
                                token_max_filter=2**10)).fit(_)
    assert bow.predict(['buenos días'])[0] == 'P'


def test_DenseBoW_select2():
    from EvoMSA.text_repr import DenseBoW
    D = list(tweet_iterator(TWEETS))    
    text_repr = DenseBoW(lang='es', 
                         emoji=False,
                         keyword=False,
                         voc_size_exponent=13,
                         n_jobs=1)
    n_names = len(text_repr.names)
    text_repr.select(D=D)
    assert len(text_repr.names) < n_names
    text_repr = DenseBoW(lang='es', 
                         emoji=False,
                         keyword=False,
                         voc_size_exponent=13,
                         n_jobs=1).select(D=D)
    assert isinstance(text_repr, DenseBoW)


def test_BoW_names():
    from EvoMSA.text_repr import BoW
    bow = BoW(lang='es')
    X = bow.transform(['hola'])
    assert len(bow.names) == X.shape[1]


def test_DenseBoW_unit():
    from EvoMSA.text_repr import DenseBoW
    D = list(tweet_iterator(TWEETS))    
    text_repr = DenseBoW(lang='es', 
                         emoji=False,
                         keyword=False,
                         n_jobs=1,
                         voc_size_exponent=13,
                         unit_vector=True)
    X = text_repr.transform(['buenos días', 'adios'])
    
    _ = np.sqrt((X ** 2).sum(axis=1))
    np.testing.assert_almost_equal(_, np.array([1, 1]))
    text_repr = DenseBoW(lang='es', 
                         emoji=False,
                         keyword=False,
                         n_jobs=1,
                         voc_size_exponent=13,
                         key=['text', 'text'],
                         unit_vector=True)
    X = text_repr.transform([dict(text='buenos días')])
    _ = np.sqrt((X ** 2).sum(axis=1))
    np.testing.assert_almost_equal(_, 1)


def test_BoW_property():
    from EvoMSA.text_repr import BoW
    bow = BoW()
    bow.kfold_class = '!'
    bow.kfold_kwargs = '*'
    assert bow._kfold_instance == '!' and bow._kfold_kwargs == '*'
    bow.estimator_class = '1'
    bow.estimator_kwargs = '2'
    assert bow._estimator_class == '1' and bow._estimator_kwargs == '2'
    bow.decision_function_name = '3'
    assert bow._decision_function == '3'
    bow.key = '4'
    bow.label_key = '5'
    assert bow._key == '4' and bow._label_key == '5'


def test_config_regressor():
    from EvoMSA.text_repr import BoW, config_regressor
    D = list(tweet_iterator(TWEETS))
    y = [x['klass'] for x in D]
    labels = {v: k for k, v in enumerate(np.unique(y))}
    y_c = np.array([labels[x['klass']] for x in D])
    bow = config_regressor(BoW(lang='es')).fit(D, y_c)
    hy = bow.predict(D)
    assert np.unique(hy).shape[0] > len(labels)


def test_DenseBoW_extend():
    from EvoMSA.text_repr import DenseBoW
    from EvoMSA.utils import MICROTC, Linear
    from EvoMSA import base
    from microtc.utils import tweet_iterator
    from os.path import isfile, dirname, join
    lang = 'es'
    text_repr = DenseBoW(lang=lang, keyword=True, v1=True,
                         emoji=False, dataset=False)
    text_repr = DenseBoW(lang=lang, keyword=False, v1=True,
                         emoji=False, dataset=False)
    diroutput = join(dirname(base.__file__), 'models')
    name = 'keywords'
    fname = join(diroutput, f'{lang}_{name}_muTC2.4.2.json.gz')
    _ = [Linear(**x) for x in tweet_iterator(fname)]
    text_repr.text_representations_extend(_)
    X = text_repr.transform(['hola'])
    assert 2113 == X.shape[1]


def test_DenseBoW_emojis_v2():
    from EvoMSA.text_repr import DenseBoW
    lang = 'ca'
    text_repr = DenseBoW(lang=lang, keyword=False,
                         voc_size_exponent=13,
                         emoji=True, dataset=False)
    X = text_repr.transform(['xxx'])
    assert X.shape[0] and X.shape[1] == 136


def test_BoW_cache():
    from EvoMSA.text_repr import BoW
    # D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es')
    X1 = bow.transform(['buenos dias'])

    bow2 = BoW(lang='es')
    bow2.cache = X1
    assert bow2._cache is not None
    X2 = bow2.transform(['XXX'])
    assert abs(X1 - X2).sum() == 0


def test_DenseBoW_cache():
    from EvoMSA.text_repr import BoW, DenseBoW
    # D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es', voc_size_exponent=13)
    tr = DenseBoW(keyword=False,
                  dataset=False,
                  voc_size_exponent=13,
                  lang='es')
    X = bow.transform(['buenos dias'])
    X1 = tr.transform(['buenos dias'])
    tr.cache = X
    X2 = tr.transform(['xxx'])
    assert np.fabs(X1 - X2).sum() == 0


def test_BoW_weights():
    from EvoMSA.text_repr import BoW
    bow = BoW(lang='es', voc_size_exponent=13)
    assert len(bow.names) == len(bow.weights)


def test_DenseBoW_weights():
    from EvoMSA.text_repr import DenseBoW
    bow = DenseBoW(lang='es',
                   keyword=False,
                   dataset=False,
                   voc_size_exponent=13)
    assert len(bow.names) == bow.weights.shape[0]
    assert len(bow.names) == bow.bias.shape[0]


def test_DenseBoW_keywords_v2():
    from EvoMSA.text_repr import DenseBoW
    lang = 'ca'
    text_repr = DenseBoW(lang=lang, keyword=True,
                         voc_size_exponent=13,
                         emoji=False, dataset=False)
    X = text_repr.transform(['xxx'])        
    assert X.shape[0] == 1 and X.shape[1] == 2022


def test_DenseBoW_fromjson(): 
    from EvoMSA.text_repr import DenseBoW
    from EvoMSA.utils import MICROTC
    from EvoMSA import base
    from os.path import join, dirname, isfile

    lang = 'ca'
    text_repr = DenseBoW(lang=lang, keyword=True,
                                    voc_size_exponent=13,
                                    emoji=False, dataset=False)
    func = 'most_common_by_type'
    d = 13
    name = 'keywords'
    filename = f'{lang}_{MICROTC}_{name}_{func}_{d}.json.gz'
    diroutput = join(dirname(base.__file__), 'models')
    path = join(diroutput, filename)
    assert isfile(path)
    text_repr2 = DenseBoW(lang=lang,
                                     keyword=False,
                                     dataset=False, emoji=False,
                                     voc_size_exponent=d).fromjson(path)
    for a, b in zip(text_repr.names, text_repr2.names):
        assert a == b
    text = ['hola']
    assert np.all(text_repr.transform(text) == text_repr2.transform(text))


def test_BoW_b4msa_kwargs():
    from EvoMSA.text_repr import BoW
    bow = BoW(b4msa_kwargs=dict(XXX='YYY'))
    assert bow.b4msa_kwargs['XXX'] == 'YYY'


def test_DenseBoW_skip_dataset(): 
    from EvoMSA.text_repr import DenseBoW
    from EvoMSA.utils import MICROTC
    from EvoMSA import base
    from os.path import join, dirname, isfile

    lang = 'es'
    text_repr = DenseBoW(lang=lang, keyword=False,
                                    voc_size_exponent=13,
                                    emoji=False, dataset=True)
    keys = set(text_repr.names[:3])
    length = len(text_repr.names)
    text_repr = DenseBoW(lang=lang, keyword=False,
                                    voc_size_exponent=13,
                                    emoji=False, dataset=True,
                                    skip_dataset=keys)
    assert (length - 3) == len(text_repr.names)


def test_DenseBoW_extend2():
    from EvoMSA.text_repr import DenseBoW
    from EvoMSA.utils import MICROTC

    lang = 'es'
    name = 'emojis'
    func = 'most_common_by_type'
    d = 13
    text_repr = DenseBoW(lang=lang,
                         keyword=False,
                         voc_size_exponent=d,
                         emoji=True, dataset=False,
                         n_jobs=-1)
    url = f'{lang}_{MICROTC}_{name}_{func}_{d}.json.gz'
    text_repr2 = DenseBoW(lang=lang, 
                          keyword=False,
                          voc_size_exponent=d,
                          emoji=False, dataset=False,
                          n_jobs=-1)
    text_repr2.text_representations_extend(url)
    for a, b in zip(text_repr.names, text_repr2.names):
        assert a == b


def test_DenseBoW_dataset():
    from EvoMSA.text_repr import DenseBoW
    dense = DenseBoW(lang='it', emoji=False, keyword=False)


def test_DenseBoW_select_bug():
    from EvoMSA.text_repr import DenseBoW
    from EvoMSA.utils import MICROTC
    D = list(tweet_iterator(TWEETS))
    pos = [x for x in D if x['klass'] == 'P']
    neg = [x for x in D if x['klass'] == 'N']
    lang = 'es'
    name = 'emojis'
    func = 'most_common_by_type'
    d = 13
    text_repr = DenseBoW(lang=lang,
                         keyword=False,
                         voc_size_exponent=d,
                         emoji=True, dataset=False,
                         n_jobs=-1)
    text_repr.select(D=pos + neg[:1]).fit(D)


def test_BoW_get_params():
    from EvoMSA.text_repr import BoW
    bow = BoW(lang='es')
    res = bow.get_params()
    assert res['label_key'] == 'klass'


def test_BoW_clone():
    from EvoMSA.text_repr import BoW
    from sklearn.base import clone
    D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='en').fit(D)
    bow2 = clone(bow)
    try:
        bow2.estimator_instance
        assert False
    except AttributeError:
        pass
    assert bow2.lang == 'en'


def test_DenseBoW_get_params():
    from EvoMSA.text_repr import DenseBoW
    lang = 'es'
    name = 'emojis'
    func = 'most_common_by_type'
    d = 13
    text_repr = DenseBoW(lang=lang,
                         keyword=False,
                         voc_size_exponent=d,
                         emoji=True, dataset=False,
                         n_jobs=-1)    
    res = text_repr.get_params()
    assert res['emoji'] and res['voc_selection'] == 'most_common_by_type'


def test_DenseBoW_clone():
    from EvoMSA.text_repr import DenseBoW
    from sklearn.base import clone    
    lang = 'es'
    name = 'emojis'
    func = 'most_common_by_type'
    d = 13
    text_repr = DenseBoW(lang=lang,
                         keyword=False,
                         voc_size_exponent=d,
                         emoji=True, dataset=False,
                         n_jobs=-1).select(subset=[0, 1, 2])
    cl = clone(text_repr)
    assert len(cl.text_representations) == 3


def test_StackGeneralization_get_params():
    from EvoMSA.text_repr import DenseBoW, StackGeneralization, BoW
    lang = 'es'
    name = 'emojis'
    func = 'most_common_by_type'
    d = 13
    dense = DenseBoW(lang=lang,
                     keyword=False,
                     voc_size_exponent=d,
                     emoji=True, dataset=False,
                     n_jobs=-1).select(subset=[0, 1, 2])
    bow = BoW(lang=lang)
    stack = StackGeneralization([bow, dense])
    res = stack.get_params()
    assert res['decision_function_name'] == 'predict_proba'


def test_StackGeneralization_clone():
    from EvoMSA.text_repr import DenseBoW, StackGeneralization, BoW
    from sklearn.base import clone
    lang = 'es'
    name = 'emojis'
    func = 'most_common_by_type'
    d = 13
    D = list(tweet_iterator(TWEETS))
    dense = DenseBoW(lang=lang,
                     keyword=False,
                     voc_size_exponent=d,
                     emoji=True, dataset=False,
                     n_jobs=-1).select(subset=[0, 1, 2]).fit(D)
    bow = BoW(lang=lang)
    stack = StackGeneralization([bow, dense])
    stack2 = clone(stack)
    res = stack.get_params()
    try:
        _ = stack2.decision_function_models[1].estimator_instance
        assert _ == '12'
    except AttributeError:
        pass