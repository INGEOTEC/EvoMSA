# Copyright 2022 Mario Graff Guerrero

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


def test_EvoDAG_decision_function():
    from EvoMSA.evodag import EvoDAG, TextRepresentations
    D = list(tweet_iterator(TWEETS))
    class _EvoDAG(TextRepresentations):
        def estimator(self):
            return EvoDAG(n_estimators=2, 
                          max_training_size=100)    
    evodag = _EvoDAG(keyword=False, emoji=False,
                     v1=True,
                     decision_function_name='decision_function').fit(D)
    hy = evodag.decision_function(D)    
    assert hy.shape[0] == 1000 and hy.shape[1] == 4


def test_EvoDAG_predict():
    from EvoMSA.evodag import EvoDAG, TextRepresentations
    D = list(tweet_iterator(TWEETS))
    class _EvoDAG(TextRepresentations):
        def estimator(self):
            return EvoDAG(n_estimators=2, 
                          max_training_size=100)    
    evodag = _EvoDAG(keyword=False, v1=True,
                     emoji=False).fit(D)
    hy = evodag.predict(D)
    assert (hy == [x['klass'] for x in D]).mean() > 0.25
    

def test_BoW_train_predict_decision_function():
    from EvoMSA.evodag import BoW
    D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es')
    hy = bow.train_predict_decision_function(D)
    assert isinstance(hy, np.ndarray)
    assert hy.shape[0] == len(D)
    bow = BoW(lang='es')


def test_BoW_predict():
    from EvoMSA.evodag import BoW
    D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es').fit(D)
    hy = bow.predict(D)
    assert hy.shape[0] == len(D)
    bow = BoW(lang='es', v1=True).fit(D)
    hy = bow.predict(D)
    assert hy.shape[0] == len(D)


def test_BoW_decision_function():
    from EvoMSA.evodag import BoW
    D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es').fit(D)
    hy = bow.decision_function(D)
    assert hy.shape[0] == len(D)
    bow = BoW(lang='es').fit([x for x in D if x['klass'] in ['P', 'N']])
    hy = bow.decision_function(D)
    assert hy.shape[0] == len(D)


def test_BoW_key():
    from EvoMSA.evodag import BoW
    D = list(tweet_iterator(TWEETS))
    O = BoW(lang='es').transform(D)    
    X = [dict(klass=x['klass'], premise=x['text']) for x in D]    
    bow = BoW(lang='es', key='premise')
    assert abs(bow.transform(X) - O).sum() == 0
    X = [dict(klass=x['klass'], premise=x['text'], conclusion=x['text']) for x in D]
    bow = BoW(lang='es', key=['premise', 'conclusion'])
    assert abs(bow.transform(X) - O * 2).sum() == 0


def test_TextRepresentations_transform():
    from EvoMSA.evodag import TextRepresentations
    D = list(tweet_iterator(TWEETS))
    text_repr = TextRepresentations(lang='es', 
                                    voc_size_exponent=13,
                                    keyword=False, 
                                    emoji=False)
    X = text_repr.transform(D)
    assert X.shape[0] == len(D)
    assert len(text_repr.text_representations) == X.shape[1]


def test_TextRepresentations_fit():
    from EvoMSA.evodag import TextRepresentations
    D = list(tweet_iterator(TWEETS))
    text_repr = TextRepresentations(lang='es', 
                                    voc_size_exponent=13,
                                    keyword=False, 
                                    emoji=False).fit(D)
    text_repr.predict(['Buen dia'])


def test_TextRepresentations_key():
    from EvoMSA.evodag import TextRepresentations
    D = list(tweet_iterator(TWEETS))
    O = TextRepresentations(lang='es', unit_vector=False,
                            voc_size_exponent=13,
                            keyword=False, emoji=False).transform(D)    
    X = [dict(klass=x['klass'], premise=x['text'], conclusion=x['text']) for x in D]
    bow = TextRepresentations(lang='es', unit_vector=False,
                              voc_size_exponent=13,
                              keyword=False, emoji=False, key=['premise', 'conclusion'])
    assert abs(bow.transform(X) - O * 2).sum() == 0    


def test_StackGeneralization():
    from EvoMSA.evodag import StackGeneralization, BoW, TextRepresentations
    D = list(tweet_iterator(TWEETS))
    text_repr = StackGeneralization(lang='es',
                                    decision_function_models=[BoW(lang='es')],
                                    transform_models=[TextRepresentations(lang='es', 
                                                                          voc_size_exponent=13,
                                                                          keyword=False, 
                                                                          emoji=False)]).fit(D)
    assert text_repr.predict(['Buen dia'])[0] == 'P'


def test_StackGeneralization_train_predict_decision_function():
    from EvoMSA.evodag import StackGeneralization, BoW, TextRepresentations
    D = list(tweet_iterator(TWEETS))
    text_repr = StackGeneralization(lang='es',
                                    decision_function_models=[BoW(lang='es')],
                                    transform_models=[TextRepresentations(lang='es', 
                                                                          keyword=False, 
                                                                          voc_size_exponent=13,
                                                                          emoji=False)])
    hy = text_repr.train_predict_decision_function(D)
    assert hy.shape[0] == len(D)
    D1 = [x for x in D if x['klass'] in ['P', 'N']]
    hy = text_repr.train_predict_decision_function(D1)
    assert hy.shape[1] == 2


def test_TextRepresentations_tr_setter():
    from EvoMSA.evodag import TextRepresentations
    D = list(tweet_iterator(TWEETS))
    text_repr = TextRepresentations(lang='es', 
                                    keyword=False, 
                                    voc_size_exponent=13,
                                    emoji=False)
    tr = text_repr.text_representations
    text_repr.text_representations = tr[:3]
    assert text_repr.transform(['Buen dia']).shape[1] == 3

def test_BoW_setter():
    from EvoMSA.evodag import BoW
    D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es')
    bow.bow = True
    assert bow._bow


def test_TextRepresentations_names():
    from EvoMSA.evodag import TextRepresentations
    text_repr = TextRepresentations(lang='es', 
                                    voc_size_exponent=13,
                                    keyword=False, 
                                    emoji=False)
    X = text_repr.transform(['buenos dias'])
    assert X.shape[1] == len(text_repr.names)


def test_TextRepresentations_select():
    from EvoMSA.evodag import TextRepresentations
    text_repr = TextRepresentations(lang='es', 
                                    voc_size_exponent=13,
                                    keyword=False, 
                                    emoji=False)
    text_repr.select([1, 10, 11])
    X = text_repr.transform(['buenos dias'])
    assert X.shape[1] == len(text_repr.names)


def test_BoW_pretrain_False():
    from EvoMSA.evodag import BoW
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


def test_TextRepresentations_keyword():
    from EvoMSA.evodag import TextRepresentations
    text_repr = TextRepresentations(lang='es', keyword=True,
                                    v1=True,
                                    emoji=False, dataset=False)
    X = text_repr.transform(['hola'])
    assert 2113 == X.shape[1]


def test_BoW_label_key():
    from EvoMSA.evodag import BoW
    D = list(tweet_iterator(TWEETS))
    _ = [dict(tt=x['text'], label=x['klass']) for x in D]
    bow = BoW(lang='es', pretrain=False, key='tt',
              label_key='label',  
              b4msa_kwargs=dict(max_dimension=True,
                                token_max_filter=2**10)).fit(_)
    assert bow.predict(['buenos días'])[0] == 'P'


def test_TextRepresentations_select2():
    from EvoMSA.evodag import TextRepresentations
    D = list(tweet_iterator(TWEETS))    
    text_repr = TextRepresentations(lang='es', 
                                    emoji=False,
                                    keyword=False,
                                    voc_size_exponent=13,
                                    n_jobs=1)
    n_names = len(text_repr.names)
    text_repr.select(D=D)
    assert len(text_repr.names) < n_names
    text_repr = TextRepresentations(lang='es', 
                                    emoji=False,
                                    keyword=False,
                                    voc_size_exponent=13,
                                    n_jobs=1).select(D=D)
    assert isinstance(text_repr, TextRepresentations)


def test_BoW_names():
    from EvoMSA.evodag import BoW
    bow = BoW(lang='es')
    X = bow.transform(['hola'])
    assert len(bow.names) == X.shape[1]


def test_TextRepresentations_unit():
    from EvoMSA.evodag import TextRepresentations
    D = list(tweet_iterator(TWEETS))    
    text_repr = TextRepresentations(lang='es', 
                                    emoji=False,
                                    keyword=False,
                                    n_jobs=1,
                                    voc_size_exponent=13,
                                    unit_vector=True)
    X = text_repr.transform(['buenos días', 'adios'])
    
    _ = np.sqrt((X ** 2).sum(axis=1))
    np.testing.assert_almost_equal(_, np.array([1, 1]))
    text_repr = TextRepresentations(lang='es', 
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
    from EvoMSA.evodag import BoW
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
    from EvoMSA.evodag import BoW, config_regressor
    D = list(tweet_iterator(TWEETS))
    y = [x['klass'] for x in D]
    labels = {v: k for k, v in enumerate(np.unique(y))}
    y_c = np.array([labels[x['klass']] for x in D])
    bow = config_regressor(BoW(lang='es')).fit(D, y_c)
    hy = bow.predict(D)
    assert np.unique(hy).shape[0] > len(labels)


def test_TextRepresentations_extend():
    from EvoMSA.evodag import TextRepresentations
    from EvoMSA.utils import MICROTC, Linear
    from EvoMSA import base
    from microtc.utils import tweet_iterator
    from os.path import isfile, dirname, join
    lang = 'es'
    text_repr = TextRepresentations(lang=lang, keyword=True, v1=True,
                                    emoji=False, dataset=False)
    text_repr = TextRepresentations(lang=lang, keyword=False, v1=True,
                                    emoji=False, dataset=False)                                    
    diroutput = join(dirname(base.__file__), 'models')
    name = 'keywords'
    fname = join(diroutput, f'{lang}_{name}_muTC2.4.2.json.gz')
    _ = [Linear(**x) for x in tweet_iterator(fname)]
    text_repr.text_representations_extend(_)
    X = text_repr.transform(['hola'])
    assert 2113 == X.shape[1]


def test_TextRepresentations_emojis_v2():
    from EvoMSA.evodag import TextRepresentations
    lang = 'ca'
    text_repr = TextRepresentations(lang=lang, keyword=False,
                                    voc_size_exponent=13,
                                    emoji=True, dataset=False)
    X = text_repr.transform(['xxx'])
    assert X.shape[0] and X.shape[1] == 136


def test_BoW_cache():
    from EvoMSA.evodag import BoW
    # D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es')
    X1 = bow.transform(['buenos dias'])

    bow2 = BoW(lang='es')
    bow2.cache = X1
    assert bow2._cache is not None
    X2 = bow2.transform(['XXX'])
    assert abs(X1 - X2).sum() == 0


def test_TextRepresentations_cache():
    from EvoMSA.evodag import BoW, TextRepresentations
    # D = list(tweet_iterator(TWEETS))
    bow = BoW(lang='es', voc_size_exponent=13)
    tr = TextRepresentations(keyword=False,
                             dataset=False,
                             voc_size_exponent=13,
                             lang='es')
    X = bow.transform(['buenos dias'])
    X1 = tr.transform(['buenos dias'])
    tr.cache = X
    X2 = tr.transform(['xxx'])
    assert np.fabs(X1 - X2).sum() == 0


def test_BoW_weights():
    from EvoMSA.evodag import BoW
    bow = BoW(lang='es', voc_size_exponent=13)
    assert len(bow.names) == len(bow.weights)


def test_TextRepresentations_weights():
    from EvoMSA.evodag import TextRepresentations
    bow = TextRepresentations(lang='es',
                              keyword=False,
                              dataset=False,                              
                              voc_size_exponent=13)
    assert len(bow.names) == bow.weights.shape[0]
    assert len(bow.names) == bow.bias.shape[0]


def test_TextRepresentations_keywords_v2():
    from EvoMSA.evodag import TextRepresentations
    lang = 'ca'
    text_repr = TextRepresentations(lang=lang, keyword=True,
                                    voc_size_exponent=13,
                                    emoji=False, dataset=False)
    X = text_repr.transform(['xxx'])        
    assert X.shape[0] == 1 and X.shape[1] == 503 