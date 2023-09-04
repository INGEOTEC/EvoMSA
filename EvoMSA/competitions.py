from EvoMSA import BoW, DenseBoW, StackGeneralization
from EvoMSA.utils import Linear, b4msa_params
from microtc.utils import tweet_iterator, load_model, save_model
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from glob import glob
from os.path import isfile
from collections import defaultdict
from os.path import basename
import numpy as np


class Comp2023(object):
    """Configurations tested on the 2023 Competitions.
    
    :param lang: language, see :ref:`BoW`
    :type: lang: str
    :param voc_size_exponent: Vocabulary size. default=17, i.e., :math:`2^{17}`.
    :type voc_size_exponent: int
    :param tailored: :ref:`DenseBoW` created with keywords selected from a problem.
    :type tailored: str
    """
    def __init__(self, lang='es', 
                 voc_size_exponent=17,
                 tailored=None) -> None:
        self.voc_size_exponent = voc_size_exponent
        self.lang = lang
        self.tailored = tailored

    def __iter__(self):
        systems = [self.bow,
                   self.bow_voc_selection,
                   self.bow_training_set,
                   self.stack_bow_keywords_emojis,
                   self.stack_bow_keywords_emojis_voc_selection,
                   self.stack_bows,
                   self.stack_2_bow_keywords,
                   self.stack_2_bow_tailored_keywords,
                   self.stack_2_bow_all_keywords,
                   self.stack_2_bow_tailored_all_keywords,
                   self.stack_3_bows,
                   self.stack_3_bows_tailored_keywords,
                   self.stack_3_bow_tailored_all_keywords]
        otra = []
        for i in systems:
            if self.lang not in ['ar', 'es', 'en'] and '_all_' in i.__name__:
                continue
            if '_tailored_' in i.__name__ and self.tailored is None:
                continue
            otra.append(i)
        systems = otra                    
        return systems.__iter__()
    
    def bow(self, D=None, y=None):
        """Pre-trained :ref:`BoW` where the tokens are selected based on a normalized frequency w.r.t. its type, i.e., bigrams, words, and q-grams of characters.

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW
        >>> D = list(tweet_iterator(TWEETS))
        >>> bow = BoW(lang='es').fit(D)
        >>> hy = bow.predict(['Buenos días'])
        """

        return BoW(lang=self.lang,
                   voc_size_exponent=self.voc_size_exponent)
    
    def bow_voc_selection(self, D=None, y=None):
        """Pre-trained :ref:`BoW` where the tokens correspond to the most frequent ones.

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW
        >>> D = list(tweet_iterator(TWEETS))
        >>> bow = BoW(lang='es',
                      voc_selection='most_common').fit(D)
        >>> hy = bow.predict(['Buenos días'])
        """

        return BoW(lang=self.lang,
                   voc_size_exponent=self.voc_size_exponent,
                   voc_selection='most_common')
    
    def bow_training_set(self, D=None, y=None):
        """:ref:`BoW` trained with the training set; the number of tokens corresponds to all the tokens in the set.

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW
        >>> from EvoMSA.utils import b4msa_params
        >>> D = list(tweet_iterator(TWEETS))
        >>> params = b4msa_params(lang='es')
        >>> del params['token_max_filter']
        >>> del params['max_dimension']                
        >>> bow = BoW(lang='es',
                      pretrain=False, 
                      b4msa_kwargs=params).fit(D)
        >>> hy = bow.predict(['Buenos días'])
        """

        params = b4msa_params(lang=self.lang)
        del params['token_max_filter']
        del params['max_dimension']
        return BoW(lang=self.lang,
                   pretrain=False, 
                   b4msa_kwargs=params)
    
    def stack_bow_keywords_emojis(self, D, y=None):
        """Stack generalization (:ref:`StackGeneralization`) approach where the base classifiers are the :ref:`BoW`, the :py:attr:`emoji`, and the :py:attr:`keywords` of :ref:`DenseBoW`.

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW, DenseBoW, StackGeneralization
        >>> D = list(tweet_iterator(TWEETS))
        >>> bow = BoW(lang='es')
        >>> keywords = DenseBoW(lang='es',
                                emoji=False,
                                dataset=False).select(D=D)
        >>> emojis = DenseBoW(lang='es',
                              keyword=False,
                              dataset=False).select(D=D)
        >>> st = StackGeneralization(decision_function_models=[bow,
                                                               keywords,
                                                               emojis]).fit(D)
        >>> hy = st.predict(['Buenos días'])
        """

        bow = self.bow()
        keywords = DenseBoW(lang=self.lang,
                            voc_size_exponent=self.voc_size_exponent,
                            emoji=False,
                            dataset=False).select(D=D, y=y)
        emojis = DenseBoW(lang=self.lang,
                          voc_size_exponent=self.voc_size_exponent,
                          keyword=False,
                          dataset=False).select(D=D, y=y)
        return StackGeneralization(decision_function_models=[bow,
                                                             keywords,
                                                             emojis])
    
    def stack_bow_keywords_emojis_voc_selection(self, D, y=None):
        """Stack generalization (:ref:`StackGeneralization`) approach where the base classifiers are the :ref:`BoW`, the :py:attr:`emoji`, and the :py:attr:`keywords` of :ref:`DenseBoW`. The tokens in these models were selected based on a normalized frequency w.r.t. its type, i.e., bigrams, words, and q-grams of characters.

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW, DenseBoW, StackGeneralization
        >>> D = list(tweet_iterator(TWEETS))
        >>> bow = BoW(lang='es',
                      voc_selection='most_common')
        >>> keywords = DenseBoW(lang='es',
                                voc_selection='most_common',
                                emoji=False,
                                dataset=False).select(D=D)
        >>> emojis = DenseBoW(lang='es',
                              voc_selection='most_common',
                              keyword=False,
                              dataset=False).select(D=D)
        >>> st = StackGeneralization(decision_function_models=[bow,
                                                               keywords,
                                                               emojis]).fit(D)
        >>> hy = st.predict(['Buenos días'])        
        """

        bow = self.bow_voc_selection()
        keywords = DenseBoW(lang=self.lang,
                            voc_selection='most_common',
                            voc_size_exponent=self.voc_size_exponent,
                            emoji=False,
                            dataset=False).select(D=D, y=y)
        emojis = DenseBoW(lang=self.lang,
                          voc_selection='most_common',
                          voc_size_exponent=self.voc_size_exponent,
                          keyword=False,
                          dataset=False).select(D=D, y=y)
        return StackGeneralization(decision_function_models=[bow,
                                                             keywords,
                                                             emojis])

    def stack_bows(self, D=None, y=None):
        """Stack generalization approach where the base classifiers are :ref:`BoW` with the two token selection procedures set in the parameter :py:attr:`voc_selection`.

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW, StackGeneralization
        >>> D = list(tweet_iterator(TWEETS))
        >>> bow = BoW(lang='es')
        >>> bow_voc = BoW(lang='es',
                          voc_selection='most_common')
        >>> st = StackGeneralization(decision_function_models=[bow,
                                                               bow_voc]).fit(D)
        >>> hy = st.predict(['Buenos días'])
        """
        bow = self.bow()
        bow_voc = self.bow_voc_selection()
        return  StackGeneralization(decision_function_models=[bow, bow_voc])

    def stack_2_bow_keywords(self, D, y=None):
        """Stack generalization approach where with four base classifiers. These correspond to two :ref:`BoW` and two dense :ref:`DenseBoW` (emojis and keywords), where the difference in each is the procedure used to select the tokens, i.e., the most frequent or normalized frequency (i.e., :py:attr:`voc_selection`).

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW, DenseBoW, StackGeneralization
        >>> D = list(tweet_iterator(TWEETS))
        >>> bow = BoW(lang='es')
        >>> bow_voc = BoW(lang='es',
                          voc_selection='most_common')
        >>> keywords = DenseBoW(lang='es',
                                dataset=False).select(D=D)
        >>> keywords_voc = DenseBoW(lang='es',
                                    voc_selection='most_common',
                                    dataset=False).select(D=D)
        >>> st = StackGeneralization(decision_function_models=[bow, bow_voc,
                                                               keywords,
                                                               keywords_voc]).fit(D)
        >>> hy = st.predict(['Buenos días'])
        """

        bow = self.bow()
        keywords = DenseBoW(lang=self.lang,
                            voc_size_exponent=self.voc_size_exponent,
                            dataset=False).select(D=D, y=y)
        bow2 = self.bow_voc_selection()
        keywords2 = DenseBoW(lang=self.lang,
                             voc_size_exponent=self.voc_size_exponent,
                             voc_selection='most_common',
                             dataset=False).select(D=D, y=y)
        return StackGeneralization(decision_function_models=[bow, bow2,
                                                             keywords,
                                                             keywords2])

    def stack_2_bow_tailored_keywords(self, D, y=None):
        """Stack generalization approach where with four base classifiers. These correspond to two :ref:`BoW` and two :ref:`DenseBoW` (emojis and keywords), where the difference in each is the procedure used to select the tokens, i.e., the most frequent or normalized frequency. The second difference is that the dense representation with normalized frequency also includes models for the most discriminant words selected by a BoW classifier in the training set. We refer to these latter representations as **tailored keywords.**
        
        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW, DenseBoW, StackGeneralization
        >>> D = list(tweet_iterator(TWEETS))
        >>> bow = BoW(lang='es')
        >>> keywords = DenseBoW(lang='es',
                                dataset=False).select(D=D)
        >>> tailored = 'IberLEF2023_DAVINCIS_task1_Es.json.gz'
        >>> keywords.text_representations_extend(tailored)
        >>> bow_voc = BoW(lang='es',
                          voc_selection='most_common')
        >>> keywords_voc = DenseBoW(lang='es',
                                    voc_selection='most_common',
                                    dataset=False).select(D=D)
        >>> st = StackGeneralization(decision_function_models=[bow, bow_voc,
                                                               keywords,
                                                               keywords_voc]).fit(D)
        >>> hy = st.predict(['Buenos días'])        
        """

        bow = self.bow()
        keywords = DenseBoW(lang='es',
                            voc_size_exponent=self.voc_size_exponent,
                            dataset=False)
        keywords.text_representations_extend(self.tailored)
        keywords.select(D=D, y=y)
        bow2 = self.bow_voc_selection()
        keywords2 = DenseBoW(lang='es',
                             voc_size_exponent=self.voc_size_exponent,
                             voc_selection='most_common',
                             dataset=False).select(D=D, y=y)
        return StackGeneralization(decision_function_models=[bow, bow2,
                                                             keywords,
                                                             keywords2])

    def stack_2_bow_all_keywords(self, D, y=None):
        """Stack generalization approach where with four base classifiers equivalently to :ref:`StackGeneralization` using :ref:`BoW` and :ref:`DenseBoW` with and without :py:attr:`voc_selection` where the difference is that the dense representations include the models created with the human-annotated datasets.
        
        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW, DenseBoW, StackGeneralization
        >>> D = list(tweet_iterator(TWEETS))        
        >>> bow = BoW(lang='es')
        >>> keywords = DenseBoW(lang='es')
        >>> sel = [k for k, v in enumerate(keywords.names)
                   if v not in ['davincis2022_1'] or 'semeval2023' not in v]
        >>> keywords.select(sel).select(D=D)
        >>> bow_voc = BoW(lang='es', voc_selection='most_common')
        >>> keywords_voc = DenseBoW(lang='es',
                                    voc_selection='most_common').select(sel).select(D=D)
        >>> st = StackGeneralization(decision_function_models=[bow,
                                                               bow_voc,
                                                               keywords,
                                                               keywords_voc]).fit(D)
        >>> hy = st.predict(['Buenos días'])
        """

        bow = self.bow()
        keywords = DenseBoW(lang='es',
                            voc_size_exponent=self.voc_size_exponent)
        sel = [k for k, v in enumerate(keywords.names)
               if v not in ['davincis2022_1'] or 'semeval2023' not in v]
        keywords.select(sel).select(D=D, y=y)
        bow2 = self.bow_voc_selection()
        keywords2 = DenseBoW(lang=self.lang,
                             voc_size_exponent=self.voc_size_exponent,
                             voc_selection='most_common').select(sel).select(D=D, y=y)
        return StackGeneralization(decision_function_models=[bow,
                                                             bow2,
                                                             keywords,
                                                             keywords2])

    def stack_2_bow_tailored_all_keywords(self, D, y=None):
        """Stack generalization approach where with four base classifiers equivalently to :ref:`StackGeneralization` using :ref:`BoW` and :ref:`DenseBoW` with and without :py:attr:`voc_selection` where the difference is that the dense representation with normalized frequency also includes the tailored keywords.

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW, DenseBoW, StackGeneralization
        >>> D = list(tweet_iterator(TWEETS))        
        >>> bow = BoW(lang='es')
        >>> keywords = DenseBoW(lang='es')
        >>> tailored = 'IberLEF2023_DAVINCIS_task1_Es.json.gz'
        >>> keywords.text_representations_extend(tailored)        
        >>> sel = [k for k, v in enumerate(keywords.names)
                   if v not in ['davincis2022_1'] or 'semeval2023' not in v]
        >>> keywords.select(sel).select(D=D)
        >>> bow_voc = BoW(lang='es', voc_selection='most_common')
        >>> keywords_voc = DenseBoW(lang='es',
                                    voc_selection='most_common').select(sel).select(D=D)
        >>> st = StackGeneralization(decision_function_models=[bow,
                                                               bow_voc,
                                                               keywords,
                                                               keywords_voc]).fit(D)
        >>> hy = st.predict(['Buenos días'])        
        """
        bow = self.bow()
        keywords = DenseBoW(lang=self.lang,
                            voc_size_exponent=self.voc_size_exponent)
        keywords.text_representations_extend(self.tailored)        
        sel = [k for k, v in enumerate(keywords.names) 
               if v not in ['davincis2022_1'] or 'semeval2023' not in v]
        keywords.select(sel).select(D=D, y=y)
        bow_voc = self.bow_voc_selection()
        keywords_voc = DenseBoW(lang=self.lang,
                                voc_size_exponent=self.voc_size_exponent,
                                voc_selection='most_common').select(sel).select(D=D, y=y)
        return StackGeneralization(decision_function_models=[bow, 
                                                             bow_voc,
                                                             keywords,
                                                             keywords_voc])
    
    def stack_3_bows(self, D=None, y=None):
        """Stack generalization approach with three base classifiers. All of them are :ref:`BoW`; the first two correspond pre-trained :ref:`BoW` with the two token selection procedures described previously (i.e., BoW default parameters and BoW using :py:attr:`voc_selection`), and the latest is a :ref:`BoW` trained on the training set.

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW, StackGeneralization
        >>> from EvoMSA.utils import b4msa_params        
        >>> D = list(tweet_iterator(TWEETS))
        >>> bow = BoW(lang='es')
        >>> bow_voc = BoW(lang='es',
                          voc_selection='most_common')
        >>> params = b4msa_params(lang='es')
        >>> del params['token_max_filter']
        >>> del params['max_dimension']                
        >>> bow_train = BoW(lang='es',
                            pretrain=False, 
                            b4msa_kwargs=params).fit(D)                          
        >>> st = StackGeneralization(decision_function_models=[bow,
                                                               bow_voc,
                                                               bow_train]).fit(D)
        >>> hy = st.predict(['Buenos días'])                
        """

        bow = self.bow()
        bow_voc = self.bow_voc_selection()
        bow_train = self.bow_training_set()
        return StackGeneralization(decision_function_models=[bow,
                                                             bow_voc,
                                                             bow_train])
    
    def stack_3_bows_tailored_keywords(self, D, y=None):
        """Stack generalization approach with five base classifiers. The first corresponds to a :ref:`BoW` trained on the training set, and the rest are used in :py:func:`EvoMSA.competitions.Comp2023.stack_2_bow_tailored_keywords`.

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW, DenseBoW, StackGeneralization
        >>> from EvoMSA.utils import b4msa_params        
        >>> D = list(tweet_iterator(TWEETS))
        >>> bow = BoW(lang='es')
        >>> keywords = DenseBoW(lang='es',
                                dataset=False).select(D=D)
        >>> tailored = 'IberLEF2023_DAVINCIS_task1_Es.json.gz'
        >>> keywords.text_representations_extend(tailored)
        >>> bow_voc = BoW(lang='es',
                          voc_selection='most_common')
        >>> keywords_voc = DenseBoW(lang='es',
                                    voc_selection='most_common',
                                    dataset=False).select(D=D)
        >>> params = b4msa_params(lang='es')
        >>> del params['token_max_filter']
        >>> del params['max_dimension']                
        >>> bow_train = BoW(lang='es',
                            pretrain=False, 
                            b4msa_kwargs=params)                                    
        >>> st = StackGeneralization(decision_function_models=[bow, bow_voc,
                                                               bow_train,
                                                               keywords,
                                                               keywords_voc]).fit(D)
        >>> hy = st.predict(['Buenos días'])                
        """
        st = self.stack_2_bow_tailored_keywords(D, y=y)
        st.decision_function_models.append(self.bow_training_set())
        return st

    def stack_3_bow_tailored_all_keywords(self, D, y=None):
        """Stack generalization approach with five base classifiers. The first corresponds to a :ref:`BoW` trained on the training set, and the rest are used in :py:func:`EvoMSA.competitions.Comp2023.stack_2_bow_tailored_all_keywords`. 

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW, DenseBoW, StackGeneralization
        >>> from EvoMSA.utils import b4msa_params 
        >>> D = list(tweet_iterator(TWEETS))        
        >>> bow = BoW(lang='es')
        >>> keywords = DenseBoW(lang='es')
        >>> tailored = 'IberLEF2023_DAVINCIS_task1_Es.json.gz'
        >>> keywords.text_representations_extend(tailored)        
        >>> sel = [k for k, v in enumerate(keywords.names)
                   if v not in ['davincis2022_1'] or 'semeval2023' not in v]
        >>> keywords.select(sel).select(D=D)
        >>> bow_voc = BoW(lang='es', voc_selection='most_common')
        >>> keywords_voc = DenseBoW(lang='es',
                                    voc_selection='most_common').select(sel).select(D=D)
        >>> params = b4msa_params(lang='es')
        >>> del params['token_max_filter']
        >>> del params['max_dimension']                
        >>> bow_train = BoW(lang='es',
                            pretrain=False, 
                            b4msa_kwargs=params)                                    
        >>> st = StackGeneralization(decision_function_models=[bow,
                                                               bow_voc,
                                                               bow_train,
                                                               keywords,
                                                               keywords_voc]).fit(D)
        >>> hy = st.predict(['Buenos días'])       
        """

        st = self.stack_2_bow_tailored_all_keywords(D, y=y)
        st.decision_function_models.append(self.bow_training_set())
        return st

