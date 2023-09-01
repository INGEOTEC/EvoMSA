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
    """Configurations tested on the 2023 Competitions"""
    def __init__(self, lang='es', voc_size_exponent=17) -> None:
        self.voc_size_exponent = voc_size_exponent
        self.lang = lang

    def bow(self):
        """Pre-trained :py:class:`EvoMSA.text_repr.BoW` where the tokens are selected based on a normalized frequency w.r.t. its type, i.e., bigrams, words, and q-grams of characters.

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW
        >>> D = list(tweet_iterator(TWEETS))
        >>> bow = BoW(lang='es').fit(D)
        >>> hy = bow.predict(['Buenos días'])
        """

        return BoW(lang=self.lang,
                   voc_size_exponent=self.voc_size_exponent)
    
    def bow_voc_selection(self):
        """Pre-trained :py:class:`EvoMSA.text_repr.BoW` where the tokens correspond to the most frequent ones.

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
    
    def bow_training_set(self):
        """:py:class`EvoMSA.text_repr.BoW` trained with the training set; the number of tokens corresponds to all the tokens in the set.

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
        """Stack generalization (:py:class:`EvoMSA.text_repr.StackGeneralization`) approach where the base classifiers are the :py:class:`EvoMSA.text_repr.BoW`, the :py:attr:`emoji`, and the :py:attr:`keywords` of :py:class:`EvoMSA.text_repr.DenseBoW`.

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
        """Stack generalization (:py:class:`EvoMSA.text_repr.StackGeneralization`) approach where the base classifiers are the :py:class:`EvoMSA.text_repr.BoW`, the :py:attr:`emoji`, and the :py:attr:`keywords` of :py:class:`EvoMSA.text_repr.DenseBoW`. The tokens in these models were selected based on a normalized frequency w.r.t. its type, i.e., bigrams, words, and q-grams of characters.

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

    def stack_bows(self):
        """Stack generalization approach where the base classifiers are :py:class:`EvoMSA.text_repr.BoW` with the two token selection procedures set in the parameter :py:attr:`voc_selection`.

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
        """Stack generalization approach where with four base classifiers. These correspond to two :py:class:`EvoMSA.text_repr.BoW` and two dense :py:class:`EvoMSA.text_repr.DenseBoW` (emojis and keywords), where the difference in each is the procedure used to select the tokens, i.e., the most frequent or normalized frequency (i.e., :py:attr:`voc_selection`).

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

    def stack_2_bow_tailored_keywords(self, tailored, D, y=None):
        """Stack generalization approach where with four base classifiers. These correspond to two :py:class:`EvoMSA.text_repr.BoW` and two :py:class:`EvoMSA.text_repr.DenseBoW` (emojis and keywords), where the difference in each is the procedure used to select the tokens, i.e., the most frequent or normalized frequency. The second difference is that the dense representation with normalized frequency also includes models for the most discriminant words selected by a BoW classifier in the training set. We refer to these latter representations as **tailored keywords.**
        
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
        keywords.text_representations_extend(tailored)
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
        """Stack generalization approach where with four base classifiers equivalently to :py:class:`EvoMSA.text_repr.StackGeneralization` using :py:class:`EvoMSA.text_repr.BoW` and :py:class:`EvoMSA.text_repr.DenseBoW` with and without :py:attr:`voc_selection` where the difference is that the dense representations include the models created with the human-annotated datasets.
        
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

    def stack_2_bow_tailored_all_keywords(self, tailored, D, y=None):
        """Stack generalization approach where with four base classifiers equivalently to :py:class:`EvoMSA.text_repr.StackGeneralization` using :py:class:`EvoMSA.text_repr.BoW` and :py:class:`EvoMSA.text_repr.DenseBoW` with and without :py:attr:`voc_selection` where the difference is that the dense representation with normalized frequency also includes the tailored keywords.

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
        keywords.text_representations_extend(tailored)        
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