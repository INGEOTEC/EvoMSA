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


from EvoMSA.utils import MODEL_LANG
from EvoMSA.utils import load_bow, load_emoji, dataset_information, load_dataset, load_keyword, b4msa_params, Linear
from EvoMSA.model_selection import KruskalFS
from b4msa.textmodel import TextModel
from microtc.weighting import TFIDF
from microtc.utils import tweet_iterator
from joblib import Parallel, delayed
from typing import Union, List, Set, Callable
from sklearn.base import BaseEstimator, clone
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix
import numpy as np


def config_regressor(instance):
    from sklearn.svm import LinearSVR
    from sklearn.model_selection import KFold

    instance.estimator_class = LinearSVR
    instance.estimator_kwargs = dict()
    instance.decision_function_name = 'predict'
    instance.kfold_class = KFold
    return instance


class BoW(BaseEstimator):
    """
    BoW is a bag-of-words text classifier. It is described in 
    "A Simple Approach to Multilingual Polarity Classification in Twitter. 
    Eric S. Tellez, Sabino Miranda-Jiménez, Mario Graff, 
    Daniela Moctezuma, Ranyart R. Suárez, Oscar S. Siordia. 
    Pattern Recognition Letters" and 
    "An Automated Text Categorization Framework based 
    on Hyperparameter Optimization. Eric S. Tellez, Daniela Moctezuma, 
    Sabino Miranda-Jímenez, Mario Graff. 
    Knowledge-Based Systems Volume 149, 1 June 2018." 

    BoW uses, by default, a pre-trained bag-of-words representation. The 
    representation was trained on 4,194,304 (:math:`2^{22}`) tweets
    randomly selected. The pre-trained representations are used
    when the parameters :attr:`lang` and :attr:`pretrain` are
    set; :attr:`pretrain` by default is set to True, and the default 
    language is Spanish (es). The available languages are:
    Arabic (ar), Catalan (ca), German (de), English (en), 
    Spanish (es), French (fr), Hindi (hi), Indonesian (in), 
    Italian (it), Japanese (ja), Korean (ko), Dutch (nl),
    Polish (pl), Portuguese (pt), Russian (ru), Tagalog (tl), 
    Turkish (tr), and Chinese (zh).
    

    :param lang: Language. (ar | ca | de | en | es | fr | hi | in | it | ja | ko | nl | pl | pt | ru | tl | tr | zh), default='es'.
    :type lang: str
    :param voc_size_exponent: Vocabulary size. default=17, i.e., :math:`2^{17}`
    :type voc_size_exponent: int
    :param voc_selection: Vocabulary (most_common_by_type | most_common). default=most_common_by_type
    :type voc_selection: str
    :param key: Key where the text is in the dictionary. (default='text')
    :type key: Union[str, List[str]]
    :param label_key: Key where the response is in the dictionary. (default='klass')
    :type label_key: str
    :param mixer_func: Function to combine the output in case of multiple texts
    :type mixer_func: Callable[[List], csr_matrix]
    :param decision_function_name: Name of the decision function (detaulf='decision_function')
    :type decision_function_name: str
    :param estimator_class: Classifier or Regressor
    :type estimator_class: class
    :param estimator_kwargs: Keyword parameters for the estimator
    :type estimator_kwargs: dict
    :param pretrain: Whether to use a pre-trained representation. default=True.
    :type pretrain: bool
    :param b4msa_kwargs: :py:class:`b4msa.textmodel.TextModel` keyword arguments used to train a bag-of-words representation. default=dict().
    :type b4msa_kwargs: dict
    :param kfold_class: Class of the KFold procedure (default=StratifiedKFold)
    :type kfold_class: class
    :param kfold_kwargs: Keyword parameters for the KFold class
    :type kfold_kwargs: dict
    :param v1: Whether to use version 1 or pretrained representations. default=False
    :type v1: bool
    :param n_jobs: Number of jobs. default=1
    :type n_jobs: int

    >>> from microtc.utils import tweet_iterator
    >>> from EvoMSA.tests.test_base import TWEETS
    >>> from EvoMSA import BoW
    >>> bow = BoW(lang='es').fit(list(tweet_iterator(TWEETS)))
    >>> bow.predict(['Buenos dias']).tolist()
    ['P']
    """
    def __init__(self, lang: str='es',
                 voc_size_exponent: int=17,
                 voc_selection: str='most_common_by_type', 
                 key: Union[str, List[str]]='text',
                 label_key: str='klass',
                 mixer_func: Callable[[List], csr_matrix]=sum,
                 decision_function_name: str='decision_function',
                 estimator_class=LinearSVC,
                 estimator_kwargs=dict(dual=True),
                 pretrain=True,
                 b4msa_kwargs=dict(),
                 kfold_class=StratifiedKFold,
                 kfold_kwargs: dict=dict(random_state=0,
                                         shuffle=True),
                 v1: bool=False,
                 n_jobs: int=1) -> None:
        assert lang is None or lang in MODEL_LANG
        if lang in MODEL_LANG:
            assert voc_size_exponent >= 13 and voc_size_exponent <= 17
            assert voc_selection in ['most_common_by_type', 'most_common']
        self.voc_size_exponent = voc_size_exponent
        self.voc_selection = voc_selection
        self.n_jobs = n_jobs
        self._lang = lang
        self.key = key
        self.label_key = label_key
        self.mixer_func = mixer_func
        self.decision_function_name = decision_function_name
        self.estimator_class = estimator_class
        self.estimator_kwargs = estimator_kwargs
        self.b4msa_kwargs = b4msa_kwargs
        self._pretrain = pretrain
        self.kfold_class = kfold_class
        self.kfold_kwargs = kfold_kwargs
        self._b4msa_estimated = False
        self.v1 = v1

    def fit(self, D: List[Union[dict, list]], 
            y: Union[np.ndarray, None]=None) -> 'BoW':
        """Estimate the parameters of the BoW (:py:func:`BoW.b4msa_fit`)
        and the classifier or regressor (:py:attr:`BoW.estimator_class` - 
        the fitted instance is accesible at :py:attr:`BoW.estimator_instance`) 
        using the dataset (`D`, `y`).

        :param D: Texts; in the case, it is a list of dictionaries the text is on the key :py:attr:`BoW.key`
        :type D: List of texts or dictionaries. 
        :param y: Response variable. The response variable can also be in `D` on the key :py:attr:`BoW.label_key`.
        :type y: Array or None

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW
        >>> import numpy as np
        >>> D = list(tweet_iterator(TWEETS))
        >>> bow = BoW(lang='es').fit(D)                
        """
        if not self.pretrain and not self._b4msa_estimated:
            self.b4msa_fit(D)
        y = self.dependent_variable(D, y=y)
        _ = self.transform(D, y=y)
        self.estimator_instance = self.estimator_class(**self.estimator_kwargs).fit(_, y)
        return self
    
    def transform(self, D: List[Union[List, dict]], y=None) -> csr_matrix:
        """Represent the texts in `D` in the vector space.

        :param D: Texts to be transformed. In the case, it is a list of dictionaries the text is on the key :py:attr:`BoW.key`
        :type D: List of texts or dictionaries. 

        >>> from EvoMSA import BoW
        >>> X = BoW(lang='en').transform(['Hi', 'Good morning'])
        >>> X = BoW(lang='en').transform([dict(text='Hi'), dict(text='Good morning')])
        >>> X.shape
        (2, 131072)
        """
        assert len(D)
        if not self.pretrain:
            assert self._b4msa_estimated
        if self.pretrain and self.cache is not None:
            X = self.cache
            self.cache = None
            return X
        if self.key == 'text' or isinstance(D[0], str):
            return self.bow.transform(D)
        assert isinstance(D[0], dict)
        if isinstance(self.key, str):
            key = self.key
            return self.bow.transform([x[key] for x in D])
        Xs = [self.bow.transform([x[key] for x in D])
              for key in self.key]
        return self.mixer_func(Xs)    

    def predict(self, D: List[Union[dict, list]]) -> np.ndarray:
        """Predict the response variable on the dataset `D`.

        :param D: Texts; in the case, it is a list of dictionaries the text is on the key :py:attr:`BoW.key`
        :type D: List of texts or dictionaries.

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW
        >>> bow = BoW(lang='es').fit(list(tweet_iterator(TWEETS)))
        >>> bow.predict(['Buenos dias']).tolist()
        ['P']                
        """
        _ = self.transform(D)
        return self.estimator_instance.predict(_)

    def decision_function(self, D: List[Union[dict, list]]) -> Union[list, np.ndarray]:
        """Decision function of the estimate response variable in `D`.

        :param D: Texts to be transformed. In the case, it is a list of dictionaries the text is on the key :py:attr:`BoW.key`
        :type D: List of texts or dictionaries.

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW
        >>> bow = BoW(lang='es').fit(list(tweet_iterator(TWEETS)))
        >>> bow.decision_function(['Buenos dias'])
        array([[-1.40547754, -1.01340503, -0.57912244,  0.90450322]])      
        """

        _ = self.transform(D)
        hy = getattr(self.estimator_instance, self.decision_function_name)(_)
        if hy.ndim == 1:
            return np.atleast_2d(hy).T
        return hy

    @property
    def bow(self):
        """Bag of Word text representation.
        
        The following example tokenizes *hi*.
        The notation is the following, the first 'hi' corresponds to the word *hi*.
        Then, there come the q-grams of characters, the token 'q:hi' represents 
        the q-gram *hi*. All the q-grams start with the prefix 'q:'. Finally, 
        the character ~ represents a space.

        >>> bow = BoW(lang='en')
        >>> bow.bow.tokenize(['hi'])
        ['hi', 'q:~h', 'q:hi', 'q:i~', 'q:~hi', 'q:hi~', 'q:~hi~']        
        """
        try:
            bow = self._bow
        except AttributeError:
            if self.pretrain:
                if self.v1:
                    self._bow = load_bow(lang=self.lang, v1=self.v1)
                else:
                    freq = load_bow(lang=self.lang,
                                    d=self.voc_size_exponent, 
                                    func=self.voc_selection)
                    params = b4msa_params(lang=self.lang,
                                        dim=self._voc_size_exponent)
                    params.update(self.b4msa_kwargs)
                    bow = TextModel(**params)
                    tfidf = TFIDF()
                    tfidf.N = freq.update_calls
                    tfidf.word2id, tfidf.wordWeight = tfidf.counter2weight(freq)
                    bow.model = tfidf
                    self._bow = bow
            else:
                self._bow = TextModel(lang=self.lang,
                                      **self.b4msa_kwargs)
            bow = self._bow
        return bow

    @bow.setter
    def bow(self, value):
        self._bow = value

    @property
    def names(self):
        """Vector space components"""
        _names = [None] * len(self.bow.id2token)
        for k, v in self.bow.id2token.items():
            _names[k] = v
        return _names

    @property
    def weights(self):
        """Vector space weights"""
        try:
            return self._weights
        except AttributeError:
            w = [None] * len(self.bow.token_weight)
            for k, v in self.bow.token_weight.items():
                w[k] = v
            self._weights = w
            return self._weights

    @property
    def estimator_instance(self):
        """Estimator - Classifier or Regressor fitted (:py:attr:`BoW.fit`) on the dataset"""
        return self._m

    @estimator_instance.setter
    def estimator_instance(self, m):
        self._m = m

    @property
    def pretrain(self):
        """Whether the to use pre-trained text representations
        
        The parameters of the BoW text representation 
        are :py:attr:`BoW.lang`, :py:attr:`BoW.voc_selection`,
        and :py:attr:`BoW.voc_size_exponent`. The aforementioned parameters are
        not available on Version 1.0 (:py:attr:`BoW.v1`).

        """
        return self._pretrain

    @property
    def lang(self):
        """Language of the pre-trained text representations"""
        return self._lang
    
    @property
    def voc_selection(self):
        """Method used to select the vocabulary"""
        return self._voc_selection
    
    @voc_selection.setter
    def voc_selection(self, value):
        self._voc_selection = value

    @property
    def voc_size_exponent(self):
        """Vocabulary size :math:`2^v`; where :math:`v` is :py:attr:`voc_size_exponent` """
        return self._voc_size_exponent
    
    @voc_size_exponent.setter
    def voc_size_exponent(self, value):
        self._voc_size_exponent = value
    
    @property
    def v1(self):
        """Whether to use the Version 1.0 text representations. 
        This version is only available for Arabic (ar), English (en), and Spanish (es).
        """
        return self._v1
    
    @v1.setter
    def v1(self, value):
        self._v1 = value

    def b4msa_fit(self, D: List[Union[List, dict]]):
        """Estimate the parameters of the BoW (:py:class:`BoW.bow`) 
        in case it is not pretrained (:py:attr:`BoW.pretrain`)
        
        :param D: Dataset
        :type D: List of texts or dictionaries.

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW
        >>> bow = BoW(pretrain=False)
        >>> bow.b4msa_fit(list(tweet_iterator(TWEETS)))
        >>> X = bow.transform(['Hola'])
        >>> X.shape
        (1, 84802)
        """
        assert len(D)
        self._b4msa_estimated = True
        if self.key == 'text' or isinstance(D[0], str):
            return self.bow.fit(D)
        assert isinstance(D[0], dict)
        if isinstance(self.key, str):
            key = self.key
            return self.bow.fit([x[key] for x in D])
        _ = [[x[key] for key in self.key] for x in D]
        return self.bow.fit(_)

    def train_predict_decision_function(self, D: List[Union[dict, list]], 
                                        y: Union[np.ndarray, None]=None) -> np.ndarray:
        """
        Method to compute the kfold predictions on dataset `D` with response `y`

        :param D: Texts to be transformed. In the case, it is a list of dictionaries the text is on the key :py:attr:`BoW.key`
        :type D: List of texts or dictionaries. 
        :param y: Response variable
        :type y: Array or None

        For example, the following code computes the accuracy using k-fold cross-validation on the dataset found on `TWEETS` 

        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> from EvoMSA import BoW
        >>> import numpy as np
        >>> D = list(tweet_iterator(TWEETS))
        >>> bow = BoW(lang='es')
        >>> df = bow.train_predict_decision_function(D)
        >>> df.shape
        (1000, 4)
        >>> hy = df.argmax(axis=1)
        >>> y = np.array([x['klass'] for x in D])
        >>> labels = np.unique(y)
        >>> accuracy = (y == labels[hy]).mean()
        """
        def train_predict(tr, vs):
            m = self.estimator_class(**self.estimator_kwargs).fit(X[tr], y[tr])
            return getattr(m, self.decision_function_name)(X[vs])

        y = self.dependent_variable(D, y=y)
        kf = self.kfold_class(**self.kfold_kwargs)
        kfolds = [x for x in kf.split(D, y)]
        X = self.transform(D, y=y)
        hys = Parallel(n_jobs=self.n_jobs)(delayed(train_predict)(tr, vs)
                                            for tr, vs in kfolds)
        K = np.unique(y).shape[0]
        if hys[0].ndim == 1:
            hy = np.empty((y.shape[0], 1))
            hys = [np.atleast_2d(x).T for x in hys]
        else:
            hy = np.empty((y.shape[0], K))
        for (_, vs), pr in zip(kfolds, hys):
            hy[vs] = pr
        return hy
    
    def dependent_variable(self, D: List[Union[dict, list]], 
                           y: Union[np.ndarray, None]=None) -> np.ndarray:
        """Obtain the response variable

        :param D: Dataset
        :type D: List of texts or dictionaries
        :param y: Response variable
        :type y: Array or None 
        """
        assert isinstance(D, list) and len(D)
        label_key = self.label_key
        if y is None:
            assert isinstance(D[0], dict)
            y = np.array([x[label_key] for x in D])
        assert isinstance(y, np.ndarray)
        return y

    @property
    def cache(self):
        """If the cache is set, it is returned when calling :py:attr:`BoW.transform`; afterward, it is unset."""
        try:
            return self._cache
        except AttributeError:
            return None
        
    @cache.setter
    def cache(self, value):
        self._cache = value

    @property
    def label_key(self):
        """Key where the response is in the dictionary."""
        return self._label_key

    @label_key.setter
    def label_key(self, value):
        self._label_key = value

    @property
    def key(self):
        """Key where the text(s) is(are) in the dictionary."""
        return self._key

    @key.setter
    def key(self, value):
        self._key = value

    @property
    def decision_function_name(self):
        """Name of the estimator's decision function"""
        return self._decision_function

    @decision_function_name.setter
    def decision_function_name(self, value):
        self._decision_function = value

    @property
    def kfold_class(self):
        """Class to produce the kfolds"""
        return self._kfold_instance

    @kfold_class.setter
    def kfold_class(self, value):
        self._kfold_instance = value

    @property
    def kfold_kwargs(self):
        """Keyword arguments of the kfold class"""
        return self._kfold_kwargs

    @kfold_kwargs.setter
    def kfold_kwargs(self, value):
        self._kfold_kwargs = value

    @property
    def estimator_class(self):
        """Class of the classifier or regressor"""
        return self._estimator_class

    @estimator_class.setter
    def estimator_class(self, value):
        self._estimator_class = value

    @property
    def estimator_kwargs(self):
        """Keyword arguments of the estimator :py:class:`BoW.estimator_class`"""
        return self._estimator_kwargs

    @estimator_kwargs.setter
    def estimator_kwargs(self, value):
        self._estimator_kwargs = value        

    @property
    def b4msa_kwargs(self):
        """Keyword arguments of B4MSA"""
        return self._b4msa_kwargs
    
    @b4msa_kwargs.setter
    def b4msa_kwargs(self, value):
        self._b4msa_kwargs = value

    @property
    def mixer_func(self):
        """The function is used to fix the output of the text's representations."""
        return self._mixer_func
    
    @mixer_func.setter
    def mixer_func(self, value):
        self._mixer_func = value

    @property
    def n_jobs(self):
        """Number of jobs used in multiprocessing."""
        return self._n_jobs
    
    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    def __sklearn_clone__(self):
        klass = self.__class__
        params = self.get_params()
        return klass(**params)

class DenseBoW(BoW):
    """
    DenseBoW is a text classifier in fact it is 
    a subclass of :py:class:`BoW` being the difference the process
    to represent the text in a vector space. This process is described in
    "`EvoMSA: A Multilingual Evolutionary Approach
    for Sentiment Analysis <https://ieeexplore.ieee.org/document/8956106>`_,
    Mario Graff, Sabino Miranda-Jimenez, Eric Sadit Tellez, Daniela Moctezuma. 
    Computational Intelligence Magazine, vol 15 no. 1, pp. 76-88, Feb. 2020."
    Particularly, in the section where the Emoji Space is described.

    :param emoji: Whether to use emoji text representations. default=True.
    :type emoji: bool
    :param dataset: Whether to use labeled dataset text representations (only available in 'ar', 'en', 'es', and 'zh'). default=True
    :type dataset: bool
    :param keyword: Whether to use keyword text representations. default=True.
    :type keyword: bool
    :param skip_dataset: Set of discarded dataset.
    :type skip_dataset: set
    :param unit_vector: Normalize vectors to have length 1. default=True
    :type unit_vector: bool 

    >>> from EvoMSA import DenseBoW
    >>> from microtc.utils import tweet_iterator
    >>> from EvoMSA.tests.test_base import TWEETS
    >>> D = list(tweet_iterator(TWEETS))
    >>> dense =  DenseBoW(lang='es')
    >>> dense.fit(D)
    >>> dense.predict(['Buenos dias']).tolist()
    ['P']    
    """
    def __init__(self, 
                 emoji: bool=True,
                 dataset: bool=True,
                 keyword: bool=True,
                 skip_dataset: Set[str]=set(),
                 estimator_kwargs=dict(dual=False),
                 unit_vector=True,
                 **kwargs) -> None:
        super(DenseBoW, self).__init__(estimator_kwargs=estimator_kwargs, **kwargs)
        self.skip_dataset = skip_dataset
        self._names = []
        self._text_representations = []
        self.unit_vector = unit_vector
        self.emoji = emoji
        self.dataset = dataset
        self.keyword = keyword

    def fit(self, *args, **kwargs) -> 'DenseBoW':
        """Estimate the parameters of the classifier or regressor 
        (:py:attr:`DenseBoW.estimator_class` - the fitted instance is accesible 
        at :py:attr:`DenseBoW.estimator_instance`) using the dataset (`D`, `y`).

        >>> from EvoMSA import DenseBoW
        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> D = list(tweet_iterator(TWEETS))
        >>> dense =  DenseBoW(lang='es').fit(D)        
        """
        return super(DenseBoW, self).fit(*args, **kwargs)

    def transform(self, D: List[Union[List, dict]], y=None) -> np.ndarray:
        """Represent the texts in `D` in the vector space.

        :param D: Texts to be transformed. In the case, it is a list of dictionaries the text is on the key :py:attr:`BoW.key`
        :type D: List of texts or dictionaries. 

        >>> from EvoMSA import DenseBoW
        >>> X = DenseBoW(lang='en').transform([dict(text='Hi'),
                                               dict(text='Good morning')])
        >>> X.shape
        (2, 1287)
        """                
        if isinstance(self.key, str):
            X = super(DenseBoW, self).transform(D, y=y)
            models = Parallel(n_jobs=self.n_jobs)(delayed(m.decision_function)(X)
                                                  for m in self.text_representations)
            _ = np.array(models).T
            if self.unit_vector:
                return _ / np.atleast_2d(np.linalg.norm(_, axis=1)).T
            else:
                return _
        assert len(D) and isinstance(D[0], dict)
        Xs = [super(DenseBoW, self).transform([x[key] for x in D], y=y)
              for key in self.key]
        with Parallel(n_jobs=self.n_jobs) as parallel:
            models = []
            for X in Xs:
                _ = parallel(delayed(m.decision_function)(X)
                             for m in self.text_representations)
                models.append(np.array(_).T)
        _ = self.mixer_func(models)
        if self.unit_vector:
            return _ / np.atleast_2d(np.linalg.norm(_, axis=1)).T
        else:
            return _            

    def predict(self, *args, **kwargs) -> np.ndarray:
        """Predict the response variable on the dataset `D`.

        >>> from EvoMSA import DenseBoW
        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> D = list(tweet_iterator(TWEETS))
        >>> dense = DenseBoW(lang='es').fit(D)
        >>> dense.predict(['Buenos dias']).tolist()
        ['P']                
        """
        return super(DenseBoW, self).predict(*args, **kwargs)

    @property
    def text_representations(self):
        """Classifiers that define the text representation."""
        return self._text_representations

    @text_representations.setter
    def text_representations(self, value):
        self._text_representations = value        

    def select(self, subset: Union[list, None]=None,
               D: List[Union[dict, list, None]]=None, 
               y: Union[np.ndarray, None]=None,
               feature_selection: Callable=KruskalFS,
               feature_selection_kwargs: dict=dict()) -> 'DenseBoW':
        """Procedure to perform feature selection or indices of the features to be selected.

        :param subset: Representations to be selected.
        :type subset: List of indices.
        :param D: Texts; in the case, it is a list of dictionaries the text is on the key :py:attr:`BoW.key`. default=None
        :type D: List of texts or dictionaries. 
        :param y: Response variable. The response variable can also be in `D` on the key :py:attr:`BoW.label_key`. default=None
        :type y: Array or None

        >>> from EvoMSA import DenseBoW
        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> T = list(tweet_iterator(TWEETS))
        >>> text_repr =  DenseBoW(lang='es').select(D=T)
        >>> text_repr.weights.shape
        (2672, 131072)        
        """
        assert subset is not None or D is not None
        if subset is not None:
            if len(subset) == 0:
                return self
            tr = self.text_representations
            self.text_representations = [tr[i] for i in subset]
            names = self.names
            self.names = [names[i] for i in subset]
            return self
        y = self.dependent_variable(D, y=y)
        X = self.transform(D)
        feature_selection = feature_selection(**feature_selection_kwargs).fit(X, y=y)
        index = feature_selection.get_support(indices=True)
        return self.select(subset=index)        

    def text_representations_extend(self, value: Union[List, str]):
        """Add dense BoW representations.

        :param value: List of models or name
        :type value: List of models or string
        """
        from EvoMSA.utils import load_url
        if isinstance(value, str):
            value = load_url(value, n_jobs=self.n_jobs)
        names = set(self.names)
        for x in value:
            label = x.labels[-1]
            if label not in names:
                self.text_representations.append(x)
                self.names.append(label)
                names.add(label)    

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        self._names = value            

    @property
    def weights(self):
        """Weights of the vector space. 
        It is matrix, i.e., :math:`\mathbf W \in \mathbb R^{M \\times d}`, where 
        :math:`M` is the dimension of the vector space (see :py:attr:`DenseBoW.names`)
        and :math:`d` is the vocabulary size.

        >>> from EvoMSA import DenseBoW
        >>> text_repr = DenseBoW(lang='es')
        >>> text_repr.weights.shape
        (2672, 131072)        
        """
        try:
            return self._weights
        except AttributeError:
            w = np.array([x._coef for x in self.text_representations])
            self._weights = w
            return self._weights

    @property
    def bias(self):
        """Bias."""
        try:
            return self._bias
        except AttributeError:
            w = np.array([x._intercept for x in self.text_representations])
            self._bias = w
            return self._bias

    @property
    def dataset(self):
        """Dense Representation based on human-annotated datasets"""
        return self._dataset
    
    @dataset.setter
    def dataset(self, value):
        self._dataset = value
        if value:
            self.load_dataset()

    @property
    def emoji(self):
        """Dense Representation based on emojis"""
        return self._emoji
    
    @emoji.setter
    def emoji(self, value):
        self._emoji = value
        if value:
            self.load_emoji()

    @property
    def keyword(self):
        """Dense Representation based on keywords"""
        return self._keyword
    
    @keyword.setter
    def keyword(self, value):
        self._keyword = value
        if value:
            self.load_keyword()
    
    @property
    def unit_vector(self):
        """Normalize representation to have one length"""
        return self._unit_vector
    
    @unit_vector.setter
    def unit_vector(self, value):
        self._unit_vector = value

    def fromjson(self, filename:str) -> 'DenseBoW':
        """Load the text representations from a json file.
        
        :param filename: Path
        :type filename: str
        """        
        models = [Linear(**kwargs)
                  for kwargs in tweet_iterator(filename)]
        self.text_representations_extend(models)
        return self
    
    def get_params(self, deep=True):
        """Obtain the parameters of the class"""
        dense_params = self._get_param_names()
        bow_params = BoW._get_param_names()
        params = dict()
        for k in bow_params + dense_params:
            params[k] = getattr(self, k)
        return params

    @property
    def skip_dataset(self):
        """Datasets discarded from the text representations"""
        return self._skip_dataset
    
    @skip_dataset.setter
    def skip_dataset(self, value):
        self._skip_dataset = value

    def load_emoji(self) -> None:
        if self.v1:
            emojis = load_emoji(lang=self.lang, v1=self.v1,
                                n_jobs=self.n_jobs)
            self.text_representations.extend(emojis)
            self.names.extend([x.labels[-1] for x in emojis])
        else:
            data = load_emoji(lang=self.lang,
                              d=self.voc_size_exponent, 
                              func=self.voc_selection,
                              n_jobs=self.n_jobs)
            self.text_representations.extend(data)
            self.names.extend([x.labels[-1] for x in data])            

    def load_keyword(self) -> None:
        if self.v1:
            _ = load_keyword(lang=self.lang, v1=self.v1,
                             n_jobs=self.n_jobs)
            self.text_representations.extend(_)
            self.names.extend([x.labels[-1] for x in _])
        else:       
            data = load_keyword(lang=self.lang,
                                d=self.voc_size_exponent, 
                                func=self.voc_selection,
                                n_jobs=self.n_jobs)
            self.text_representations.extend(data)
            self.names.extend([x.labels[-1] for x in data])            

    def load_dataset(self) -> None:
        if self.lang not in ['ar', 'zh', 'en', 'es']:
            return
        if self.v1:
            names = [name for name in dataset_information(lang=self.lang)
                    if name not in self._skip_dataset]
            _ = Parallel(n_jobs=self.n_jobs)(delayed(load_dataset)(lang=self.lang, name=name, v1=self.v1)
                                            for name in names)
            [self.text_representations.extend(k) for k in _]
            [self.names.extend([name] if len(k) == 1 else [f'{name}({i.labels[-1]})' for i in k])
            for k, name in zip(_, names)]
        else:
            data = load_dataset(lang=self.lang, name='datasets',
                                d=self.voc_size_exponent, 
                                func=self.voc_selection)
            _ = [x for x in data if x.labels[-1] not in self.skip_dataset]
            self.text_representations.extend(_)
            self.names.extend([x.labels[-1] for x in _])

    def __sklearn_clone__(self):
        klass = self.__class__
        params = self.get_params()
        models = ['emoji', 'keyword', 'dataset']
        args = {k: params[k] for k in models}
        params.update({k: False for k in models})
        ins = klass(**params)
        ins.text_representations_extend(self.text_representations)
        for k, v in args.items():
            setattr(ins, f'_{k}', v)
        return ins


TextRepresentations = DenseBoW

class StackGeneralization(BoW):
    """The idea behind stack generalization is to train an estimator on the predictions made by the base classifiers or regressors.

    :param decision_function_models: Represent the text by calling the decision function
    :type decision_function_models: List of :py:class:`BoW` or :py:class:`DenseBoW`
    :param transform_models: Represent the text by calling the transform
    :type transform_models: List of :py:class:`BoW` or :py:class:`DenseBoW`

    >>> from EvoMSA import DenseBoW, BoW, StackGeneralization
    >>> from microtc.utils import tweet_iterator
    >>> from EvoMSA.tests.test_base import TWEETS    
    >>> emoji =  DenseBoW(lang='es', dataset=False, keyword=False)
    >>> dataset = DenseBoW(lang='es', emoji=False, keyword=False)
    >>> bow = BoW(lang='es')
    >>> stacking = StackGeneralization(decision_function_models=[bow],
                                       transform_models=[dataset, emoji])
    >>> stacking.fit(list(tweet_iterator(TWEETS)))
    >>> stacking.predict(['Buenos dias']).tolist()
    ['P']
    """
    def __init__(self, decision_function_models: list=[], 
                 transform_models: list=[],
                 decision_function_name: str='predict_proba',
                 estimator_class=GaussianNB,
                 estimator_kwargs=dict(),
                 n_jobs: int=1,
                 **kwargs) -> None:
        assert len(decision_function_models) or len(transform_models)
        assert n_jobs == 1
        super(StackGeneralization, self).__init__(n_jobs=n_jobs,
                                                  decision_function_name=decision_function_name,
                                                  estimator_class=estimator_class,
                                                  estimator_kwargs=estimator_kwargs,
                                                  **kwargs)
        self._decision_function_models = decision_function_models
        self._transform_models = transform_models
        self.estimated = False

    def fit(self, *args, **kwargs) -> 'StackGeneralization':
        """
        >>> from EvoMSA import DenseBoW, BoW, StackGeneralization
        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS    
        >>> D = list(tweet_iterator(TWEETS))
        >>> emoji =  DenseBoW(lang='es', dataset=False, keyword=False)
        >>> dataset = DenseBoW(lang='es', emoji=False, keyword=False)
        >>> bow = BoW(lang='es')
        >>> stacking = StackGeneralization(decision_function_models=[bow],
                                           transform_models=[dataset, emoji]).fit(D)
        """        
        super(StackGeneralization, self).fit(*args, **kwargs)
        self._estimated = True
        return self

    def transform(self, D: List[Union[List, dict]], y=None) -> np.ndarray:
        """Represent the texts in `D` in the vector space.

        :param D: Texts to be transformed. In the case, it is a list of dictionaries the text is on the key :py:attr:`key`
        :type D: List of texts or dictionaries.

        >>> from EvoMSA import DenseBoW, BoW, StackGeneralization
        >>> from microtc.utils import tweet_iterator
        >>> from EvoMSA.tests.test_base import TWEETS
        >>> D = list(tweet_iterator(TWEETS))
        >>> emoji =  DenseBoW(lang='es', dataset=False, keyword=False)
        >>> dataset = DenseBoW(lang='es', emoji=False, keyword=False)
        >>> bow = BoW(lang='es')
        >>> df_models = [dataset, emoji, bow]
        >>> stacking = StackGeneralization(decision_function_models=df_models).fit(D)
        >>> stacking.transform(['buenos días'])
        array([[-1.56701076, -0.95614898, -0.39118087, 0.45360793, -1.65985598,
                -1.08645745, -0.67770805, 0.9703371, -1.40547817, -1.01340492,
                -0.57912169, 0.90450232]])
        """

        Xs = [text_repr.transform(D)
              for text_repr in self._transform_models]
        if not self.estimated:
            [text_repr.fit(D, y=y)
             for text_repr in self._decision_function_models]
            Xs += [text_repr.train_predict_decision_function(D, y=y)
                   for text_repr in self._decision_function_models]
            return np.concatenate(Xs, axis=1)
        Xs += [text_repr.decision_function(D)
               for text_repr in self._decision_function_models]
        return np.concatenate(Xs, axis=1)
    
    @property
    def estimated(self):
        return self._estimated
    
    @estimated.setter
    def estimated(self, value):
        self._estimated = value

    def train_predict_decision_function(self, *args, **kwargs) -> np.ndarray:
        assert not self.estimated
        return super(StackGeneralization, self).train_predict_decision_function(*args, **kwargs)
    
    @property
    def decision_function_models(self):
        """These models create the vector space by calling the decision function."""
        return self._decision_function_models

    @property
    def transform_models(self):
        """These models create the vector space by calling the transform."""
        return self._transform_models
    
    def __sklearn_clone__(self):
        klass = self.__class__
        params = self.get_params()
        params['decision_function_models'] = [clone(x) 
                                              for x in self.decision_function_models]
        params['transform_models'] = [clone(x)
                                      for x in self.transform_models]
        return klass(**params)