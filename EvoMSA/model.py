# Copyright 2018 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from b4msa.textmodel import TextModel
from microtc.utils import tweet_iterator
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC
from ConceptModelling.thumbs_up_down import ThumbsUpDown, _ARABIC, _ENGLISH, _SPANISH, PATH as ConPATH
import os
from microtc.utils import save_model
from sklearn.neighbors import KDTree


class BaseTextModel(object):
    """Base class for text model

    :param corpus: Text to build the text model
    :type corpus: list or dict
    """

    def __init__(self, corpus=None, **kwargs):
        pass

    def fit(self, X):
        """
        Train the model

        :param X: Corpus
        :type X: list
        :rtype: instance
        """

        pass

    def __getitem__(self, x):
        pass

    def tokenize(self, text):
        pass

    def transform(self, X):
        return np.array([self.__getitem__(x) for x in X])


class BaseClassifier(object):
    """Base class for the classifier"""

    def __init__(self, random_state=0):
        pass

    def fit(self, X, y):

        """Method to train the classifier
        :param X: Independent variable
        :type X: np.array or csc_matrix
        :param y: Dependent variable
        :type y: np.array
        :rtype: self
        """

        return self

    def decision_function(self, X):
        """Classifier's decision function

        :param X: Independent variable
        :type X: np.array or csc_matrix
        :rtype: np.array
        """

        pass


class OutputClassifier(object):
    """LinearSVC that outputs the training set and test set using the environment varible OUTPUT"""

    def __init__(self, random_state=0, output=None):
        self._output = os.getenv('OUTPUT', output)
        assert self._output is not None
        self._random_state = random_state

    def fit(self, X, y):
        self.m = LinearSVC(random_state=self._random_state).fit(X, y)
        try:
            X = np.array(X.todense())
        except AttributeError:
            pass
        with open('%s_train.csv' % self._output, 'w') as fpt:
            for x, _y in zip(X, y):
                fpt.write(",".join([str(_) for _ in x]))
                fpt.write(",%s\n" % str(_y))
        return self

    def decision_function(self, X):
        hy = self.m.decision_function(X)
        try:
            X = np.array(X.todense())
        except AttributeError:
            pass
        with open('%s_test.csv' % self._output, 'w') as fpt:
            for x in X:
                fpt.write(",".join([str(_) for _ in x]))
                fpt.write("\n")
        return hy


class EvoMSAWrapper(BaseTextModel):
    def __init__(self, evomsa=None):
        assert evomsa is not None
        evomsa.n_jobs = 1
        self._evomsa = evomsa

    def transform(self, X):
        return self._evomsa.predict_proba(X)


class Identity(BaseTextModel, BaseClassifier):
    """Identity function used as either text model or classifier or regressor"""

    def __getitem__(self, x):
        return x

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        try:
            return X.toarray()
        except AttributeError:
            return X

    def predict_proba(self, X):
        return self.decision_function(X)


class HA(BaseTextModel):
    """Wrapper of b4msa.textmodel.TextModel and LinearSVC"""
    def __init__(self, **kwargs):
        self._tm = TextModel(**kwargs)
        self._cl = LinearSVC()

    def fit(self, X, y):
        self._tm.fit(X)
        self._cl.fit(self._tm.transform(X), y)
        return self

    def transform(self, X):
        res = self._cl.decision_function(self._tm.transform(X))
        if res.ndim == 1:
            return np.atleast_2d(res).T
        return res

    @classmethod
    def create_space(cls, fname, output, **kwargs):
        """Create the model from a file of json

        :param fname: Path to the file containing the json
        :type fname: str
        :param output: Path to store the model
        :type output: str
        :param kwargs: Keywords pass to TextModel
        """

        X = [x for x in tweet_iterator(fname)]
        m = cls(**kwargs)
        m.fit(X, [x['klass'] for x in X])
        save_model(m, output)


class Projection(BaseTextModel):
    def __init__(self, docs=None, textModel=None, projection=None):
        assert docs is None
        self._textModel = textModel
        self._projection = projection

    def transform(self, X):
        return np.dot(self._textModel.transform(X), self._projection)

    def __getitem__(self, x):
        return np.dot(self._textModel[x], self._projection)


class LabeledDataSet(BaseTextModel, BaseClassifier):
    """Create a text classifier using b4msa.textmodel.TextModel and LinearSVC

    :param docs: do not use
    :type docs: None
    :param textModel: text model e.g., b4msa.textmodel.TextModel
    :param coef: coefficients obtained from LinearSVC
    :type coef: array
    :param intercept: bias obtained from LinearSVC
    :type intercept: array
    :param labels: list of labels or classes
    :type labels: list
    """
    def __init__(self, docs=None, textModel=None, coef=None, intercept=None, labels=None):
        assert docs is None
        self._textModel = textModel
        self._coef = coef
        self._intercept = intercept
        self._labels = labels
        self._text = os.getenv('TEXT', default='text')

    @property
    def textModel(self):
        from .cython_utils import TextModelPredict
        try:
            return self._textModelPredict
        except AttributeError:
            self._textModelPredict = TextModelPredict(self._textModel,
                                                      self._coef,
                                                      self._intercept)
        return self._textModelPredict

    def __getstate__(self):
        """Remove attributes before the pickle"""

        r = self.__dict__.copy()
        try:
            del r['_textModelPredict']
        except KeyError:
            pass
        return r

    def fit(self, X, y=None):
        return self

    def get_text(self, text):
        key = self._text
        if isinstance(text, (list, tuple)):
            return " | ".join([x[key] for x in text])
        elif isinstance(text, str):
            return text
        return text[key]

    def decision_function(self, X):
        return self.transform(X)

    def predict_proba(self, X):
        return self.decision_function(X)

    def transform(self, X):
        output = []
        self.textModel.transform([self.get_text(x) for x in X], output)
        return np.array(output)

    def __getitem__(self, x):
        return np.array(self.textModel[self.get_text(x)])

    @classmethod
    def _create_space(cls, fname, **kwargs):
        """Create the space from a file of json

        :param fname: Path to the file containing the json
        :type fname: str
        :param kwargs: Keywords pass to TextModel
        """
        import random
        from .utils import linearSVC_array
        from collections import Counter
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **kwargs):
                return x

        data = [x for x in tweet_iterator(fname)]
        random.shuffle(data)
        tm = TextModel(**kwargs).fit([x['text'] for x in data[:128000]])
        tm._num_terms = tm.model.num_terms
        # klass, nele = np.unique([x['klass'] for x in data], return_counts=True)
        _ = [(k, v) for k, v in Counter([x['klass'] for x in data]).items()]
        _.sort(key=lambda x: x[0])
        klass = [x[0] for x in _]
        nele = [x[1] for x in _]
        h = {v: k for k, v in enumerate(klass)}
        MODELS = []
        for ident, k in tqdm(enumerate(klass)):
            elepklass = [0 for __ in klass]
            cnt = nele[ident]
            cntpklass = int(cnt / (len(klass) - 1))
            D = [(x, 1) for x in data if x['klass'] == k]
            for x in data:
                if x['klass'] == k:
                    continue
                if elepklass[h[x['klass']]] > cntpklass:
                    continue
                elepklass[h[x['klass']]] = elepklass[h[x['klass']]] + 1
                D.append((x, -1))
            m = LinearSVC().fit(tm.tonp([tm[x[0]['text']] for x in D]), [x[1] for x in D])
            MODELS.append(m)
        coef, intercept = linearSVC_array(MODELS)
        return tm, coef, intercept, klass

    @classmethod
    def create_space(cls, fname, output=None, **kwargs):
        """Create the space from a file of json

        :param fname: Path to the file containing the json
        :type fname: str
        :param output: Path to store the model, it is cls.model_fname if None
        :type output: str
        :param kwargs: Keywords pass to TextModel
        """
        tm, coef, intercept, klass = cls._create_space(fname, **kwargs)
        if output is None:
            output = cls.model_fname()
        ins = cls(textModel=tm, coef=coef, intercept=intercept, labels=klass)
        save_model(ins, output)


class EmoSpace(LabeledDataSet):
    """Base class of Emoji Space. Use EmoSpaceAr or EmoSpaceEn or EmoSpaceEs instead of this class
    """

    def __init__(self, docs=None, model_cl=None, **kwargs):
        if model_cl is None:
            textModel, coef, intercept, labels = self.get_model()
        else:
            textModel, coef, intercept, labels = model_cl
        super(EmoSpace, self).__init__(textModel=textModel, coef=coef,
                                       intercept=intercept, labels=labels)

    def get_model(self):
        from .utils import get_model
        model_fname = self.model_fname()
        return get_model(model_fname)

    @classmethod
    def _create_space(cls, fname, emo_option='delete', **kwargs):
        """Create the space from a file of json

        :param fname: Path to the file containing the json
        :type fname: str
        :param kwargs: Keywords pass to TextModel
        """
        return super(EmoSpace, cls)._create_space(fname, emo_option=emo_option, **kwargs)

    @classmethod
    def create_space(cls, fname, output=None, **kwargs):
        """Create the space from a file of json

        :param fname: Path to the file containing the json
        :type fname: str
        :param output: Path to store the model, it is cls.model_fname if None
        :type output: str
        :param kwargs: Keywords pass to TextModel
        """
        tm, coef, intercept, klass = cls._create_space(fname, **kwargs)
        if output is None:
            output = cls.model_fname()
        save_model([tm, coef, intercept, klass], output)


class EmoSpaceEs(EmoSpace):
    """Spanish text model or classifier based on a Emoji Space
    """
    @staticmethod
    def model_fname():
        import EvoMSA
        return 'emo-v%s-es.evoemo' % EvoMSA.__version__[:3]

    @classmethod
    def create_space(cls, fname, output=None, lang='es', **kwargs):
        super(EmoSpaceEs, cls).create_space(fname, output=output, lang=lang, **kwargs)


class EmoSpaceEn(EmoSpace):
    """English text model or classifier based on a Emoji Space"""

    @staticmethod
    def model_fname():
        import EvoMSA
        return 'emo-v%s-en.evoemo' % EvoMSA.__version__[:3]

    @classmethod
    def create_space(cls, fname, output=None, lang='en', **kwargs):
        super(EmoSpaceEn, cls).create_space(fname, output=output, lang=lang, **kwargs)


class EmoSpaceAr(EmoSpace):
    """Arabic text model or classifier based on a Emoji Space"""

    @staticmethod
    def model_fname():
        import EvoMSA
        return 'emo-v%s-ar.evoemo' % EvoMSA.__version__[:3]

    @classmethod
    def create_space(cls, fname, output=None, lang='ar', **kwargs):
        super(EmoSpaceAr, cls).create_space(fname, output=output, lang=lang, **kwargs)


class Corpus(BaseTextModel):
    """Text model using only words"""

    def __init__(self, corpus=None, **kwargs):
        self._text = os.getenv('TEXT', default='text')
        self._m = {}
        self._num_terms = 0
        self._training = True
        self._textModel = TextModel([''], token_list=[-1])
        if corpus is not None:
            self.fit(corpus)

    def get_text(self, text):
        return text[self._text]

    def fit(self, c):
        [self.__getitem__(x) for x in c]
        self._training = False
        return self

    @property
    def num_terms(self):
        return self._num_terms

    def tokenize(self, text):
        if isinstance(text, dict):
            text = self.get_text(text)
        if isinstance(text, (list, tuple)):
            tokens = []
            for _text in text:
                tokens.extend(self._textModel.tokenize(_text))
            return tokens
        else:
            return self._textModel.tokenize(text)

    def transform(self, texts):
        """Convert test into a vector

        :param texts: List of text to be transformed
        :type text: list

        :rtype: list

        Example:

        >>> from microtc.textmodel import TextModel
        >>> corpus = ['buenos dias catedras', 'catedras conacyt']
        >>> textmodel = TextModel().fit(corpus)
        >>> X = textmodel.transform(corpus)
        """
        return self._textModel.tonp([self.__getitem__(x) for x in texts])

    def __getitem__(self, d):
        tokens = []
        for t in self.tokenize(d):
            try:
                index, k = self._m[t]
                if self._training:
                    self._m[t] = [index, k+1]
            except KeyError:
                if not self._training:
                    continue
                index, k = self._num_terms, 1
                self._m[t] = [index, k]
                self._num_terms += 1
            tokens.append([index, k])
        return tokens


class AggressivenessAr(Corpus):
    """Arabic text model using an aggressive corpus"""

    def __init__(self, *args, **kwargs):
        fname = os.path.join(os.path.dirname(__file__), 'conf',
                             'aggressiveness.ar')
        corpus = []
        for x in tweet_iterator(fname):
            corpus += x['words']
        super(AggressivenessAr, self).__init__(corpus)


class AggressivenessEn(Corpus):
    """English text model using an aggressive corpus"""

    def __init__(self, *args, **kwargs):
        fname = os.path.join(os.path.dirname(__file__), 'conf',
                             'aggressiveness.en')
        corpus = []
        for x in tweet_iterator(fname):
            corpus += x['words']
        super(AggressivenessEn, self).__init__(corpus)


class AggressivenessEs(Corpus):
    """Spanish text model using an aggressive corpus"""

    def __init__(self, *args, **kwargs):
        fname = os.path.join(os.path.dirname(__file__), 'conf',
                             'aggressiveness.es')
        corpus = []
        for x in tweet_iterator(fname):
            corpus += x['words']
        super(AggressivenessEs, self).__init__(corpus)


class Bernulli(BaseClassifier):
    """Bernulli classifier"""

    def __init__(self, random_state=0):
        self._num_terms = -1

    @property
    def num_terms(self):
        return self._num_terms

    def fit(self, X, klass):
        self._num_terms = X.shape[1]
        klasses = np.unique(klass)
        pr = np.zeros((klasses.shape[0], self.num_terms))
        for i, k in zip(X, klass):
            index = i.indices
            if index.shape[0] > 0:
                pr[k, index] += 1
        _ = np.atleast_2d(self.num_terms + np.array([(klass == _k).sum() for _k in klasses])).T
        pr = (1 + pr) / _
        self._wj = np.log(pr)
        self._vj = np.log(1 - pr)
        return self

    def predict(self, X):
        return np.argmax(self.decision_function_raw(X), axis=1)

    def decision_function(self, X):
        _ = self.predict_proba(X)
        df = _ * 2 - 1
        df[df > 1] = 1
        df[df < -1] = -1
        return df

    def predict_proba(self, X):
        X = self.decision_function_raw(X)
        pr = np.exp(X)
        den = pr.sum(axis=1)
        _ = pr / np.atleast_2d(den).T
        return _

    def decision_function_raw(self, X):
        wj = self._wj
        vj = self._vj
        hy = []
        for d in X:
            x = np.zeros(self.num_terms)
            index = d.indices
            if index.shape[0] > 0:
                x[index] = 1
            _ = ((x * wj) + (1 - x) * vj).sum(axis=1)
            hy.append(_)
        return np.array(hy)


class Multinomial(Bernulli):
    """Multinomial classifier"""

    def fit(self, X, klass):
        self._num_terms = X.shape[1]
        klasses = np.unique(klass)
        pr = np.zeros((klasses.shape[0], self.num_terms))
        for i, k in zip(X, klass):
            index = i.indices
            if index.shape[0] > 0:
                pr[k, index] += 1
        den = pr.sum(axis=1)
        self._log_xj = np.log((1 + pr) / np.atleast_2d(self.num_terms + den).T)
        return self

    def decision_function_raw(self, X):
        xj = self._log_xj
        hy = []
        for d in X:
            x = np.zeros(self.num_terms)
            index = d.indices
            if index.shape[0] > 0:
                x[index] = 1
            _ = (xj * x).sum(axis=1)
            hy.append(_)
        return np.array(hy)


class ThumbsUpDownEs(ThumbsUpDown, BaseTextModel):
    """Spanish thumbs up and down model"""

    def __init__(self, *args, **kwargs):
        super(ThumbsUpDownEs, self).__init__(lang=_SPANISH, stemming=False)

    def fit(self, X):
        return self


class ThumbsUpDownEn(ThumbsUpDown, BaseTextModel):
    """English thumbs up and down model"""

    def __init__(self, *args, **kwargs):
        super(ThumbsUpDownEn, self).__init__(lang=_ENGLISH, stemming=False)

    def fit(self, X):
        return self


class ThumbsUpDownAr(ThumbsUpDown, BaseTextModel):
    """Arabic thumbs up and down model"""

    def __init__(self, *args, **kwargs):
        super(ThumbsUpDownAr, self).__init__(lang=_ARABIC, stemming=False)

    def fit(self, X):
        return self


class Vec(BaseTextModel):
    """Read the key vec, useful to incorporate external knowledge as FastText print-sentence-vectors"""

    def __getitem__(self, x):
        return x['vec']

    def fit(self, X):
        return self


class SemanticToken(BaseTextModel):
    def __init__(self, corpus=None, threshold=0.001, token_min_filter=0.001,
                 token_list=[-2, -1],
                 num_option='delete', usr_option='delete',
                 url_option='delete', emo_option='delete', **kwargs):
        self._text = os.getenv('TEXT', default='text')
        self._textmodel = TextModel(None, token_list=token_list,
                                    threshold=threshold,
                                    token_min_filter=token_min_filter,
                                    num_option=num_option, usr_option=usr_option,
                                    url_option=url_option, emo_option=emo_option,
                                    **kwargs)
        self._threshold = threshold
        if corpus is not None:
            self.fit(corpus)

    def fit(self, X):
        self.init(X)
        return self

    @property
    def threshold(self):
        """Threshold used to remove those tokens that less than 1 - entropy

        :rtype: float
        """

        return self._threshold

    @property
    def textmodel(self):
        """TextModel instance"""

        return self._textmodel

    @property
    def id2token(self):
        """Map from id to token

        :rtype: list"""

        return self._id2token

    def init(self, corpus):
        """Initial model"""

        words = self.tokens(corpus)
        self._weight = np.ones(len(words))
        # key = self.semantic_space._text
        X = self.semantic_space.transform([str(x) for x in words])
        self._kdtree = KDTree(X, metric='manhattan')
        w = self.entropy(self.transform(corpus), corpus, ntokens=X.shape[0])
        w = np.where(w > self.threshold)[0]
        self._kdtree = KDTree(X[w], metric='manhattan')
        self._weight = self._weight[w]
        self._id2token = [words[x] for x in w]
        self.compute_idf(self.transform(corpus))

    def entropy(self, Xt, corpus, ntokens):
        """Compute the entropy"""

        y = [x['klass'] for x in corpus]
        klasses = np.unique(y)
        nklasses = klasses.shape[0]
        weight = np.zeros((klasses.shape[0], ntokens))
        for ki, klass in enumerate(klasses):
            for _y, tokens in zip(y, Xt):
                if _y != klass:
                    continue
                for x in np.unique([_[0] for _ in tokens]):
                    weight[ki, x] += 1
        weight = weight / weight.sum(axis=0)
        weight[~np.isfinite(weight)] = 1.0 / nklasses
        logc = np.log2(weight)
        logc[~np.isfinite(logc)] = 0
        if nklasses > 2:
            logc = logc / np.log2(nklasses)
        return (1 + (weight * logc).sum(axis=0))

    def compute_idf(self, Xt):
        N = len(Xt)
        weight = np.zeros_like(self.weight)
        for tokens in Xt:
            for x in np.unique([_[0] for _ in tokens]):
                weight[x] += 1
        self._weight = np.log2(N / weight)

    @property
    def weight(self):
        """weight per token

        :rtype: list"""

        return self._weight

    def tokens(self, corpus):
        """Tokens used for modeling"""
        self.textmodel.fit(corpus)
        return [x for x in self.textmodel.model._w2id.keys()]

    @property
    def semantic_space(self):
        """Semantic space

        :rtype: instance
        """

        try:
            return self._semantic_space
        except AttributeError:
            self._semantic_space = EmoSpaceEs()
        return self._semantic_space

    def get_text(self, text):
        """Return self._text key from text

        :param text: Text
        :type text: dict
        """

        return text[self._text]

    def tokenize(self, text):
        """Tokenize a text
        :param text: Text
        :type text: dict or str

        :rtype: list of str
        """

        textmodel = self.textmodel
        return textmodel.tokenize(text)

    def transform(self, X):
        from collections import Counter
        w = self.weight
        kdtree = self._kdtree
        tokens_doc = [self.tokenize(x) for x in X]
        tokens = []
        [tokens.__iadd__(x) for x in tokens_doc]
        tokens = [x for x in Counter(tokens).keys()]
        # key = self.semantic_space._text
        xx = self.semantic_space.transform(tokens)
        m = {t: (ind[0], w[ind[0]] / (1 + d[0])) for t, d, ind in zip(tokens, *kdtree.query(xx))}
        Xt = []
        for x in tokens_doc:
            unique = {}
            for _ in x:
                r = m[_]
                try:
                    unique[r[0]] += r[1]
                except KeyError:
                    unique[r[0]] = r[1]
            n = np.sqrt(sum([x * x for _, x in unique.items()]))
            Xt.append([(i, x/n) for i, x in unique.items()])
        return Xt


class SemanticTokenEs(SemanticToken):
    def __init__(self, corpus=None, stopwords='delete', **kwargs):
        super(SemanticTokenEs, self).__init__(corpus=corpus, stopwords=stopwords,
                                              lang='es', **kwargs)


class SemanticTokenEn(SemanticToken):
    def __init__(self, corpus=None, del_dup=False, stopwords='delete', **kwargs):
        super(SemanticTokenEn, self).__init__(corpus=corpus, del_dup=del_dup,
                                              stopwords=stopwords,
                                              lang='en', **kwargs)

    @property
    def semantic_space(self):
        """Semantic space

        :rtype: instance
        """

        try:
            return self._semantic_space
        except AttributeError:
            self._semantic_space = EmoSpaceEn()
        return self._semantic_space


class SemanticTokenAr(SemanticToken):
    def __init__(self, corpus=None, stopwords='delete', **kwargs):
        super(SemanticTokenAr, self).__init__(corpus=corpus, stopwords=stopwords,
                                              lang='ar', **kwargs)

    @property
    def semantic_space(self):
        """Semantic space

        :rtype: instance
        """

        try:
            return self._semantic_space
        except AttributeError:
            self._semantic_space = EmoSpaceAr()
        return self._semantic_space


class SemanticAffectiveEs(SemanticTokenEs):
    def tokens(self, corpus):
        """Tokens used for modeling"""
        fname = os.path.join(ConPATH, 'data', 'es.affective.words.json')
        lst = []
        for x in tweet_iterator(fname):
            lst += x['words']
        return lst


class SemanticAffectiveAr(SemanticTokenAr):
    def tokens(self, corpus):
        """Tokens used for modeling"""
        fname = os.path.join(ConPATH, 'data', 'ar.affective.words.json')
        lst = []
        for x in tweet_iterator(fname):
            lst += x['words']
        return lst


class SemanticAffectiveEn(SemanticTokenEn):
    def tokens(self, corpus):
        """Tokens used for modeling"""
        fname = os.path.join(ConPATH, 'data', 'en.affective.words.json')
        lst = []
        for x in tweet_iterator(fname):
            lst += x['words']
        return lst
