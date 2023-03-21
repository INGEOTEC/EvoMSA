.. _BoW:

:py:class:`BoW`
====================================
.. image:: https://github.com/INGEOTEC/EvoMSA/actions/workflows/test.yaml/badge.svg
		:target: https://github.com/INGEOTEC/EvoMSA/actions/workflows/test.yaml

.. image:: https://coveralls.io/repos/github/INGEOTEC/EvoMSA/badge.svg?branch=develop
		:target: https://coveralls.io/github/INGEOTEC/EvoMSA?branch=develop

.. image:: https://badge.fury.io/py/EvoMSA.svg
		:target: https://badge.fury.io/py/EvoMSA

.. image:: https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/evomsa-feedstock?branchName=main
	    :target: https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=16466&branchName=main

.. image:: https://img.shields.io/conda/vn/conda-forge/evomsa.svg
		:target: https://anaconda.org/conda-forge/evomsa

.. image:: https://img.shields.io/conda/pn/conda-forge/evomsa.svg
		:target: https://anaconda.org/conda-forge/evomsa

.. image:: https://readthedocs.org/projects/evomsa/badge/?version=docs
		:target: https://evomsa.readthedocs.io/en/docs/?badge=docs

.. image:: https://colab.research.google.com/assets/colab-badge.svg
		:target: https://colab.research.google.com/github/INGEOTEC/EvoMSA/blob/master/docs/Quickstart.ipynb	   

:py:class:`BoW` is a text classifier :math:`c` with signature 
:math:`c \equiv g \circ m`, where :math:`m` stands for the 
bag-or-words representation and :math:`g` is the classifier
(the default is a linear Support Vector Machine).

The classifier :math:`g` is trained on a dataset :math:`\mathcal D`
of pairs (:math:`x`, :math:`y`), where :math:`x` is a text and
:math:`y` is the label associated with it. The bag-of-words
representation :math:`m` is a pre-trained model with 
term frequency inverse document frequency (TFIDF) as the weighting scheme;
the parameters of pre-trained :math:`m` were estimated from 4,194,304 (:math:`2^{22}`)
tweets randomly selected.

The bag-of-words representation used is described in
"`A Simple Approach to Multilingual Polarity Classification in Twitter <https://www.sciencedirect.com/science/article/abs/pii/S0167865517301721>`_. 
Eric S. Tellez, Sabino Miranda-Jiménez, Mario Graff, 
Daniela Moctezuma, Ranyart R. Suárez, Oscar S. Siordia. 
Pattern Recognition Letters" and
"`An Automated Text Categorization Framework based 
on Hyperparameter Optimization <https://www.sciencedirect.com/science/article/abs/pii/S0950705118301217>`_. Eric S. Tellez, Daniela Moctezuma, 
Sabino Miranda-Jímenez, Mario Graff. 
Knowledge-Based Systems Volume 149, 1 June 2018."

Bag-of-Words Representation
--------------------------------

The core idea of a bag of words is that after the text is normalized and tokenized, each 
token :math:`t` is associated with a vector :math:`\mathbf{v_t} \in \mathbb R^d` 
where the :math:`i`-th component, i.e., :math:`\mathbf{v_t}_i`, contains the IDF value of 
the token :math:`t` and :math:`\forall_{j \neq i} \mathbf{v_t}_j=0`. 
The set of vectors :math:`\mathbf v` corresponds to the vocabulary,
there are :math:`d` different tokens in the vocabulary, and by definition
:math:`\forall_{i \neq j} \mathbf{v_i} \cdot \mathbf{v_j} = 0`, where
:math:`\mathbf{v_i} \in \mathbb R^d`, :math:`\mathbf{v_j} \in \mathbb R^d`,
and :math:`(\cdot)` is the dot product. It is worth mentioning that any 
token outside the vocabulary is discarded. The vocabulary size of the 
pre-trained bag-of-words representations is :math:`d=2^{14}`,
i.e., there are 16,384 different tokens for each language.

Using this notation, a text :math:`x` is represented by the sequence
of its tokens, i.e., :math:`(t_1, t_2, \ldots)`; the sequence 
can have repeated tokens, e.g., :math:`t_j = t_k`. 
Then each token is associated with its 
respective vector :math:`\mathbf v` (keeping the repetitions),
i.e., :math:`(\mathbf{v_{t_1}}, \mathbf{v_{t_2}}, \ldots)`.
Finally, the text :math:`x` is represented as: 

.. math:: 
	\mathbf x = \frac{\sum_t \mathbf{v_t}}{\mid\mid \sum_t \mathbf{v_t} \mid\mid},

where the sum goes for all the elements of the sequence,
:math:`\mathbf x \in \mathbb R^d`, and :math:`\mid\mid \mathbf w \mid\mid` 
is the Euclidean norm of vector :math:`\mathbf w`. 
The term frequency is implicitly computed in the sum 
because the process allows token repetitions.

This process is illustrated by representing the text *good morning*
in the pre-trained bag-of-words model. The first step is to import 
and initialize the class :py:class:`BoW` as done in the following 
instructions.

>>> from EvoMSA import BoW
>>> bow = BoW(lang='en')

The method :py:attr:`BoW.transform` receives a list of text to be represented
in the vector space of the bag of words. It returns a sparse matrix where 
the number of rows corresponds to the number of texts transformed and the columns
are the vocabulary. The following instruction processes the text *good morning*
and stored it in a variable :py:attr:`X`.

>>> X = bow.transform(['good morning'])
>>> X
<1x131072 sparse matrix of type '<class 'numpy.float64'>'
	with 39 stored elements in Compressed Sparse Row format>

The non-zero components are found in :py:attr:`X.indices`

>>> X.indices
array([   7,    9,   21,   24,   31,   34,   37,   40,   44,   61,  150,
        178,  202,  217,  229,  277,  309,  528,  573,  601,  663,  717,
        776,  780,  789,  825,  890,  937,  976, 1099, 1908, 2125, 2624,
       3225, 3315, 3677, 4177, 6558, 9443], dtype=int32)

However, one might wonder which token corresponds to each component; 
this information is in :py:attr:`BoW.names`. For example, 
the tokens associated with components 1099 and 4177 are: *good*
and *morning*, as can be seen below. 

>>> bow.names[1099], bow.names[4177]
('good', 'morning')

The IDF values associate to each token are in the dictionary 
:py:attr:`BoW.bow.token_weight`, e.g., the IDF value of text 
*morning* is 

>>> bow.bow.token_weight[4177]
7.079042426281991

Nonetheless, the value that the component 4177 has in the variable :py:attr:`X` is
0.2523 because the vector that represents *good morning* has been
normalized to have a unit length. 

Text Classifier
--------------------------------

Once the texts are in a vector space, then any classifier that works 
with vectors can be used; the one used by default is a 
linear Support Vector Machine (SVM).

To illustrate the process of creating a text classifier with :py:class:`BoW`, the 
following dataset will be used. 

>>> from EvoMSA import base
>>> from microtc.utils import tweet_iterator
>>> from os.path import join, dirname
>>> tweets = join(dirname(base.__file__), 'tests', 'tweets.json')
>>> D = list(tweet_iterator(tweets))

The dataset stored in :py:attr:`D` is a toy sentiment analysis dataset,
in Spanish, with four labels, positive, negative, neutral, and none. 
It is a list of dictionaries where the dictionary has two keys *text* 
and *klass*; the former has the text and the latter the label. 

The text classifier is trained with the following instruction. 

>>> bow = BoW(lang='es').fit(D)

where the language (:py:attr:`lang`) is set to Spanish (es), and 
:py:attr:`fit` receives the labeled dataset. 

The method :py:attr:`BoW.predict` is used to predict the label 
of a list of texts. For example, the label of the text *buenos días* (*good morning*)
is computed as:

>>> bow.predict(['buenos días'])
array(['P'], dtype='<U4')

where the label 'P' corresponds to the positive class. 

There are scenarios where it is more important to estimate the value(s) 
used to classify a particular instance; in the case of SVM, 
this is known as the decision function, and in the case of a 
Naive Bayes classifier, this is the probability of each class. 
This information can be found in :py:attr:`BoW.decision_function`
as can be seen in the following code.

>>> bow.decision_function(['buenos días'])
array([[-1.4054791 , -1.0134042 , -0.57912116,  0.90450178]])

API
--------------------------------

.. toctree::
   :maxdepth: 3

   bow_api
