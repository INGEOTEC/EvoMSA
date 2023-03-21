.. _TextRepresentations:

:py:class:`TextRepresentations`
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


:py:class:`TextRepresentations` is a text classifier in fact it is 
a subclass of :py:class:`BoW` being the difference the process
to represent the text in a vector space. This process is described in
"`EvoMSA: A Multilingual Evolutionary Approach
for Sentiment Analysis <https://ieeexplore.ieee.org/document/8956106>`_,
Mario Graff, Sabino Miranda-Jimenez, Eric Sadit Tellez, Daniela Moctezuma. 
Computational Intelligence Magazine, vol 15 no. 1, pp. 76-88, Feb. 2020."
Particularly, in the section where the Emoji Space is described.

Text Representation
--------------------------------

The idea is to represent a text in a vector space where the components 
have a more complex meaning than the bag-of-words model. 
In a bag of words, the meaning of each component corresponds to the 
token associated with it, and the IDF value gives it its importance. 

This complex behavior comes from associating each component to the
decision value of a text classifier trained on a labeled dataset which
is different from the one solving, albeit nothing forbids to be related to it. 
The datasets, from which these decision functions come, can be built either 
using a self-supervised approach or by annotating texts.

Without loss of generality, it is assumed that there are :math:`M` 
labeled datasets each one contains a binary text classification problem;  
noting that that if a dataset has :math:`K` labels, then this dataset 
can be represented as :math:`K` binary classification problems following 
the one versus the rest approach, i.e., it is transformed to :math:`K` datasets.

For each of these :math:`M` binary text classification 
problems a :py:class:`BoW` model is built 
using the default parameters (a pre-trained bag-of-words 
representation and a linear SVM as the classifier). 
That is, there are :math:`M` binary text classifiers, i.e., 
:math:`(c_1, c_2, \ldots, c_M)`. Additionally, the decision function of 
:math:`c_i` is a number where the sign indicates the class. 
Consequently, the text representation is the vector 
obtained by concatenating the decision functions 
of the :math:`M` classifiers and then normalizing the vector 
to have length 1. 

That is, a text :math:`x` is first represented 
with vector :math:`\mathbf{x^{'}} \in \mathbb R^M` where the 
value :math:`\mathbf{x^{'}}_i` corresponds to 
the decision function of :math:`c_i`. Given that the classifier 
:math:`c_i` is a linear SVM, the decision function corresponds
to the dot product between the input vector and  
the weight vector :math:`\mathbf w_i` plus the bias :math:`\mathbf w_{i_0}`,
where the weight vector and the bias are the parameters of the 
classifier. That is, 
the value :math:`\mathbf{x^{'}}_i` corresponds to

.. math:: 
	\mathbf{x^{'}}_i = \mathbf w_i \cdot \frac{\sum_t \mathbf{v_t}}{\mid\mid \sum_k \mathbf{v_k} \mid\mid} + \mathbf w_{i_0},

where :math:`\mathbf{v_t}` is the vector associated to the 
token :math:`t` of the text :math:`x`. In matrix notation, vector :math:`\mathbf{x'}` is

.. math:: 
	\mathbf{x^{'}} = \mathbf W \cdot \frac{\sum_t \mathbf{v_t}}{\mid\mid \sum_k \mathbf{v_k} \mid\mid} + \mathbf{w_0},

where matrix :math:`\mathbf W \in \mathbb R^{M \times d}` contains the weights, 
and :math:`\mathbf{w_0} \in \mathbb R^M` is the bias. Another way to see 
the previous formulation is by defining a 
vector :math:`\mathbf{u_t} = \frac{1}{\mid\mid \sum_k \mathbf{v_k} \mid\mid} \mathbf W \mathbf{v_t}` 
Consequently, vector :math:`\mathbf{x'}` is defined as 

.. math:: 
	\mathbf{x'} = \sum_t \mathbf{u_t} + \mathbf{w_0}.

It can be observed that each token is associated with a vector in the space defined by the 
classifiers and that :math:`\mathbf{x'}` is the sum of all vectors plus the bias vector (i.e., :math:`\mathbf{w_0}`). 
Finally, the vector representing the text :math:`x` is the normalized :math:`\mathbf{x^{'}}`, i.e.,

.. math::
	\mathbf x = \frac{\mathbf{x^{'}}}{\mid\mid \mathbf{x^{'}} \mid\mid}.


EvoMSA 2.0 has three sets of text representations, namely :py:attr:`dataset`,
:py:attr:`emoji`, and :py:attr:`keyword`. The dataset text representations 
were built on datasets coming from text-categorization competitions, 
the emoji representations are self-supervised datasets where
each task is to predict the presence of the emoji; and the keyword 
representations are also self-supervised datasets where the task is to 
predict the presence of a word; the words selected correspond to the
words of the vocabulary in the pre-trained bag-of-words model. 

To illustrate the usage of these representations, the text
*I love this song* is represented on the emoji space. The first step
is to initialize the model which can be done with the following
instructions.

>>> from EvoMSA import TextRepresentations
>>> emoji = TextRepresentations(lang='en', emoji=True, keyword=False, dataset=False)

The method :py:attr:`TextRepresentations.transform` receives a list
of text to be represented on this vector space, the following code
stores the output matrix in the variable :py:attr:`X`.

>>> X = emoji.transform(['I love this song'])

Equivalent, the attribute :py:attr:`TextRepresentations.names`
contains the description of each component, for example
the following code shows the value for the component
with index 9 and its description.

>>> X[:, 59], emoji.names[59]
(array([0.05337313]), '🎶')

The value 0.05 indicates that the emoji would be present
in the sentence *I love this song.*

API
--------------------------------

.. toctree::
   :maxdepth: 3

   text_repr_api
