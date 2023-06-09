.. _v2:

====================================
EvoMSA 2.0
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
		:target: https://colab.research.google.com/github/INGEOTEC/EvoMSA/blob/docs/docs/EvoMSA2.ipynb	   



EvoMSA is a stack generalization algorithm specialized in text classification problems. Text classification is a Natural Language Processing task focused on identifying a text’s category. A standard approach to tackle text classification problems is to pose it as a supervised learning problem. In supervised learning, everything starts with a dataset composed of pairs of inputs and outputs; in this case, the inputs are texts, and the outputs correspond to the associated labels or categories. The aim is that the developed algorithm can automatically assign a label to any given text independently, whether it was in the original dataset. The feasible categories are only those found on the original dataset. In some circumstances, the method can also inform the confidence it has in its prediction so the user can decide whether to use or discard it.

The key idea of EvoMSA is to combine different text classifiers using a stack generalization approach. A text classifier :math:`c`, can be seen as a composition of two functions, i.e., :math:`c \equiv g \circ m`; where :math:`m` transforms the text into a vector space, i.e., :math:`m: \text{text} \rightarrow \mathbb R^d` and :math:`g` is the classifier (:math:`g: \mathbb R^d \rightarrow \mathbb N`) or regressor (:math:`g: \mathbb R^d \rightarrow \mathbb R`). 

EvoMSA focused on developing diverse text representations (:math:`m`), fixing the classifier :math:`g` as a linear Support Vector Machine. The text representations (functions :math:`m`) found in EvoMSA can be grouped into two. On the one hand, there is a traditional :ref:`Bag of Words (BoW) <bow>` representation; on the other, there is a dense representation, namely :ref:`dense BoW <densebow>`.

:ref:`v2` supports more languages than the previous version, currently it supports Arabic (ar), Catalan (ca), German (de), English (en), Spanish (es), French (fr), Hindi (hi), Indonesian (in), Italian (it), Japanese (ja), Korean (ko), Dutch (nl), Polish (pl), Portuguese (pt), Russian (ru), Tagalog (tl), Turkish (tr), and Chinese (zh).

:ref:`v2` simplifies the previous version (:ref:`EvoMSA <v1>`), and now it has only three main classes.

.. toctree::
   :maxdepth: 1

   bow
   text_repr
   stack

:ref:`BoW` and :ref:`DenseBoW` are text classifiers; :ref:`BoW` is the parent of :ref:`DenseBoW`. The stack generalization technique is implemented in :ref:`StackGeneralization`.

:ref:`v2` has been tested in many text classifier competitions without modifications. The aim is to offer a better understanding of how these algorithms perform in a new situation and what would be the difference in performance with an algorithm tailored to the new problem. In the following link, we will describe the specifics of each configuration.

.. toctree::
   :maxdepth: 1

   competition


Quickstart Guide
====================================

This section describes the usage of EvoMSA using a dummy text classification problem; mainly, it is a sentiment analysis dataset, in Spanish, with four labels: positive (P), negative (N), neutral (NEU), and none (NONE). The problem can be found in EvoMSA's tests. 

The guide includes the creation of three text classifiers, one using a BoW model, the other using a dense BoW, and the last classifier combines the previous two models using a stacking mechanism. 

The first step is to install EvoMSA, which is described below.

Installing EvoMSA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step is to install the library, which can be done using the `conda package manager <https://conda-forge.org/>`_ with the following instruction. 

.. code:: bash

	  conda install -c conda-forge EvoMSA

A more general approach to installing EvoMSA is through the use of the command pip, as illustrated in the following instruction. 

.. code:: bash

	  pip install EvoMSA


Libraries and Text Classification Problem    
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once EvoMSA is installed, one must load a few libraries. The first line loads EvoMSA core classes. Line 2 contains the pathname where the text classification problem is. Finally, line 3 is a method to read a file containing a JSON per line.  

.. code-block:: python

    >>> from EvoMSA import BoW, DenseBoW, StackGeneralization
    >>> from EvoMSA.tests.test_base import TWEETS
    >>> from microtc.utils import tweet_iterator
	

The text classification problem can be read using the following instruction. It is stored in a variable D which is a list of dictionaries. The second line shows the content of the first element in D.

.. code-block:: python

    >>> D = list(tweet_iterator(TWEETS))
    >>> D[0]
    {'text': '| R #OnPoint | #Summer16 ...... 🚣🏽🌴🌴🌴 @ Ocho Rios, Jamacia https://t.co/8xkfjhk52L',
     'klass': 'NONE',
     'q_voc_ratio': 0.4585635359116022}

The field :py:attr:`text` is self-described, and the field :py:attr:`klass`` contains the label associated with that text.


:ref:`bow` Classifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first text classifier presented is the pre-trained BoW. The following line initializes the classifier, the first part initializes the class, and the second corresponds to the estimate of the parameters of the linear SVM. 

.. code-block:: python

    >>> bow = BoW(lang='es').fit(D)

After training the text classifier, it can make predictions. For instance, the first line predicts the training set, while the second line predicts the phrase *good morning* in Spanish, *buenos días.*

.. code-block:: python

    >>> hy = bow.predict(D)
    >>> bow.predict(['buenos días'])	
    array(['P'], dtype='<U4')

It can be observed that the predicted class for *buenos días* is positive (P).


:ref:`densebow` Classifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, the second method is trained using the dataset following the same steps. The subsequent instruction shows the code to train the text classifier. 

.. code-block:: python

    >>> dense = DenseBoW(lang='es').fit(D)	

The code to predict is equivalent; therefore, the prediction for the phrase *good morning* is only shown.

    >>> dense.predict(['buenos días'])
    array(['P'], dtype='<U4')

:ref:`Stack Generalization <StackGeneralization>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The final text classifier uses a stack generalization approach. The first step is to create the base text classifiers corresponding to the two previous text classifiers, :ref:`bow`, and :ref:`densebow`.


.. code-block:: python

    >>> bow = BoW(lang='es')
    >>> dense = DenseBoW(lang='es')	

It is worth noting that the base classifiers were not trained; as can be seen, the method fit was not called. These base classifiers will be trained inside the stack generalization algorithm. 

The second step is to initialize and train the stack generalization class, shown in the following instruction. 

.. code-block:: python

    >>> stack = StackGeneralization([bow, dense]).fit(D)

One does not need to specify the language to stack generalization because the base text classifiers give the language. 

The code to predict is kept constant in all the classes; therefore, the following code predicts the class for the phrase *good morning.*

.. code-block:: python

    >>> stack.predict(['buenos días'])
    array(['P'], dtype='<U4')

Citing
==========

If you find EvoMSA useful for any academic/scientific purpose, we would appreciate citations to the following reference:
  
.. code:: bibtex

	  @article{DBLP:journals/corr/abs-1812-02307,
	  author = {Mario Graff and Sabino Miranda{-}Jim{\'{e}}nez
	                 and Eric Sadit Tellez and Daniela Moctezuma},
	  title     = {EvoMSA: {A} Multilingual Evolutionary Approach for Sentiment Analysis},
	  journal   = {Computational Intelligence Magazine},
	  volume    = {15},
	  issue     = {1},
	  year      = {2020},
	  pages     = {76 -- 88},
	  url       = {https://ieeexplore.ieee.org/document/8956106},
	  month     = {Feb.}
	  }

	


API
====================================

.. toctree::
   :maxdepth: 1

   bow_api
   text_repr_api
   stack_api

:ref:`EvoMSA first version <v1>`
====================================

The documentation of EvoMSA first version can be found in the following sections. 


.. toctree::
   :maxdepth: 2

   v1