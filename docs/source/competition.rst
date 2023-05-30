.. _competition:

Text Classifier Competitions
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


Text classification (TC) is a Natural Language Processing (NLP) task focused on identifying a text's label. A standard approach to tackle text classification problems is to pose it as a supervised learning problem. In supervised learning, everything starts with a dataset composed of pairs of inputs and outputs; in this case, the inputs are texts, and the outputs correspond to the associated labels. The aim is that the developed algorithm can automatically assign a label to any given text independently, whether it was in the original dataset. The feasible classes are only those found on the original dataset. In some circumstances, the method can also inform the confidence it has in its prediction so the user can decide whether to use or discard it.

Following a supervised learning approach requires that the input is in amenable representation for the learning algorithm; usually, this could be a vector. One of the most common methods to represent a text into a vector is to use a Bag of Word (:ref:`bow`) model, which works by having a fixed vocabulary where each component represents an element in the vocabulary and the presence of it in the text is given by a non-zero value.

The text classifier's performance depends on the representation quality and the classifier used. Deciding which representation and algorithm to use is daunting; in this contribution, we describe a set of classifiers that can be used, out of the box, for a new text classification problem. These classifiers are based on the :ref:`BoW` model. Nonetheless, some methods, namely :ref:`TextRepresentations`, represent the text following two stages. The first one uses a set of BoW models and classifiers trained on self-supervised problems, where each task predicts the presence of a particular token. Consequently, the text is presented in a vector where each component is associated with a token, and the existence of it is encoded in the value. The methods used BoW models, and TextRepresentations were combined using a stack generalization approach, namely :ref:`StackGeneralization`. 

The text classifiers presented have been tested in many text classifier competitions without modifications. The aim is to offer a better understanding of how these algorithms perform in a new situation and what would be the difference in performance with an algorithm tailored to the new problem. We test 13 different algorithms for each task of each competition. The configuration having the best performance was submitted to the contest. The best performance was computed using either a k-fold cross-validation or a validation set, depending on the information provided by the challenge.


Systems
=================================================

.. code-block:: python

	from EvoMSA import BoW, TextRepresentations, StackGeneralization
	from sklearn.model_selection import StratifiedKFold


:ref:`BoW` default parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	def bow(lang, tr, vs, **kwargs):
		bow = BoW(lang=lang).fit(tr)
		hy = bow.predict(vs)
		return hy


:ref:`BoW` using :py:attr:`voc_selection` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	def bow_voc_selection(lang, tr, vs, **kwargs):
		bow = BoW(lang=lang, voc_selection='most_common').fit(tr)
		hy = bow.predict(vs)
		return hy

:ref:`BoW` trained on the training set 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	def bow3(lang, tr, vs, **kwargs):
		params = b4msa_params(lang=lang)
		del params['token_max_filter']
		del params['max_dimension']
		bow_no_pre = BoW(lang=lang, pretrain=False, b4msa_kwargs=params).fit(tr)
		return bow_no_pre.predict(vs)


:ref:`StackGeneralization` with :ref:`BoW` and :ref:`TextRepresentations` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	def bow_keywords_emojis(lang, tr, vs, **kwargs):
		bow = BoW(lang=lang)
		keywords = TextRepresentations(lang=lang, 
                                       emoji=False, 
                                       dataset=False).select(D=tr)
		emojis = TextRepresentations(lang=lang, 
                                     keyword=False, 
                                     dataset=False).select(D=tr)
		stack = StackGeneralization(decision_function_models=[bow, keywords, emojis]).fit(tr)
		X = bow.transform(vs)
		for x in [bow, keywords, emojis]:
			x.cache = X    
		return stack.predict(vs)


:ref:`StackGeneralization` with :ref:`BoW` and :ref:`TextRepresentations` using :py:attr:`voc_selection` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	def bow_keywords_emojis_voc_selection(lang, tr, vs, **kwargs):
		bow = BoW(lang=lang, voc_selection='most_common')
		keywords = TextRepresentations(lang=lang, voc_selection='most_common',
                                       emoji=False, 
                                       dataset=False).select(D=tr)
		emojis = TextRepresentations(lang=lang, voc_selection='most_common',
                                     keyword=False, 
                                     dataset=False).select(D=tr)
		stack = StackGeneralization(decision_function_models=[bow, keywords, emojis]).fit(tr)
		X = bow.transform(vs)
		for x in [bow, keywords, emojis]:
			x.cache = X    
		return stack.predict(vs)		