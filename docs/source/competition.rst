.. _competition:

====================================
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

The text classifier's performance depends on the representation quality and the classifier used. Deciding which representation and algorithm to use is daunting; in this contribution, we describe a set of classifiers that can be used, out of the box, for a new text classification problem. These classifiers are based on the :ref:`BoW` model. Nonetheless, some methods, namely :ref:`DenseBoW`, represent the text following two stages. The first one uses a set of BoW models and classifiers trained on self-supervised problems, where each task predicts the presence of a particular token. Consequently, the text is presented in a vector where each component is associated with a token, and the existence of it is encoded in the value. The methods used BoW models, and DenseBoW were combined using a stack generalization approach, namely :ref:`StackGeneralization`. 

The text classifiers presented have been tested in many text classifier competitions without modifications. The aim is to offer a better understanding of how these algorithms perform in a new situation and what would be the difference in performance with an algorithm tailored to the new problem. We test 13 different algorithms for each task of each competition. The configuration having the best performance was submitted to the contest. The best performance was computed using either a k-fold cross-validation or a validation set, depending on the information provided by the challenge.

Results
------------------------------

.. list-table:: Competitions 
    :header-rows: 1

    * - Competitions 
      - Winner 
      - :ref:`v2` 
      - Difference
    * - HaSpeeDe
      -
      - 
      -
    * - HODI 
      - 0.81079 
      - 0.71527
      - 13.4%
    * - ACTI
      - 0.85712
      - 0.78207
      - 9.6%
    * - PoliticIT (Global)
      - 0.824057
      - 0.762001
      - 8.1%
    * - PoliticIT (Gender)
      - 0.824287
      - 0.732259
      - 12.6%
    * - PoliticIT (Ideology Binary)
      - 0.928223
      - 0.848525
      - 9.4%
    * - PoliticIT (Ideology Multiclass)
      - 0.751477
      - 0.705220 
      - 6.6%
    * - PoliticEs (Global)
      - 0.811319
      - 0.777584
      - 4.3%
    * - PoliticEs (Gender)
      - 0.829633
      - 0.711549
      - 16.6%
    * - PoliticEs (Profession)
      - 0.860824
      - 0.837945 
      - 2.7%
    * - PoliticEs (Ideology Binary)
      - 0.896715
      - 0.891394
      - 0.6%
    * - PoliticEs (Ideology Multiclass) 
      - 0.691334
      - 0.669448
      - 3.3%
    * - DAVINCIS 
      - 0.9264
      - 0.8903
      - 4.1%
    * - REST-MEX (Global)
      - 0.7790190145
      - 0.7375714730
      - 5.6%
    * - REST-MEX (Polarity)
      - 0.621691991
      - 0.554880778
      - 12.0%
    * - REST-MEX (Type)
      - 0.99032231
      - 0.980539122
      - 1.0%
    * - REST-MEX (Country)
      - 0.942028113
      - 0.927052594
      - 1.6%    
    * - HOMO-MEX
      - 0.8847
      - 0.8050
      - 9.9%
    * - HOPE (ES)
      - 0.9161
      - 0.4198
      - 118.2%
    * - HOPE (EN)
      - 0.5012
      - 0.4429
      - 13.2%
    * - DIPROMATS (ES)
      - 0.8089
      - 0.7485
      - 8.1%
    * - DIPROMATS (EN)
      - 0.8090
      - 0.7255
      - 11.5%
    * - HUHU
      - 0.820
      - 0.775
      - 5.8%


Systems
-----------------------------------------------

We test 13 different combinations of :ref:`BoW` and :ref:`DenseBoW` models. These models include the use of the two procedures to select the vocabulary (parameter voc_selection), the use of pre-trained :ref:`BoW`, and the creation of the :ref:`BoW` representation with the given training set. Additionally, we create text representations tailored to the problem at hand. That is the words with more discriminant power in a :ref:`BoW` classifier, trained on the training set, are selected as the labels in self-supervised problems. 

.. code-block:: python

	from EvoMSA import BoW, DenseBoW, StackGeneralization
	from EvoMSA.utils import Linear, b4msa_params
	from sklearn.model_selection import StratifiedKFold
	import numpy as np


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

	def bow_training_set(lang, tr, vs, **kwargs):
		params = b4msa_params(lang=lang)
		del params['token_max_filter']
		del params['max_dimension']
		bow_no_pre = BoW(lang=lang, pretrain=False, b4msa_kwargs=params).fit(tr)
		return bow_no_pre.predict(vs)


:ref:`StackGeneralization` with :ref:`BoW` and :ref:`DenseBoW` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	def stack_bow_keywords_emojis(lang, tr, vs, **kwargs):
		bow = BoW(lang=lang)
		keywords = DenseBoW(lang=lang, 
                                       emoji=False, 
                                       dataset=False).select(D=tr)
		emojis = DenseBoW(lang=lang, 
                                     keyword=False, 
                                     dataset=False).select(D=tr)
		stack = StackGeneralization(decision_function_models=[bow, keywords, emojis]).fit(tr)
		X = bow.transform(vs)
		for x in [bow, keywords, emojis]:
			x.cache = X    
		return stack.predict(vs)


:ref:`StackGeneralization` with :ref:`BoW` and :ref:`DenseBoW` using :py:attr:`voc_selection` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	def stack_bow_keywords_emojis_voc_selection(lang, tr, vs, **kwargs):
		bow = BoW(lang=lang, voc_selection='most_common')
		keywords = DenseBoW(lang=lang, voc_selection='most_common',
                                       emoji=False, 
                                       dataset=False).select(D=tr)
		emojis = DenseBoW(lang=lang, voc_selection='most_common',
                                     keyword=False, 
                                     dataset=False).select(D=tr)
		stack = StackGeneralization(decision_function_models=[bow, keywords, emojis]).fit(tr)
		X = bow.transform(vs)
		for x in [bow, keywords, emojis]:
			x.cache = X    
		return stack.predict(vs)


:ref:`StackGeneralization` with two :ref:`BoW` models 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	def stack_bows(lang, tr, vs, **kwargs):
		bow = BoW(lang=lang)
		bow2 = BoW(lang=lang, voc_selection='most_common')
		stack = StackGeneralization(decision_function_models=[bow, bow2]).fit(tr)
		return stack.predict(vs)


:ref:`StackGeneralization` using :ref:`BoW` and :ref:`DenseBoW` with and without :py:attr:`voc_selection` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^		

.. code-block:: python

	def stack_2_bow_keywords(lang, tr, vs, **kwargs):
		bow = BoW(lang=lang)      
		keywords = DenseBoW(lang=lang, dataset=False).select(D=tr)
		bow2 = BoW(lang=lang, voc_selection='most_common')
		keywords2 = DenseBoW(lang=lang, voc_selection='most_common',
										dataset=False).select(D=tr)
		stack = StackGeneralization(decision_function_models=[bow, bow2,
		                                                      keywords,
															  keywords2]).fit(tr)
		X = bow.transform(vs)
		for x in [bow, keywords]:
			x.cache = X
		X = bow2.transform(vs)
		for x in [bow2, keywords2]:
			x.cache = X
		return stack.predict(vs)


:ref:`StackGeneralization` using :ref:`BoW` and tailored :ref:`DenseBoW` with and without :py:attr:`voc_selection` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	def stack_2_bow_tailored_keywords(lang, tr, vs, keywords=None, **kwargs):
		models = [Linear(**kwargs)
				for kwargs in tweet_iterator(keywords)]    
		bow = BoW(lang=lang)      
		keywords = DenseBoW(lang=lang, dataset=False)
		keywords.text_representations_extend(models)
		keywords.select(D=tr)
		bow2 = BoW(lang=lang, voc_selection='most_common')
		keywords2 = DenseBoW(lang=lang, voc_selection='most_common',
										dataset=False).select(D=tr)
		stack = StackGeneralization(decision_function_models=[bow, bow2,
		                                                      keywords,
															  keywords2]).fit(tr)
		X = bow.transform(vs)
		for x in [bow, keywords]:
			x.cache = X
		X = bow2.transform(vs)
		for x in [bow2, keywords2]:
			x.cache = X
		return stack.predict(vs)


:ref:`StackGeneralization` using :ref:`BoW` and all :ref:`DenseBoW` with and without :py:attr:`voc_selection` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	def stack_2_bow_all_keywords(lang, tr, vs, **kwargs):
		bow = BoW(lang=lang)      
		keywords = DenseBoW(lang=lang)
		sel = [k for k, v in enumerate(keywords.names) if v not in ['davincis2022_1'] or 'semeval2023' not in v]
		keywords.select(sel).select(D=tr)
		bow2 = BoW(lang=lang, voc_selection='most_common')
		keywords2 = DenseBoW(lang=lang,
										voc_selection='most_common').select(sel).select(D=tr)
		stack = StackGeneralization(decision_function_models=[bow, bow2, keywords, keywords2]).fit(tr)
		X = bow.transform(vs)
		for x in [bow, keywords]:
			x.cache = X
		X = bow2.transform(vs)
		for x in [bow2, keywords2]:
			x.cache = X
		return stack.predict(vs)


:ref:`StackGeneralization` using :ref:`BoW` tailored and datasets :ref:`DenseBoW` with and without :py:attr:`voc_selection` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	def stack_2_bow_tailored_all_keywords(lang, tr, vs, keywords=None, **kwargs):
		models = [Linear(**kwargs)
				for kwargs in tweet_iterator(keywords)]    
		bow = BoW(lang=lang)      
		keywords = DenseBoW(lang=lang)
		sel = [k for k, v in enumerate(keywords.names)
			if v not in ['davincis2022_1'] or 'semeval2023' not in v]
		keywords.select(sel)
		keywords.text_representations_extend(models)
		keywords.select(D=tr)
		bow2 = BoW(lang=lang, voc_selection='most_common')
		keywords2 = DenseBoW(lang=lang,
										voc_selection='most_common').select(sel).select(D=tr)
		stack = StackGeneralization(decision_function_models=[bow, bow2, keywords, keywords2]).fit(tr)
		X = bow.transform(vs)
		for x in [bow, keywords]:
			x.cache = X
		X = bow2.transform(vs)
		for x in [bow2, keywords2]:
			x.cache = X
		return stack.predict(vs)


:ref:`StackGeneralization` with three :ref:`BoW` models 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^	

.. code-block:: python

	def stack_3_bows(lang, tr, vs, **kwargs):
		params = b4msa_params(lang=lang)
		del params['token_max_filter']
		del params['max_dimension']
		bow_no_pre = BoW(lang=lang, pretrain=False, b4msa_kwargs=params)
		bow = BoW(lang=lang)
		bow2 = BoW(lang=lang, voc_selection='most_common')
		stack = StackGeneralization(decision_function_models=[bow_no_pre, bow, bow2]).fit(tr)
		return stack.predict(vs)


:ref:`StackGeneralization` using :ref:`BoW` and all :ref:`DenseBoW` with and without :py:attr:`voc_selection` plus :ref:`BoW` trained on the training set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	def stack_3_bows_all_keywords(lang, tr, vs, keywords=None, **kwargs):
		models = [Linear(**kwargs)
				for kwargs in tweet_iterator(keywords)]
		params = b4msa_params(lang=lang)
		del params['token_max_filter']
		del params['max_dimension']
		bow_no_pre = BoW(lang=lang, pretrain=False, b4msa_kwargs=params)

		bow = BoW(lang=lang)      
		keywords = DenseBoW(lang=lang, dataset=False)
		keywords.text_representations_extend(models)
		keywords.select(D=tr)
		bow2 = BoW(lang=lang, voc_selection='most_common')
		keywords2 = DenseBoW(lang=lang, voc_selection='most_common',
										dataset=False).select(D=tr)
		stack = StackGeneralization(decision_function_models=[bow_no_pre, bow, bow2, 
															keywords, keywords2]).fit(tr)
		X = bow.transform(vs)
		for x in [bow, keywords]:
			x.cache = X
		X = bow2.transform(vs)
		for x in [bow2, keywords2]:
			x.cache = X
		return stack.predict(vs)


:ref:`StackGeneralization` using :ref:`BoW` and all :ref:`DenseBoW` with and without :py:attr:`voc_selection` plus :ref:`BoW` trained on the training set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	def stack_3_bow_tailored_all_keywords(lang, tr, vs, keywords=None, **kwargs):
		params = b4msa_params(lang=lang)
		del params['token_max_filter']
		del params['max_dimension']
		bow_no_pre = BoW(lang=lang, pretrain=False, b4msa_kwargs=params)
		models = [Linear(**kwargs)
				for kwargs in tweet_iterator(keywords)]    
		bow = BoW(lang=lang)      
		keywords = DenseBoW(lang=lang)
		sel = [k for k, v in enumerate(keywords.names)
			if v not in ['davincis2022_1'] or 'semeval2023' not in v]
		keywords.select(sel)
		keywords.text_representations_extend(models)
		keywords.select(D=tr)
		bow2 = BoW(lang=lang, voc_selection='most_common')
		keywords2 = DenseBoW(lang=lang,
										voc_selection='most_common').select(sel).select(D=tr)
		stack = StackGeneralization(decision_function_models=[bow_no_pre, bow, bow2,
															keywords, keywords2]).fit(tr)
		X = bow.transform(vs)
		for x in [bow, keywords]:
			x.cache = X
		X = bow2.transform(vs)
		for x in [bow2, keywords2]:
			x.cache = X
		return stack.predict(vs)


Predictions
------------------------------

Competitions
------------------------------
