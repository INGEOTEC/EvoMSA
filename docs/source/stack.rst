.. _StackGeneralization:

====================================
:py:class:`StackGeneralization`
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

Different text classifiers have been implemented in :ref:`v2`, namely :ref:`BoW`, and :ref:`DenseBoW`; the next step is to use an algorithm to combine the outputs of these classifiers. We combined the classifier's outputs using a stack generalization approach (implemented in :py:class:`StackGeneralization`).

The idea behind stack generalization is to train an estimator on the predictions made by the base classifiers or regressors. The estimator trained is the one that will make the final prediction. In order to train it, one needs a different dataset than the one used on the base algorithms. However, it is also feasible to emulate this new dataset using k-fold cross-validation. That is, the first step is to train and predict all the elements of the training set using the base learning algorithms. Then, the predictions obtained are the inputs of the estimator using the corresponding outputs. 

Text Representation
--------------------------------

The stack generalization method is introduced by first describing the vector space used to represent the text. The first step is to have a labeled dataset; the dataset used can be found in EvoMSA with the following instructions. 

>>> from microtc.utils import tweet_iterator
>>> from EvoMSA.tests.test_base import TWEETS
>>> D = list(tweet_iterator(TWEETS))

The second step is initializing the base estimators, i.e., the text classifiers. We used the text classifiers described so far, namely :ref:`BoW` and :ref:`DenseBoW`. These classifiers are initialized with the following instructions. 

>>> from EvoMSA import BoW, DenseBoW, StackGeneralization
>>> SIZE = 15
>>> LANG = 'es'
>>> bow = BoW(lang=LANG,
              voc_size_exponent=SIZE)
>>> dense = DenseBoW(lang=LANG,
                     voc_size_exponent=SIZE,
                     emoji=True,
                     keyword=True,
                     dataset=False)

Once the text classifiers are ready, these can be used on the stacking. There are two ways in which the models can be included; one is using the parameter :py:attr:`decision_function_models` and the other with the parameter :py:attr:`transform_models`. The former parameter is the default behavior of stack generalization, where the estimators are trained, and then their predictions are used to train another learning algorithm. The latter parameter is to include the vector space of the models as features of the stack estimator. More information about this will be provided later.

The following example creates a stack generalization using the standard approach -- where the predictions of the base classifiers are used as features. 

>>> stack = StackGeneralization(decision_function_models=[bow, dense]).fit(D)

.. note::

	It is equivalent to use the following instructions. 

	>>> import numpy as np
	>>> X = [x['text'] for x in D]
	>>> y = np.r_[[x['klass'] for x in D]]
	>>> stack = StackGeneralization([bow, dense]).fit(X, y)

In order to illustrate the vector space used by the stack classifier, the following instruction represent the text *buenos días* (*good morning*) in the space created by the base classifiers. As can be seen, the vector is in :math:`\mathbb R^8`, where the first four components correspond to the predictions made by the :ref:`BoW <bow_tc>` model, and the last four are the :ref:`DenseBoW <text_repr_tc>` predictions. 

>>> stack.transform(['buenos días'])
array([[-1.44976031, -1.04133473, -0.2415961, 0.58071146,
	    -2.2368439, -1.21958811, -0.4779458, 1.37261126]])

The parameter :py:attr:`transform_models` is exemplified in the following instruction. It can be observed that the model :py:attr:`text_repr` is used as input for the parameter. The difference is depicted when the text is transformed into the vector space. The vector space is in :math:`\mathbb R^{2676}`, where the last four components are the predictions of the :ref:`BoW <bow_tc>` model, and the rest correspond to the :py:attr:`transform` method of :ref:`DenseBoW <text_repr_vector_space>`. 

>>> stack2 = StackGeneralization(decision_function_models=[bow], 
                                 transform_models=[dense]).fit(D)
>>> X = stack2.transform(['buenos días'])
>>> X.shape
(1, 2619)
>>> X[0, -4:]
array([-1.4497599 , -1.04133594, -0.24159535,  0.58070953])

Text Classifier
--------------------------------

The last step is to use the instance to predict the text *buenos días* (*good morning*), as done in the following instruction.

>>> stack.predict(['buenos días'])
array(['P'], dtype='<U4')

where the label 'P' corresponds to the positive class. 

There are scenarios where it is more important to estimate the value(s) used to classify a particular instance; in the case of SVM, this is known as the decision function, and in the case of a Naive Bayes classifier, this is the probability of each class. This information can be found in :py:attr:`StackGeneralization.decision_function` as can be seen in the following code.

>>> stack.decision_function(['buenos días'])
array([[1.36906246e-09, 3.17821439e-17, 3.12052669e-04, 9.99687946e-01]])


API
--------------------------------

.. toctree::
   :maxdepth: 2

   stack_api
