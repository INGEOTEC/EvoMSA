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
		:target: https://colab.research.google.com/github/INGEOTEC/EvoMSA/blob/master/docs/Quickstart.ipynb	   



EvoMSA is a stack generalization algorithm specialized in text classification problems. A text classifier :math:`c`, can be seen as a composition of two functions, i.e., :math:`c \equiv g \circ m`; where :math:`m` transforms the text into a vector space, i.e., :math:`m: \text{text} \rightarrow \mathbb R^d` and :math:`g` is the classifier (:math:`g: \mathbb R^d \rightarrow \mathbb N`) or regressor (:math:`g: \mathbb R^d \rightarrow \mathbb R`). Stack generalization is a technique to combine classifiers (regressors) to produce another classifier (regressor) responsible for making the prediction. 

:ref:`v2` removes, from :ref:`EvoMSA <v1>`, two text representations, i.e., functions :math:`m`, particularly the sentiment lexicon-based model, and the aggressiveness model. It was decided to remove them because these models are the ones that require more work to be implemented in another language and, on the other hand, are the ones that contribute less to the performance of the algorithm. However, :ref:`v2` increments the number of human-annotated models, the emoji models, and introduces a new model, namely keyword models.

:ref:`v2` supports more languages than the previous version, currently it supports Arabic (ar), Catalan (ca), German (de), English (en), Spanish (es), French (fr), Hindi (hi), Indonesian (in), Italian (it), Japanese (ja), Korean (ko), Dutch (nl), Polish (pl), Portuguese (pt), Russian (ru), Tagalog (tl), Turkish (tr), and Chinese (zh). It also provides pre-trained models that include the bag-of-words text representations, emoji, and keyword models. These models were trained on Twitter data. 

The other enhancement is on the implementation. There are three main classes: 

.. toctree::
   :maxdepth: 1

   bow
   text_repr
   stack

:ref:`BoW` and :ref:`DenseBoW` are text classifiers; :ref:`BoW` is the parent of :ref:`DenseBoW`. The stack generalization technique is implemented in :ref:`StackGeneralization`.

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

	
Installing EvoMSA
====================================

EvoMSA can be easly install using anaconda

.. code:: bash

	  conda install -c conda-forge EvoMSA

or can be install using pip, it depends on numpy, scipy, scikit-learn and b4msa.

.. code:: bash

	  pip install cython
	  pip install sparsearray
	  pip install evodag
	  pip install EvoMSA


:ref:`Text Classifier Competitions <competition>`
=====================================================

:ref:`v2` has been tested in many text classifier competitions without modifications. The aim is to offer a better understanding of how these algorithms perform in a new situation and what would be the difference in performance with an algorithm tailored to the new problem. In the following link, we will describe the specifics of each configuration.

.. toctree::
   :maxdepth: 1

   competition

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