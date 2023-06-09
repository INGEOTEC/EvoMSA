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