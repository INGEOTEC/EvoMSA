.. EvoMSA documentation master file, created by
   sphinx-quickstart on Fri Aug  3 07:02:12 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EvoMSA
==================================
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

.. image:: https://readthedocs.org/projects/evomsa/badge/?version=latest
		:target: https://evomsa.readthedocs.io/en/docs/?badge=docs

.. image:: https://colab.research.google.com/assets/colab-badge.svg
		:target: https://colab.research.google.com/github/INGEOTEC/EvoMSA/blob/master/docs/Quickstart.ipynb	   

EvoMSA is a Sentiment Analysis System based on `B4MSA
<https://github.com/ingeotec/b4msa>`_ and `EvoDAG
<https://github.com/mgraffg/EvoDAG>`_. EvoMSA is a stack
generalization algorithm specialized on text classification
problems. It works by combining the output of different :ref:`text models <tm>` to
produce the final prediction.

EvoMSA is a two-stage procedure; the first step transforms the text
into a vector space with dimensions related to the number of classes, and then,
the second stage trains a supervised learning algorithm.

The first stage is a composition of two functions, :math:`g \circ m`, where
:math:`m` is a :ref:`text model <tm>` that transforms a text into a vector (i.e., :math:`m: \text{text} \rightarrow \mathbb R^d`)
and :math:`g` is a classifier or regressor (i.e., :math:`g: \mathbb R^d \rightarrow \mathbb R^c`),
:math:`d` depends on :math:`m`, and :math:`c` is the number of classes or labels.

EvoMSA contains different :ref:`text models <tm>` (i.e., :math:`m`), which can be selected using flags in the class constructor.
The :ref:`text models <tm>` implemented are:

* :py:class:`b4msa.textmodel.TextModel` model trained with the training set (it is set by default :py:attr:`TR`)
* :ref:`emospace` (it is evoked using :py:attr:`EvoMSA.base.EvoMSA(Emo=True, lang="en")`)
* :ref:`th` (it is evoked using :py:attr:`EvoMSA.base.EvoMSA(TH=True, lang="en")`)  
* :ref:`ha` (it is evoked using :py:attr:`EvoMSA.base.EvoMSA(HA=True, lang="en")`)
* :ref:`aggress` (it is evoked using :py:attr:`EvoMSA.base.EvoMSA(Aggress=True, lang="en")`)

where :py:attr:`lang` specifies the language and can be either *ar*,
*en*, or, *es* that corresponds to Arabic, English, and Spanish,
respectively. On the other hand, :math:`g` is a classifier or regressor, and by default,
it uses :py:class:`sklearn.svm.LinearSVC`.

The second stage is the stacking method, which is a classifier or
regressor. EvoMSA uses by default EvoDAG (i.e.,
:py:class:`EvoDAG.model.EvoDAGE`); however, this method can be changed
with tha parameter :py:attr:`stacked_method`, e.g.,
:py:attr:`EvoMSA.base.EvoMSA(stacked_method="sklearn.naive_bayes.GaussianNB")`.

EvoMSA is described in `EvoMSA: A Multilingual Evolutionary Approach
for Sentiment Analysis <https://ieeexplore.ieee.org/document/8956106>`_, Mario Graff, Sabino Miranda-Jimenez, Eric
Sadit Tellez, Daniela Moctezuma. Computational Intelligence Magazine, vol 15 no. 1, pp. 76-88, Feb. 2020.
In this document, we try to follow as much as possible the notation used in the CIM paper; we believe
this can help to grasp as easily as possible EvoMSA's goals. 

Quickstart Guide
===================

We have decided to make a live quickstart guide, it covers the
installation, the use of EvoMSA with different text models, and it
ends by explaining how the text models can be used on their
own. Finally, the notebook can be found at the docs directory on
GitHub.

.. raw:: html

	 <iframe width="560" height="315" src="https://www.youtube.com/embed/cyg4wBrJZdU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


Citing
======

If you find EvoMSA useful for any academic/scientific purpose, we
would appreciate citations to the following reference:
  
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
=======================

EvoMSA can be easly install using anaconda

.. code:: bash

	  conda install -c conda-forge EvoMSA

or can be install using pip, it depends on numpy, scipy, 
scikit-learn and b4msa.

.. code:: bash

	  pip install cython
	  pip install sparsearray
	  pip install evodag
	  pip install EvoMSA

Usage
=========

EvoMSA can be used from using the following commands.

Read the dataset

>>> from EvoMSA import base
>>> from microtc.utils import tweet_iterator
>>> import os
>>> tweets = os.path.join(os.path.dirname(base.__file__), 'tests', 'tweets.json')
>>> D = list(tweet_iterator(tweets))
>>> X = [x['text'] for x in D]
>>> y = [x['klass'] for x in D]

Once the dataset is loaded, it is time to create an EvoMSA model, let
us create an EvoMSA model enhaced with :ref:`emospace`.

>>> from EvoMSA.base import EvoMSA
>>> evo = EvoMSA(Emo=True, lang='es').fit(X, y)

Predict a sentence in Spanish

>>> evo.predict(['EvoMSA esta funcionando'])

EvoMSA uses by default :py:class:`EvoDAG.model.EvoDAGE` as stacked classifier; however, this is a parameter that can be modified. Let us, for example use :py:class:`sklearn.naive_bayes.GaussianNB` in the previous example.

>>> evo = EvoMSA(Emo=True, lang='es',
                 stacked_method='sklearn.naive_bayes.GaussianNB').fit(X, y)
>>> evo.predict(['EvoMSA esta funcionando'])

.. _tm:

Text Models
=================

Besides the default text model (i.e.,
:py:class:`b4msa.textmodel.TextModel`), EvoMSA has four text models
(EvoMSA's CIM paper presents only the first three models) for Arabic,
English and Spanish languages that can be selected with a flag in the
constructor, these are:

.. toctree::
   :maxdepth: 2

   emospace
   th
   ha
   aggress
   
..
   * :ref:`emospace`.
   * :ref:`th`. 
   * :ref:`ha`.
   * :ref:`aggress`.  

Nonetheless, more text models can be included in EvoMSA. EvoMSA's core
idea is to facilitate the inclusion of diverse text models. We have
been using EvoMSA (as INGEOTEC team) on different competitions run at
the Workshop of Semantic Evaluation as well as other
sentiment-analysis tasks and traditional text classification problems.

During this time, we have created different text models -- some of
them using the datasets provided by the competition's organizers and
others inspired by our previous work -- in different languages. We
have decided to make public these text models organizing them by language.

.. toctree::
   :maxdepth: 2
   
   arabic
   english
   spanish
   cites


EvoMSA's classes
==================

.. toctree::
   :maxdepth: 2

   base  
   utils
   
