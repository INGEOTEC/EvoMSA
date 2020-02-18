.. EvoMSA documentation master file, created by
   sphinx-quickstart on Fri Aug  3 07:02:12 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EvoMSA
==================================
.. image:: https://travis-ci.org/INGEOTEC/EvoMSA.svg?branch=master
	   :target: https://travis-ci.org/INGEOTEC/EvoMSA

.. image:: https://ci.appveyor.com/api/projects/status/wg01w00evm7pb8po?svg=true
	   :target: https://ci.appveyor.com/project/mgraffg/evomsa

.. image:: https://coveralls.io/repos/github/INGEOTEC/EvoMSA/badge.svg?branch=master	    
	   :target: https://coveralls.io/github/INGEOTEC/EvoMSA?branch=master

.. image:: https://anaconda.org/ingeotec/evomsa/badges/version.svg
	   :target: https://anaconda.org/ingeotec/evomsa

.. image:: https://badge.fury.io/py/EvoMSA.svg
	   :target: https://badge.fury.io/py/EvoMSA

.. image:: https://readthedocs.org/projects/evomsa/badge/?version=latest
	   :target: https://evomsa.readthedocs.io/en/latest/?badge=latest

EvoMSA is a Sentiment Analysis System based on `B4MSA
<https://github.com/ingeotec/b4msa>`_ and `EvoDAG
<https://github.com/mgraffg/EvoDAG>`_. EvoMSA is a stack
generalization algorithm specialized on text classification
problems. It works by combining the output of different text models to
produce the final prediction.

EvoMSA is a two-stage procedure; the first step transforms the text
into a vector space with dimensions related to the number of classes, and then,
the second stage trains a supervised learning algorithm.

The first stage is a composition of two functions, :math:`g \circ m`, where
:math:`m` is a text model that transforms a text into a vector (i.e., :math:`m: \text{text} \rightarrow \mathbb R^d`)
and :math:`g` is a classifier or regressor (i.e., :math:`g: \mathbb R^d \rightarrow \mathbb R^c`),
:math:`d` depends on :math:`m`, and :math:`c` is the number of classes or labels.

EvoMSA contains different text models (i.e., :math:`m`), which can be selected using flags in the class constructor.
The text models implemented are:

* :py:class:`b4msa.textmodel.TextModel` model trained with the training set (it is set by default :py:attr:`TR`)
* :ref:`emospace` (it is evoked using :py:attr:`EvoMSA.base.EvoMSA(Emo=True, lang="en")`)
* :ref:`th` (it is evoked using :py:attr:`EvoMSA.base.EvoMSA(TH=True, lang="en")`)  
* :ref:`ha` (it is evoked using :py:attr:`EvoMSA.base.EvoMSA(HA=True, lang="en")`)

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

	  conda install -c ingeotec EvoMSA

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
>>> D = [[x['text'], x['klass']] for x in tweet_iterator(tweets)]

Once the dataset is loaded, it is time to create an EvoMSA model, let
us create an EvoMSA model enhaced with :ref:`emospace`.

>>> from EvoMSA.base import EvoMSA
>>> evo = EvoMSA(Emo=True, lang='es').fit([x[0] for x in D], [x[1] for x in D])

Predict a sentence in Spanish

>>> evo.predict(['EvoMSA esta funcionando'])


Text Models
=================

Besides the default text model (i.e.,
:py:class:`b4msa.textmodel.TextModel`), EvoMSA has three text models
for Arabic, English and Spanish languages that can be selected with a
flag in the constructor, these are:


* :ref:`emospace`.
* :ref:`th`. 
* :ref:`ha`. 

Nonetheless, more text models can be included in EvoMSA. EvoMSA's core
idea is to facilitate the inclusion of diverse text models. We have
been using EvoMSA (as INGEOTEC team) on different competitions run at
the Workshop of Semantic Evaluation as well as other sentiment
analysis tasks and traditional text classification problems.

During this time, we have created different text models -- some of
them using the datasets provided by the competition's organizers and
others inspired by our previous work -- in different languages. We
have decided to make public these text models created, which are
organized by Language.

.. toctree::
   :maxdepth: 2
   
   arabic
   english
   spanish


EvoMSA's classes
==================

.. toctree::
   :maxdepth: 2

   base
   emospace
   th
   ha
   utils
   model_selection
   
