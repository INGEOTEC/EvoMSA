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
generalisation algorithm specialised on text classification
problems. It works by combining the output of different text models to
produce the final prediction. These text models are:

* B4MSA model trained with the training set (it is set by default)
* :ref:`emospace` (it is evoked using :py:attr:`EvoMSA.base.EvoMSA(Emo=True, lang="en")`)
* :ref:`th` (it is evoked using :py:attr:`EvoMSA.base.EvoMSA(TH=True, lang="en")`)  
* :ref:`ha` (it is evoked using :py:attr:`EvoMSA.base.EvoMSA(HA=True, lang="en")`)

where :py:attr:`lang` specifies the language and can be either *ar*,
*en*, or *es*, that corresponds to Arabic, English and Spanish,
respectively.

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


Citing
======

If you find EvoMSA useful for any academic/scientific purpose, we
would appreciate citations to the following reference:
  
.. code:: bibtex

	  @article{DBLP:journals/corr/abs-1812-02307,
	  author = {Mario Graff and Sabino Miranda{-}Jim{\'{e}}nez
	                 and Eric Sadit Tellez and Daniela Moctezuma},
	  title     = {EvoMSA: {A} Multilingual Evolutionary Approach for Sentiment Analysis},
	  journal   = {CoRR},
	  volume    = {abs/1812.02307},
	  year      = {2018},
	  url       = {http://arxiv.org/abs/1812.02307},
	  archivePrefix = {arXiv},
	  eprint    = {1812.02307},
	  timestamp = {Tue, 01 Jan 2019 15:01:25 +0100},
	  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1812-02307},
	  bibsource = {dblp computer science bibliography, https://dblp.org}
	  }

	
Installing EvoMSA
=======================

EvoMSA can be easly install using anaconda

.. code:: bash

	  conda install -c ingeotec EvoMSA

or can be install using pip, it depends on numpy, scipy, 
scikit-learn and b4msa.

.. code:: bash
	  
	  pip install EvoMSA

EvoMSA
=============

.. autoclass:: EvoMSA.base.EvoMSA
   :members:
	      
Text Models
==================================
.. toctree::
   :maxdepth: 2

   emospace
   th
   ha


Extra modules
==================	      

.. toctree::
   :maxdepth: 2

   utils      
   
