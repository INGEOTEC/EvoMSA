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

EvoMSA ia a Sentiment Analysis System based on `B4MSA
<https://github.com/ingeotec/b4msa>`_ and `EvoDAG
<https://github.com/mgraffg/EvoDAG>`_. EvoMSA is a stack
generalisation algorithm specialised on text classification
problems. It works by combining the output of different text models to
produce the final prediction.

:ref:`emospace`

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


Extra modules
==================	      

.. toctree::
   :maxdepth: 2

   utils      
   
