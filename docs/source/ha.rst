.. _ha:

Human annotated model
=====================

The human annotated model uses `B4MSA
<https://github.com/ingeotec/b4msa>`_  trained on a sentiment analysis
dateset, these datasets are for Arabic, English, and Spanish.

This model can be used as follow, firstly let us read a dataset to train EvoMSA.

>>> from EvoMSA import base
>>> from microtc.utils import tweet_iterator
>>> import os
>>> tweets = os.path.join(os.path.dirname(base.__file__), 'tests', 'tweets.json')
>>> D = [[x['text'], x['klass']] for x in tweet_iterator(tweets)]

Once the dataset is load, EvoMSA using Emoji Space in Spanish is
trained as follows:

>>> from EvoMSA.base import EvoMSA
>>> evo = EvoMSA(HA=True, lang='es').fit([x[0] for x in D], [x[1] for x in D])
>>> evo.predict(['buenos dias'])



Human annotated's classes
================================

.. autoclass:: EvoMSA.model.LabeledDataSet
	       :members:
