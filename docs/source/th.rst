.. _th:


Sentiment lexicon-based model
=================================

This model introduces a Sentiment Lexicon-based model into EvoMSA. The
idea is to count the number of positive and negative words that appear
on an affective lexicon. This model is  appropriately described `here <https://arxiv.org/abs/1812.02307>`_.

This model has been implemented for Arabic, English, and Spanish,
and can be used as follows:

For example, let us read a dataset to train EvoMSA.

>>> from EvoMSA import base
>>> from microtc.utils import tweet_iterator
>>> import os
>>> tweets = os.path.join(os.path.dirname(base.__file__), 'tests', 'tweets.json')
>>> D = [[x['text'], x['klass']] for x in tweet_iterator(tweets)]

Once the dataset is load, EvoMSA using lexicon model is
trained as follows:

>>> from EvoMSA.base import EvoMSA
>>> evo = EvoMSA(TH=True, lang='es').fit([x[0] for x in D], [x[1] for x in D])
>>> evo.predict(['buenos dias'])

Particularly, the following classes implement the lexicon-based model:

* :py:class:`EvoMSA.model.ThumbsUpDownAr`
* :py:class:`EvoMSA.model.ThumbsUpDownEn`
* :py:class:`EvoMSA.model.ThumbsUpDownEs`

These models can be tested as follow:

>>> from EvoMSA.model import ThumbsUpDownEn
>>> th = ThumbsUpDownEn()
>>> th['good morning']


Lexicon-based model's classes
=================================

.. autoclass:: EvoMSA.model.ThumbsUpDownAr		 
	       :members:

.. autoclass:: EvoMSA.model.ThumbsUpDownEn
	       :members:
		  
.. autoclass:: EvoMSA.model.ThumbsUpDownEs
	       :members:

