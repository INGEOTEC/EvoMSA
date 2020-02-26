.. _ha:

Human annotated model
=====================

The human-annotated model is a composition of two functions, i.e.,
:math:`g \circ m_b` where :math:`m_b` is created using the procedure
described in :ref:`arabic`, :ref:`english`, and :ref:`spanish` for
Arabic, English and Spanish, respectively.  The classifier :math:`g`
is trained with human-annotated datasets obtained from `Arabic
Sentiment Analysis and Cross-lingual Sentiment Resources
<http://saifmohammad.com/WebPages/ArabicSA.html>`_ for Arabic, and for
English and Spanish it is used the data from [Human-Annotated]_.

The model has the following signature :math:`g \circ m_b: \text{text}
\rightarrow \mathbb R^3`, where the first component corresponds to the
negative class, the second to the neutral, and the third to the
positive.

This model can be used as follow, firstly let us read a dataset to train EvoMSA.

>>> from EvoMSA import base
>>> from microtc.utils import tweet_iterator
>>> import os
>>> tweets = os.path.join(os.path.dirname(base.__file__), 'tests', 'tweets.json')
>>> D = list(tweet_iterator(tweets))
>>> X = [x['text'] for x in D]
>>> y = [x['klass'] for x in D]

Once the dataset is load, EvoMSA using HA Space in Spanish is
trained as follows:

>>> from EvoMSA.base import EvoMSA
>>> evo = EvoMSA(HA=True, lang='es').fit(X, y)
>>> evo.predict(['buenos dias'])
