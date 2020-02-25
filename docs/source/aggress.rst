.. _aggress:


Aggressiveness Model
=====================

This model is a binary bag of words where the words are aggressive words. That is, it only considers the presence or absence of aggressive words. The number of words for Arabic is 289,  English 352, and Spanish 214.

This model can be used as follow, firstly let us read a dataset to train EvoMSA.

>>> from EvoMSA import base
>>> from microtc.utils import tweet_iterator
>>> import os
>>> tweets = os.path.join(os.path.dirname(base.__file__), 'tests', 'tweets.json')
>>> D = list(tweet_iterator(tweets))
>>> X = [x['text'] for x in D]
>>> y = [x['klass'] for x in D]

Once the dataset is load, EvoMSA using Aggressiveness Space in Spanish is
trained as follows:

>>> from EvoMSA.base import EvoMSA
>>> evo = EvoMSA(Aggress=True, lang='es').fit(X, y)
>>> evo.predict(['buenos dias'])
