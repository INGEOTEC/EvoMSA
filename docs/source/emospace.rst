.. _emospace:

Emoji space
===========

This text model is inspired by `DeepMoji
<https://arxiv.org/abs/1708.00524>`_; the idea is to create a function
:math:`m_{\text{emo}}: \text{text} \rightarrow \mathbb{R}^{64}` that
predicts which emoji would be the most probable given a text. To do
so, we proposed a composition of two functions , i.e., :math:`g \circ
m_b` where :math:`m_b` is created using the procedure described in
:ref:`arabic`, :ref:`english`, and :ref:`spanish` for Arabic, English
and Spanish, respectively. The second part, i.e., :math:`g`, is a
linear SVM trained with 3.2 million examples of the 64 most frequent
emojis per language. The result is that emojis are different for each
language; the emoji used can be seen in this `manuscript
<https://arxiv.org/abs/1812.02307>`_ Figure 2.

The Emoji Space is created for Arabic, English and
Spanish. These models can be selected using the parameters
:py:attr:`EvoMSA.base.EvoMSA(Emo=True, lang="en")` where
:py:attr:`lang` specifies the language and can be either *ar*, *en*, or *es*. 

For example, let us read a dataset to train EvoMSA.

>>> from EvoMSA import base
>>> from microtc.utils import tweet_iterator
>>> import os
>>> tweets = os.path.join(os.path.dirname(base.__file__), 'tests', 'tweets.json')
>>> D = list(tweet_iterator(tweets))
>>> X = [x['text'] for x in D]
>>> y = [x['klass'] for x in D]

Once the dataset is load, EvoMSA using Emoji Space in Spanish is
trained as follows:

>>> from EvoMSA.base import EvoMSA
>>> evo = EvoMSA(Emo=True, lang='es').fit(X, y)
>>> evo.predict(['buenos dias'])


As mentioned previously, the model represents a given text into a 64 dimentional space, one can see this representation as follows. 

>>> emo = evo.textModels[1]
>>> emo['buenos dias']

it can be observed that the output is a vector :math:`\in
\mathbb{R}^{64}` where each component correspond an emoji which is
stored in the following list

>>> emo._labels

The three best-ranked emoji for *good morning* (`buenos dias`) and *I love that song* (`me encanta esa canción`) are:

>>> import numpy as np
>>> [emo._labels[x] for x in np.argsort(emo['buenos dias'])[::-1][:3]]
['😄', '😴', '☺']
>>> [emo._labels[x] for x in np.argsort(emo['me encanta esa canción'])[::-1][:3]]
['💓', '♫', '💞']

