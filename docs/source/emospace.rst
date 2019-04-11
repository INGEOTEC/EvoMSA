.. _emospace:

Emoji space
===========

This text model is inspired by `DeepMoji
<https://arxiv.org/abs/1708.00524>`_; the idea is to create a function
:math:`m: \text{text} \rightarrow \mathbb{R}^{64}` that predicts which
emoji would be the most probable given a text. To do so, we learn a
B4MSA model using 3.2 million examples of the 64 most frequent emojis
per language. The result is that emojis are different for each
language; the emoji used can be seen in this `manuscript
<https://arxiv.org/abs/1812.02307>`_ Figure 2. 

The Emoji Space is created for Arabic, English and
Spanish. These models can be selected using the parameters
:py:attr:`EvoMSA.base.EvoMSA(Emo=True, lang="en")` where
:py:attr:`lang` specifies the language and can be either *ar*, *en*, or *es*. 
    
Particularly, the following classes implement the Emoji Space:

* :py:class:`EvoMSA.model.EmoSpaceAr`
* :py:class:`EvoMSA.model.EmoSpaceEn`
* :py:class:`EvoMSA.model.EmoSpaceEs`

These models can be tested as follow:

>>> from EvoMSA.model import EmoSpaceEn
>>> emo = EmoSpaceEn()
>>> emo['good morning']

it can be observed that the output is a vector :math:`\in
\mathbb{R}^{64}` where each component correspond an emoji which is
stored in the following list

>>> emo._labels

Emoji fixed for all languages
=====================================

Trying to address another type of applications such as transfer
learning, we decided to use the 64 most frequent emojis in English in
the construction of Emoji Space. 

These models can be download from

* `Arabic Emoji Space <http://ingeotec.mx/~mgraffg/models/emo-static-ar.evoemo>`_
* `English Emoji Space <http://ingeotec.mx/~mgraffg/models/emo-static-en.evoemo>`_
* `Spanish Emoji Space <http://ingeotec.mx/~mgraffg/models/emo-static-es.evoemo>`_

These models can be used in EvoMSA as follows:

>>> from EvoMSA.base import EvoMSA
>>> evo = EvoMSA(models=[['emo-static-en.evoemo', 'sklearn.svm.LinearSVC']]).fit([x[0] for x in D], [x[1] for x in D])
>>> evo.predict(['good morning'])

These models can be tested by their own as follows:

>>> from microtc.utils import load_model
>>> emo = load_model('emo-static-en.evoemo')
>>> emo['good morning']


Emoji space's classes
=======================

.. autoclass:: EvoMSA.model.EmoSpace
	       :members:
	       :private-members:
		 
.. autoclass:: EvoMSA.model.EmoSpaceAr   
	       :members:

.. autoclass:: EvoMSA.model.EmoSpaceEn
	       :members:

.. autoclass:: EvoMSA.model.EmoSpaceEs   
	       :members:
