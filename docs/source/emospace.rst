.. _emospace:

Emoji space
===========

This text model is inspired by `DeepMoji
<https://arxiv.org/abs/1708.00524>`_; the idea is to create a function
:math:`m: \text{text} \rightarrow \mathbb{R}^{64}` that predicts which
emoji would be the most probable given a text. To do so, we learn a
B4MSA model using 3.2 millon examples of the 64 most frequent emojis
per language. The Emoji Space is created for Arabic, English and
Spanish, and these models can be used by importing the following classes:

* :py:class:`EvoMSA.model.EmoSpaceAr`
* :py:class:`EvoMSA.model.EmoSpaceEn`
* :py:class:`EvoMSA.model.EmoSpaceEs`

These models can be tested as follow:

>>> from EvoMSA.model import EmoSpaceEn
>>> emo = EmoSpaceEn()
>>> emo['good morning']

Inside EvoMSA these models can be selected using the parameters
:py:attr:`EvoMSA.base.EvoMSA(Emo=True, lang="en")` where
:py:attr:`lang` specifies the language and can be either *ar*, *en*, or *es*. 
    
  
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
