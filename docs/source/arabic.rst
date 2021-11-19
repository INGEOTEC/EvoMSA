.. _arabic:

Arabic
===========================

The first text model created is based on tweets collected from January
2017 until February 2020. From this collection we randomly selected
approximately 5,000,000, excluding retweets and tweets having less
than two words, to create the text model. The text model is created
using the following paramters.


.. code:: python

	  from b4msa.textmodel import TextModel
	  tm = TextModel(usr_option="delete",
	                 num_option="delete",
                         url_option="delete",
			 emo_option="none",
                         token_min_filter=0.001,
                         token_max_filter=0.999)

The aforementioned model is a bag of word model, where the number of
tokens is 24,906 (i.e., :math:`m_b: \text{text} \rightarrow \mathbb
R^{24906}`).
			 
This model (without using the text model trained with the training set) can be used as follow :
	  
>>> from EvoMSA.base import EvoMSA
>>> evo = EvoMSA(TR=False, B4MSA=True, lang='ar')

The next table shows the different models we have produced for the
Arabic language. The first row contains the :ref:`ha`. The remaining
rows include the different models for two competitions run at
SemEval.

+---------------------------------------------------------------------------------------------------+---------------------------------+
| Source                                                                                            | Model                           |
+===================================================================================================+=================================+
| `Arabic Sentiment Analysis and Cross-lingual Sentiment Resources`_                                | :ref:`ha`                       |
+---------------------------------------------------------------------------------------------------+---------------------------------+
| `SemEval-2017 Task 4: Sentiment analysis in Twitter`_ [Task-4]_                                   | `semeval2017_Ar.evomsa`_        |
+---------------------------------------------------------------------------------------------------+---------------------------------+
| `SemEval-2018 Task 1: Affect in tweets <https://www.aclweb.org/anthology/S18-1001/>`_ [Task-1]_   | `semeval2018_anger_Ar.evomsa`_  |
|                                                                                                   | `semeval2018_fear_Ar.evomsa`_   |
|                                                                                                   | `semeval2018_joy_Ar.evomsa`_    |
|                                                                                                   | `semeval2018_sadness_Ar.evomsa`_|
|                                                                                                   | `semeval2018_valence_Ar.evomsa`_|
+---------------------------------------------------------------------------------------------------+---------------------------------+

The following code shows the usage of two of the models with suffix `.evomsa`. The models used are `semeval2017_Ar.evomsa`_, `semeval2018_valence_Ar.evomsa`_, as well as :math:`m_b` text-model, and without the text model obtained with the training set. 

>>> from EvoMSA.utils import download
>>> sem2017 = download('semeval2017_Ar.evomsa')
>>> valence = download('semeval2018_valence_Ar.evomsa')
>>> evo = EvoMSA(TR=False, B4MSA=True, lang='ar',
                 models=[[sem2017, "sklearn.svm.LinearSVC"],
                         [valence, "sklearn.svm.LinearSVC"]])

where :py:class:`sklearn.svm.LinearSVC` can be any classifier following the structure of `sklearn <https://scikit-learn.org/>`_.
`evo` is a instance of EvoMSA and it is missing to train it with a training set, i.e., call :py:func:`evo.fit`.

.. _Arabic Sentiment Analysis and Cross-lingual Sentiment Resources: http://saifmohammad.com/WebPages/ArabicSA.html
.. _SemEval-2017 Task 4\: Sentiment analysis in Twitter: https://www.aclweb.org/anthology/S17-2088/
.. _semeval2017_Ar.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2017_Ar.evomsa
.. _semeval2018_anger_Ar.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_anger_Ar.evomsa
.. _semeval2018_fear_Ar.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_fear_Ar.evomsa
.. _semeval2018_joy_Ar.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_joy_Ar.evomsa
.. _semeval2018_sadness_Ar.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_sadness_Ar.evomsa
.. _semeval2018_valence_Ar.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_valence_Ar.evomsa
