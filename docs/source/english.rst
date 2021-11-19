.. _english:

English
===========================

The first text model created is based on tweets collected from July
2016 until February 2020. From this collection we randomly selected
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
tokens is 17,827 (i.e., :math:`m_b: \text{text} \rightarrow \mathbb
R^{17827}`).
			 
This model (without using the text model trained with the training set) can be used as follow:
	  
>>> from EvoMSA.base import EvoMSA
>>> evo = EvoMSA(TR=False, B4MSA=True, lang='en')

The next table shows the different models we have produced for the
English language.

+---------------------------------------------------------------------------------------------------+---------------------------------+
| Source                                                                                            | Model                           |
+===================================================================================================+=================================+
| `Sentiment strength detection forthe social web`_ [SS]_                                           | `SS-Youtube_En.evomsa`_         |                         
+---------------------------------------------------------------------------------------------------+---------------------------------+
| `A corpus forresearch on deliberation and debate`_ [SCv1]_                                        | `SCv1_En.evomsa`_               |
+---------------------------------------------------------------------------------------------------+---------------------------------+
| `Creating and characterizing a diverse corpus of sarcasm in dialogue`_ [SCv2-GEN]_                | `SCv2-GEN_En.evomsa`_           |
+---------------------------------------------------------------------------------------------------+---------------------------------+
| `Multilingual Twitter sentiment classification: The role of human annotators`_ [Human-Annotated]_ | :ref:`ha`                       |
+---------------------------------------------------------------------------------------------------+---------------------------------+
| `SemEval-2017 Task 4: Sentiment analysis in Twitter`_ [Task-4]_                                   | `semeval2017_En.evomsa`_        |
+---------------------------------------------------------------------------------------------------+---------------------------------+
| `SemEval-2018 Task 1: Affect in tweets <https://www.aclweb.org/anthology/S18-1001/>`_ [Task-1]_   | `semeval2018_anger_En.evomsa`_  |
|                                                                                                   | `semeval2018_fear_En.evomsa`_   |
|                                                                                                   | `semeval2018_joy_En.evomsa`_    |
|                                                                                                   | `semeval2018_sadness_En.evomsa`_|
|                                                                                                   | `semeval2018_valence_En.evomsa`_|
+---------------------------------------------------------------------------------------------------+---------------------------------+

The following code shows the usage of two of the models with suffix `.evomsa`. The models used are `semeval2017_En.evomsa`_, `SCv1_En.evomsa`_, as well as :math:`m_b` text-model, and without the text model obtained with the training set. 

>>> from EvoMSA.utils import download
>>> sem2017 = download('semeval2017_En.evomsa')
>>> scv1 = download('SCv1_En.evomsa')
>>> evo = EvoMSA(TR=False, B4MSA=True, lang='ar',
                 models=[[sem2017, "sklearn.svm.LinearSVC"],
                         [scv1, "sklearn.svm.LinearSVC"]])

where :py:class:`sklearn.svm.LinearSVC` can be any classifier following the structure of `sklearn <https://scikit-learn.org/>`_.
`evo` is a instance of EvoMSA and it is missing to train it with a training set, i.e., call :py:func:`evo.fit`.

It is important to mention that all the pre-trained models can be used alone. For example, the following lines show how to use
`SCv1_En.evomsa`_.

>>> from EvoMSA.utils import download
>>> from microtc.utils import load_model
>>> tm = load_model(download('SCv1_En.evomsa'))

At this point the model is in :py:attr:`tm`; this text model can be used with the function :py:func:`tm.predict`.

>>> tm.predict(["Have a nice day", "I hate this movie"])

.. _Sentiment strength detection forthe social web: https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.21662
.. _A corpus forresearch on deliberation and debate: http://www.lrec-conf.org/proceedings/lrec2012/pdf/1078_Paper.pdf
.. _Creating and characterizing a diverse corpus of sarcasm in dialogue: https://www.aclweb.org/anthology/W16-3604/
.. _Multilingual Twitter sentiment classification\: The role of human annotators: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155036
.. _SemEval-2017 Task 4\: Sentiment analysis in Twitter: https://www.aclweb.org/anthology/S17-2088
.. _semeval2017_En.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2017_En.evomsa
.. _semeval2018_anger_En.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_anger_En.evomsa
.. _semeval2018_fear_En.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_fear_En.evomsa
.. _semeval2018_joy_En.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_joy_En.evomsa
.. _semeval2018_sadness_En.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_sadness_En.evomsa
.. _semeval2018_valence_En.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_valence_En.evomsa
.. _SS-Youtube_En.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/SS-Youtube_En.evomsa
.. _SCv1_En.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/SCv1_En.evomsa
.. _SCv2-GEN_En.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/SCv2-GEN_En.evomsa
