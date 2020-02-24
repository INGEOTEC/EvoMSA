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
			 

+---------------------------------------------------------------------------------------------------------------------+---------------+
| Source                                                                                                              | Model         |
+=====================================================================================================================+===============+
| `Arabic Sentiment Analysis and Cross-lingual Sentiment Resources <http://saifmohammad.com/WebPages/ArabicSA.html>`_ | xxx           |
+---------------------------------------------------------------------------------------------------------------------+---------------+
| `SemEval-2017 Task 4: Sentiment analysis in Twitter <https://www.aclweb.org/anthology/S17-2088/>`_ [Task-4]_        | yyy           |
+---------------------------------------------------------------------------------------------------------------------+---------------+
| `SemEval-2018 Task 1: Affect in tweets <https://www.aclweb.org/anthology/S18-1001/>`_ [Task-1]_                     | www           |
+---------------------------------------------------------------------------------------------------------------------+---------------+


