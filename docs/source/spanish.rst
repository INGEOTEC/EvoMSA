.. _spanish:

Spanish
===========================

The first text model created is based on tweets collected from December
2015 until February 2020. From this collection we randomly selected
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
tokens is 15,227 (i.e., :math:`m_b: \text{text} \rightarrow \mathbb
R^{15227}`).


+-----------------------------------------------------------------------------------------------------------------------------------+---------------+
| Source                                                                                                                            | Model         |
+===================================================================================================================================+===============+
| `Overview of TASS 2017 <http://ceur-ws.org/Vol-1896/p0_overview_tass2017.pdf>`_ [TASS2017]_                                       |               |
+-----------------------------------------------------------------------------------------------------------------------------------+---------------+
| `Overview of TASS 2018: Opinions, health and emotions <http://ceur-ws.org/Vol-2172/p0_overview_tass2018.pdf>`_ [TASS2018]_        |               |
+-----------------------------------------------------------------------------------------------------------------------------------+---------------+
| `SemEval-2018 Task 1: Affect in tweets <https://www.aclweb.org/anthology/S18-1001/>`_ [Task-1]_                                   | www           |
+-----------------------------------------------------------------------------------------------------------------------------------+---------------+
| `Overview of MEX-A3T at IberEval 2018 <http://ceur-ws.org/Vol-2150/overview-mex-a3t.pdf>`_ [MEX-A3T]_                             |               |
+-----------------------------------------------------------------------------------------------------------------------------------+---------------+
| `Overview of the HAHA Task <http://ceur-ws.org/Vol-2150/overview-HAHA.pdf>`_ [HAHA]_                                              |               |
+-----------------------------------------------------------------------------------------------------------------------------------+---------------+
