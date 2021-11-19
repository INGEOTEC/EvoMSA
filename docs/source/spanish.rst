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

This model (without using the text model trained with the training set) can be used as follow:
	  
>>> from EvoMSA.base import EvoMSA
>>> evo = EvoMSA(TR=False, B4MSA=True, lang='es')

The next table shows the different models we have produced for the
Spanish language.

+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------+
| Source                                                                                                                      | Model                           |
+=============================================================================================================================+=================================+
| `Multilingual Twitter sentiment classification: The role of human annotators`_ [Human-Annotated]_                           | :ref:`ha`                       |
+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------+
| `Overview of TASS 2017 <http://ceur-ws.org/Vol-1896/p0_overview_tass2017.pdf>`_ [TASS2017]_                                 | `tass2016_Es.evomsa`_           |
|                                                                                                                             | `tass2017_Es.evomsa`_           |
+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------+
| `Overview of TASS 2018: Opinions, health and emotions <http://ceur-ws.org/Vol-2172/p0_overview_tass2018.pdf>`_ [TASS2018]_  | `tass2018_s1_l1_Es.evomsa`_     |
|                                                                                                                             | `tass2018_s1_l2_Es.evomsa`_     |
|                                                                                                                             | `tass2018_s2_Es.evomsa`_        |
+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------+
| `SemEval-2018 Task 1: Affect in tweets <https://www.aclweb.org/anthology/S18-1001/>`_ [Task-1]_                             | `semeval2018_anger_Es.evomsa`_  |
|                                                                                                                             | `semeval2018_fear_Es.evomsa`_   |    
|                                                                                                   			      | `semeval2018_joy_Es.evomsa`_    |
|                                                                                                   			      | `semeval2018_sadness_Es.evomsa`_|
|                                                                                                   			      | `semeval2018_valence_Es.evomsa`_|
+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------+
| `Overview of MEX-A3T at IberEval 2018 <http://ceur-ws.org/Vol-2150/overview-mex-a3t.pdf>`_ [MEX-A3T]_                       | `mexa3t2018_aggress_Es.evomsa`_ |
+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------+
| `Overview of the HAHA Task <http://ceur-ws.org/Vol-2150/overview-HAHA.pdf>`_ [HAHA]_                                        | `haha2018_Es.evomsa`_           |
+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------+
| `Overview of the Task on Automatic Misogyny Identification at IberEval 2018`_ [AMI]_                                        | `misoginia_Es.evomsa`_          |
+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------+
| `Estado de ánimo de los tuiteros en México`_                                                                                | `INEGI-MX_Es.evomsa`_           |
+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------+

The following code shows the usage of two of the models with suffix
`.evomsa`. The models used are `haha2018_Es.evomsa`_,
`mexa3t2018_aggress_Es.evomsa`_, as well as :math:`m_b` text-model,
and without the text model obtained with the training set.

>>> from EvoMSA.utils import download
>>> from EvoMSA.base import EvoMSA
>>> from microtc.utils import tweet_iterator
>>> import os
>>> tweets = os.path.join(os.path.dirname(base.__file__), 'tests', 'tweets.json')
>>> D = list(tweet_iterator(tweets))
>>> X = [x['text'] for x in D]
>>> y = [x['klass'] for x in D]
>>> haha = download('haha2018_Es.evomsa')
>>> mexa3t = download('mexa3t2018_aggress_Es.evomsa')
>>> evo = EvoMSA(TR=False, B4MSA=True, lang='es',
                 models=[[haha, "sklearn.svm.LinearSVC"],
                         [mexa3t, "sklearn.svm.LinearSVC"]])
>>> evo.fit(X, y)			 

where :py:class:`sklearn.svm.LinearSVC` can be any classifier following the structure of `sklearn <https://scikit-learn.org/>`_.

Predict a sentence in Spanish

>>> evo.predict(['EvoMSA esta funcionando'])

.. _Multilingual Twitter sentiment classification\: The role of human annotators: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155036
.. _tass2016_Es.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/tass2016_Es.evomsa
.. _tass2017_Es.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/tass2017_Es.evomsa
.. _tass2018_s1_l1_Es.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/tass2018_s1_l1_Es.evomsa
.. _tass2018_s1_l2_Es.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/tass2018_s1_l2_Es.evomsa
.. _tass2018_s2_Es.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/tass2018_s2_Es.evomsa
.. _semeval2018_anger_Es.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_anger_Es.evomsa
.. _semeval2018_fear_Es.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_fear_Es.evomsa
.. _semeval2018_joy_Es.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_joy_Es.evomsa
.. _semeval2018_sadness_Es.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_sadness_Es.evomsa
.. _semeval2018_valence_Es.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/semeval2018_valence_Es.evomsa
.. _mexa3t2018_aggress_Es.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/mexa3t2018_aggress_Es.evomsa
.. _haha2018_Es.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/haha2018_Es.evomsa
.. _Estado de ánimo de los tuiteros en México: https://www.inegi.org.mx/app/animotuitero
.. _INEGI-MX_Es.tm: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/INEGI-MX_Es.tm
.. _Overview of the Task on Automatic Misogyny Identification at IberEval 2018: http://ceur-ws.org/Vol-2150/overview-AMI.pdf
.. _misoginia_Es.evomsa: https://github.com/INGEOTEC/EvoMSA/raw/master/EvoMSA/models/misoginia_Es.evomsa
