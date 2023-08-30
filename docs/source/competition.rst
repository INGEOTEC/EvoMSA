.. _competition:

====================================
Text Classifier Competitions
====================================
.. image:: https://github.com/INGEOTEC/EvoMSA/actions/workflows/test.yaml/badge.svg
		:target: https://github.com/INGEOTEC/EvoMSA/actions/workflows/test.yaml

.. image:: https://coveralls.io/repos/github/INGEOTEC/EvoMSA/badge.svg?branch=develop
		:target: https://coveralls.io/github/INGEOTEC/EvoMSA?branch=develop

.. image:: https://badge.fury.io/py/EvoMSA.svg
		:target: https://badge.fury.io/py/EvoMSA

.. image:: https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/evomsa-feedstock?branchName=main
	    :target: https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=16466&branchName=main

.. image:: https://img.shields.io/conda/vn/conda-forge/evomsa.svg
		:target: https://anaconda.org/conda-forge/evomsa

.. image:: https://img.shields.io/conda/pn/conda-forge/evomsa.svg
		:target: https://anaconda.org/conda-forge/evomsa

.. image:: https://readthedocs.org/projects/evomsa/badge/?version=docs
		:target: https://evomsa.readthedocs.io/en/docs/?badge=docs

Text classification (TC) is a Natural Language Processing (NLP) task focused on identifying a text's label. A standard approach to tackle text classification problems is to pose it as a supervised learning problem. In supervised learning, everything starts with a dataset composed of pairs of inputs and outputs; in this case, the inputs are texts, and the outputs correspond to the associated labels. The aim is that the developed algorithm can automatically assign a label to any given text independently, whether it was in the original dataset. The feasible classes are only those found on the original dataset. In some circumstances, the method can also inform the confidence it has in its prediction so the user can decide whether to use or discard it.

Following a supervised learning approach requires that the input is in amenable representation for the learning algorithm; usually, this could be a vector. One of the most common methods to represent a text into a vector is to use a Bag of Word (:ref:`bow`) model, which works by having a fixed vocabulary where each component represents an element in the vocabulary and the presence of it in the text is given by a non-zero value.

The text classifier's performance depends on the representation quality and the classifier used. Deciding which representation and algorithm to use is daunting; in this contribution, we describe a set of classifiers that can be used, out of the box, for a new text classification problem. These classifiers are based on the :ref:`BoW` model. Nonetheless, some methods, namely :ref:`DenseBoW`, represent the text following two stages. The first one uses a set of BoW models and classifiers trained on self-supervised problems, where each task predicts the presence of a particular token. Consequently, the text is presented in a vector where each component is associated with a token, and the existence of it is encoded in the value. The methods used BoW models, and DenseBoW were combined using a stack generalization approach, namely :ref:`StackGeneralization`. 

The text classifiers presented have been tested in many text classifier competitions without modifications. The aim is to offer a better understanding of how these algorithms perform in a new situation and what would be the difference in performance with an algorithm tailored to the new problem. We test 13 different algorithms for each task of each competition. The configuration having the best performance was submitted to the contest. The best performance was computed using either a k-fold cross-validation or a validation set, depending on the information provided by the challenge.

Results
------------------------------

Following an unconventional approach, the performance of :ref:`v2` in different competitions is presented before describing the parameters used and the challenges. The following table presents the performance; it includes the performance of the system that wins the competition, the performance of :ref:`v2`, and the difference between them in percentage. 

.. list-table:: Competitions 
    :header-rows: 1

    * - Competitions
      - Edition
      - Winner 
      - :ref:`v2` 
      - Difference
    * - :ref:`HaSpeeDe3 (textual) <haspeede>`
      - 2023 
      - 0.9128
      - 0.8845 (:ref:`Conf. <func_stack_2_bow_tailored_keywords>`)
      - 3.2%
    * - :ref:`HaSpeeDe3 (XReligiousHate) <haspeede>`
      - 2023 
      - 0.6525
      - 0.5522 (:ref:`Conf. <func_stack_2_bow_tailored_keywords>`)
      - 18.2% 
    * - :ref:`HODI <hodi>` 
      - 2023
      - 0.81079 
      - 0.71527 (:ref:`Conf. <func_stack_3_bows_tailored_keywords>`)
      - 13.4%
    * - :ref:`ACTI <acti>`
      - 2023
      - 0.85712
      - 0.78207 (:ref:`Conf. <func_stack_3_bows_tailored_keywords>`)
      - 9.6%
    * - :ref:`PoliticIT (Global) <politicit>`
      - 2023
      - 0.824057
      - 0.762001
      - 8.1%
    * - :ref:`PoliticIT (Gender) <politicit>`
      - 2023
      - 0.824287
      - 0.732259 (:ref:`Conf. <func_stack_3_bows>`)
      - 12.6%
    * - :ref:`PoliticIT (Ideology Binary) <politicit>`
      - 2023
      - 0.928223
      - 0.848525 (:ref:`Conf. <func_bow_training_set>`)
      - 9.4%
    * - :ref:`PoliticIT (Ideology Multiclass) <politicit>`
      - 2023
      - 0.751477
      - 0.705220 (:ref:`Conf. <func_stack_3_bows>`)
      - 6.6%
    * - :ref:`PoliticEs (Global) <politicses>`
      - 2023
      - 0.811319
      - 0.777584
      - 4.3%
    * - :ref:`PoliticEs (Gender) <politicses>`
      - 2023
      - 0.829633
      - 0.711549 (:ref:`Conf. <func_stack_3_bows>`)
      - 16.6%
    * - :ref:`PoliticEs (Profession) <politicses>`
      - 2023
      - 0.860824
      - 0.837945 (:ref:`Conf. <func_stack_3_bows>`)
      - 2.7%
    * - :ref:`PoliticEs (Ideology Binary) <politicses>`
      - 2023
      - 0.896715
      - 0.891394 (:ref:`Conf. <func_stack_3_bows>`)
      - 0.6%
    * - :ref:`PoliticEs (Ideology Multiclass) <politicses>`
      - 2023
      - 0.691334
      - 0.669448 (:ref:`Conf. <func_stack_3_bows>`)
      - 3.3%
    * - :ref:`DAVINCIS <davincis>`
      - 2023
      - 0.9264
      - 0.8903 (:ref:`Conf. <func_stack_2_bow_tailored_all_keywords>`)
      - 4.1%
    * - :ref:`DAVINCIS <davincis-2022>`
      - 2022
      - 0.7817
      - 0.7510 (:ref:`Conf. <func_stack_2_bow_all_keywords>`)
      - 4.1%
    * - :ref:`REST-MEX (Global) <rest-mex>`
      - 2023
      - 0.7790190145
      - 0.7375714730
      - 5.6%
    * - :ref:`REST-MEX (Polarity) <rest-mex>`
      - 2023
      - 0.621691991
      - 0.554880778 (:ref:`Conf. <func_stack_bows>`)
      - 12.0%
    * - :ref:`REST-MEX (Type) <rest-mex>`
      - 2023
      - 0.99032231
      - 0.980539122 (:ref:`Conf. <func_bow_training_set>`)
      - 1.0%
    * - :ref:`REST-MEX (Country) <rest-mex>`
      - 2023
      - 0.942028113
      - 0.927052594 (:ref:`Conf. <func_bow_training_set>`)
      - 1.6%    
    * - :ref:`HOMO-MEX <homo-mex>`
      - 2023
      - 0.8847
      - 0.8050 (:ref:`Conf. <func_stack_3_bow_tailored_all_keywords>`)
      - 9.9%
    * - :ref:`HOPE (ES) <hope>`
      - 2023
      - 0.9161
      - 0.4198 (:ref:`Conf. <func_stack_bow_keywords_emojis_voc_selection>`)
      - 118.2%
    * - :ref:`HOPE (EN) <hope>`
      - 2023
      - 0.5012
      - 0.4429 (:ref:`Conf. <func_stack_bow_keywords_emojis>`)
      - 13.2%
    * - :ref:`DIPROMATS (ES) <dipromats>`
      - 2023
      - 0.8089
      - 0.7485 (:ref:`Conf. <func_stack_3_bows>`)
      - 8.1%
    * - :ref:`DIPROMATS (EN) <dipromats>`
      - 2023
      - 0.8090
      - 0.7255 (:ref:`Conf. <func_stack_3_bow_tailored_all_keywords>`)
      - 11.5%
    * - :ref:`HUHU <huhu>`
      - 2023
      - 0.820
      - 0.775 (:ref:`Conf. <func_stack_2_bow_all_keywords>`)
      - 5.8%
    * - :ref:`TASS <tass>`
      - 2017
      - 0.577  
      - 0.525 (:ref:`Conf. <func_stack_2_bow_tailored_all_keywords>`)
      - 9.9%
    * - :ref:`EDOS (A) <edos>`
      - 2023
      - 0.8746
      - 0.7890 (:ref:`Conf. <func_stack_2_bow_keywords>`)
      - 10.8%
    * - :ref:`EDOS (B) <edos>`
      - 2023
      - 0.7326
      - 0.5413 (:ref:`Conf. <func_stack_bow_keywords_emojis>`)
      - 35.3%
    * - :ref:`EDOS (C) <edos>`
      - 2023
      - 0.5606
      - 0.3500 (:ref:`Conf. <func_stack_2_bow_all_keywords>`)
      - 60.2%

Competitions
------------------------------
.. _haspeede:

`Hate Speech Detection (HaSpeeDe3) <http://www.di.unito.it/~tutreeb/haspeede-evalita23/index.html>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.8778
      - 1.0000
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.8769
      - 0.2720
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.8756
      - 0.0500
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.8754
      - 0.1380
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.8751
      - 0.1600
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.8745
      - 0.1020
    * - :ref:`bow <func_bow>`
      - 0.8740
      - 0.0780
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.8727
      - 0.0080
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.8723
      - 0.0260
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.8688
      - 0.0000
.. _hodi:

`Homotransphobia Detection in Italian (HODI) <https://hodi-evalita.github.io>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.7883
      - 1.0000
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.7875
      - 0.3900
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.7864
      - 0.2360
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.7862
      - 0.2640
    * - :ref:`bow <func_bow>`
      - 0.7842
      - 0.1100
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.7838
      - 0.0060
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.7833
      - 0.0620
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.7830
      - 0.0000
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.7765
      - 0.0000
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.7758
      - 0.0000

.. _acti:

`Automatic Conspiracy Theory Identification (ACTI) <https://russogiuseppe.github.io/ACTI>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.7914
      - 1.0000
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.7872
      - 0.1180
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.7855
      - 0.1700
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.7855
      - 0.0740
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.7804
      - 0.0240
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.7785
      - 0.0260
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.7780
      - 0.0200
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.7765
      - 0.0040
    * - :ref:`bow <func_bow>`
      - 0.7758
      - 0.0040
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.7755
      - 0.0000


.. _politicit:

`Political Ideology Detection in Italian Texts (PoliticIT) <https://codalab.lisn.upsaclay.fr/competitions/8507>`_ 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation (Gender)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.9792
      - 1.0000
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.9583
      - 0.2120
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.9583
      - 0.2340
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.9375
      - 0.1260
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.8748
      - 0.0200
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.8748
      - 0.0200
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.8536
      - 0.0160
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.8536
      - 0.0160
    * - :ref:`bow <func_bow>`
      - 0.8307
      - 0.0000
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.8307
      - 0.0000


.. list-table:: Performance in Cross-validation (Ideology Binary)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 1.0000
      - 1.0000
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.9714
      - 0.1580
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.9714
      - 0.1580
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.8712
      - 0.0200
    * - :ref:`bow <func_bow>`
      - 0.8487
      - 0.0120
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.8487
      - 0.0120
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.8271
      - 0.0060
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.8271
      - 0.0060
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.7856
      - 0.0040
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.7856
      - 0.0040


.. list-table:: Performance in Cross-validation (Ideology Multiclass)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.9834
      - 1.0000
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.9834
      - 1.0000
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.9823
      - 0.4100
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.7756
      - 0.0020
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.7271
      - 0.0000
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.7271
      - 0.0000
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.7111
      - 0.0000
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.7111
      - 0.0000
    * - :ref:`bow <func_bow>`
      - 0.5308
      - 0.0000
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.5308
      - 0.0000


.. _politicses:

`Political Ideology Detection in Spanish Texts (PoliticEs) <https://codalab.lisn.upsaclay.fr/competitions/10173>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation (Gender)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 1.0000
      - 1.0000
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.9764
      - 0.1080
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.9643
      - 0.0660
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.9643
      - 0.0660
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.9406
      - 0.0200
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.9406
      - 0.0200
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.9406
      - 0.0200
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.9406
      - 0.0200
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.9406
      - 0.0200
    * - :ref:`bow <func_bow>`
      - 0.9398
      - 0.0320
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.9398
      - 0.0320
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.9291
      - 0.0180
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.9291
      - 0.0180



.. list-table:: Performance in Cross-validation (Profession)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 1.0000
      - 1.0000
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 1.0000
      - 1.0000
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 1.0000
      - 1.0000
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 1.0000
      - 1.0000
    * - :ref:`bow <func_bow>`
      - 0.9756
      - 0.0680
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.9756
      - 0.0680
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.9597
      - 0.1920
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.9352
      - 0.1000
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.9352
      - 0.1000
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.9105
      - 0.0920
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.9105
      - 0.0920
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.9022
      - 0.0880
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.9022
      - 0.0880


.. list-table:: Performance in Cross-validation (Ideology Binary)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 1.0000
      - 1.0000
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 1.0000
      - 1.0000
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 1.0000
      - 1.0000
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 1.0000
      - 1.0000
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.9657
      - 0.0740
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.9657
      - 0.0760
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.9657
      - 0.0760
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.9657
      - 0.0760
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.9657
      - 0.0760
    * - :ref:`bow <func_bow>`
      - 0.9545
      - 0.0420
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.9545
      - 0.0420
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.9545
      - 0.0420
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.9541
      - 0.0620


.. list-table:: Performance in Cross-validation (Ideology Multiclass)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 1.0000
      - 1.0000
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 1.0000
      - 1.0000
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.9889
      - 0.1780
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.9889
      - 0.1780
    * - :ref:`bow <func_bow>`
      - 0.9644
      - 0.0400
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.9644
      - 0.0400
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.9369
      - 0.0160
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.9225
      - 0.0000
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.9225
      - 0.0000
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.9121
      - 0.0040
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.9121
      - 0.0040
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.8475
      - 0.0000
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.8467
      - 0.0000


.. _davincis:

`Detection of Aggressive and Violent Incidents from Social Media in Spanish (DAVINCIS) <https://sites.google.com/view/davincis-iberlef-2023>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.8984
      - 1.0000
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.8971
      - 0.2260
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.8968
      - 0.2120
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.8966
      - 0.1580
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.8955
      - 0.0540
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.8951
      - 0.0440
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.8931
      - 0.0760
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.8907
      - 0.0260
    * - :ref:`bow <func_bow>`
      - 0.8894
      - 0.0180
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.8892
      - 0.0060
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.8879
      - 0.0020
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.8863
      - 0.0000
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.8859
      - 0.0000

.. _davincis-2022:

`Detection of Aggressive and Violent Incidents from Social Media in Spanish (DAVINCIS 2022) <https://sites.google.com/view/davincis-iberlef/home>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.8447
      - 1.0000
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.8361
      - 0.0000
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.8219
      - 0.0000
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.7595
      - 0.0000
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.7588
      - 0.0000
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.7572
      - 0.0000
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.7555
      - 0.0000
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.7525
      - 0.0000
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.7342
      - 0.0000
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.7337
      - 0.0000
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.7329
      - 0.0000
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.7329
      - 0.0000
    * - :ref:`bow <func_bow>`
      - 0.7324
      - 0.0000


.. _rest-mex:

`Research on Sentiment Analysis Task for Mexican Tourist Texts (REST-MEX) <https://sites.google.com/cimat.mx/rest-mex2023>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation (Polarity)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.5605
      - 1.0000
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.5603
      - 0.4140
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.5472
      - 0.0000
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.5467
      - 0.0000
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.5448
      - 0.0000
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.5446
      - 0.0000
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.5431
      - 0.0000
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.5420
      - 0.0000
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.5346
      - 0.0000
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.5310
      - 0.0000
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.5179
      - 0.0000
    * - :ref:`bow <func_bow>`
      - 0.5167
      - 0.0000
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.5152
      - 0.0000

.. list-table:: Performance in Cross-validation (Type)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.9802
      - 1.0000
    * - :ref:`bow <func_bow>`
      - 0.9793
      - 0.0040
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.9793
      - 0.0000
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.9792
      - 0.0000
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.9783
      - 0.0000
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.9783
      - 0.0000
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.9782
      - 0.0000
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.9773
      - 0.0000
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.9773
      - 0.0000
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.9769
      - 0.0000
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.9768
      - 0.0000
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.9743
      - 0.0000
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.9742
      - 0.0000


.. list-table:: Performance in Cross-validation (Country)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.9260
      - 1.0000
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.9225
      - 0.0000
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.9200
      - 0.0000
    * - :ref:`bow <func_bow>`
      - 0.9194
      - 0.0000
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.9167
      - 0.0000
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.9166
      - 0.0000
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.9164
      - 0.0000
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.9101
      - 0.0000
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.9097
      - 0.0000
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.9076
      - 0.0000
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.9076
      - 0.0000
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.8951
      - 0.0000
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.8949
      - 0.0000


.. _homo-mex:

`Hate Speech Detection towards the Mexican Spanish Speaking LGBT+ Population (HOMO-MEX) <https://codalab.lisn.upsaclay.fr/competitions/10019>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.7914
      - 1.0000
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.7912
      - 0.4460
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.7908
      - 0.3420
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.7904
      - 0.2980
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.7903
      - 0.2700
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.7901
      - 0.0740
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.7885
      - 0.1300
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.7880
      - 0.1460
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.7871
      - 0.0660
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.7861
      - 0.0160
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.7689
      - 0.0000
    * - :ref:`bow <func_bow>`
      - 0.7669
      - 0.0000
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.7553
      - 0.0000


.. _hope:

`Multilingual Hope Speech Detection (HOPE) <https://codalab.lisn.upsaclay.fr/competitions/10215>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation (Spanish)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.8224
      - 1.0000
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.8217
      - 0.3580
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.8192
      - 0.3680
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.8192
      - 0.3040
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.8192
      - 0.3680
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.8159
      - 0.1740
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.8152
      - 0.1500
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.8119
      - 0.1020
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.7998
      - 0.0460
    * - :ref:`bow <func_bow>`
      - 0.7966
      - 0.0260
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.7966
      - 0.0260
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.7902
      - 0.0040
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.7807
      - 0.0000


.. list-table:: Performance in Cross-validation (English)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.7807
      - 1.0000
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.7801
      - 0.4600
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.7753
      - 0.2860
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.7691
      - 0.0300
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.7690
      - 0.0260
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.7680
      - 0.0220
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.7662
      - 0.0120
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.7661
      - 0.0200
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.7640
      - 0.0120
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.7575
      - 0.0020
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.7379
      - 0.0020
    * - :ref:`bow <func_bow>`
      - 0.7300
      - 0.0000
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.7221
      - 0.0000


.. _dipromats:

`Automatic Detection and Characterization of Propaganda Techniques from Diplomats (DIPROMATS) <https://sites.google.com/view/dipromats2023>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation (Spanish)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.6551
      - 1.0000
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.6544
      - 0.4180
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.6515
      - 0.2200
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.6514
      - 0.2500
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.6488
      - 0.1120
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.6486
      - 0.1360
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.6486
      - 0.1360
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.6486
      - 0.1520
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.6485
      - 0.1480
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.6484
      - 0.1180
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.6290
      - 0.0080
    * - :ref:`bow <func_bow>`
      - 0.6136
      - 0.0000
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.6123
      - 0.0000


.. list-table:: Performance in Cross-validation (English)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.6498
      - 1.0000
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.6489
      - 0.2260
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.6471
      - 0.1280
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.6448
      - 0.0440
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.6446
      - 0.0140
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.6443
      - 0.0240
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.6386
      - 0.0080
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.6381
      - 0.0000
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.6377
      - 0.0040
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.6327
      - 0.0000
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.6043
      - 0.0000
    * - :ref:`bow <func_bow>`
      - 0.5961
      - 0.0000
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.5922
      - 0.0000



.. _huhu:

`HUrtful HUmour (HUHU) <https://sites.google.com/view/huhuatiberlef23>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation 
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.7417
      - 1.0000
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.7416
      - 0.4700
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.7379
      - 0.1700
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.7377
      - 0.1340
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.7358
      - 0.1000
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.7347
      - 0.0180
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.7334
      - 0.0620
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.7298
      - 0.0100
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.7127
      - 0.0000
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.7103
      - 0.0000
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.7034
      - 0.0000
    * - :ref:`bow <func_bow>`
      - 0.6969
      - 0.0000
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.6833
      - 0.0000

.. _tass:

`Workshop on Sentiment Analysis at SEPLN (TASS) <https://ceur-ws.org/Vol-1896/p0_overview_tass2017.pdf>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation (A)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.5461
      - 1.0000
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.5452
      - 0.2640
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.5431
      - 0.2280
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.5428
      - 0.0720
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.5418
      - 0.0200
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.5417
      - 0.0220
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.5405
      - 0.0080
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.5386
      - 0.0140
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.5186
      - 0.0000
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.5078
      - 0.0000
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.4950
      - 0.0000
    * - :ref:`bow <func_bow>`
      - 0.4933
      - 0.0000
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.4919
      - 0.0000


.. _edos:

`Explainable Detection of Online Sexism (EDOS) <https://arxiv.org/pdf/2303.04222.pdf>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation (A)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.7622
      - 1.0000
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.7580
      - 0.2220
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.7567
      - 0.0960
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.7532
      - 0.1100
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.7517
      - 0.0720
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.7503
      - 0.0600
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.7502
      - 0.0280
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.7487
      - 0.0300
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.7486
      - 0.0540
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.7478
      - 0.0100
    * - :ref:`bow <func_bow>`
      - 0.7398
      - 0.0060
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.7354
      - 0.0020
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.7350
      - 0.0000

.. list-table:: Performance in Cross-validation (B)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.5247
      - 1.0000
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.5123
      - 0.1580
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.5088
      - 0.1540
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.5064
      - 0.1040
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.5002
      - 0.1440
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.4969
      - 0.1000
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.4950
      - 0.0960
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.4929
      - 0.0760
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.4924
      - 0.0080
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.4909
      - 0.1000
    * - :ref:`bow <func_bow>`
      - 0.4597
      - 0.0340
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.4450
      - 0.0140
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.4427
      - 0.0140

.. list-table:: Performance in Cross-validation (C)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :ref:`stack_2_bow_all_keywords <func_stack_2_bow_all_keywords>`
      - 0.3236
      - 1.0000
    * - :ref:`stack_2_bow_tailored_all_keywords <func_stack_2_bow_tailored_all_keywords>`
      - 0.3145
      - 0.0980
    * - :ref:`stack_bow_keywords_emojis <func_stack_bow_keywords_emojis>`
      - 0.3123
      - 0.2760
    * - :ref:`stack_2_bow_tailored_keywords <func_stack_2_bow_tailored_keywords>`
      - 0.3069
      - 0.1460
    * - :ref:`stack_3_bow_tailored_all_keywords <func_stack_3_bow_tailored_all_keywords>`
      - 0.3035
      - 0.0020
    * - :ref:`stack_bow_keywords_emojis_voc_selection <func_stack_bow_keywords_emojis_voc_selection>`
      - 0.2943
      - 0.0580
    * - :ref:`stack_3_bows_tailored_keywords <func_stack_3_bows_tailored_keywords>`
      - 0.2924
      - 0.0240
    * - :ref:`stack_2_bow_keywords <func_stack_2_bow_keywords>`
      - 0.2870
      - 0.0120
    * - :ref:`bow_voc_selection <func_bow_voc_selection>`
      - 0.2700
      - 0.0140
    * - :ref:`bow <func_bow>`
      - 0.2685
      - 0.0140
    * - :ref:`stack_3_bows <func_stack_3_bows>`
      - 0.2556
      - 0.0000
    * - :ref:`bow_training_set <func_bow_training_set>`
      - 0.2530
      - 0.0080
    * - :ref:`stack_bows <func_stack_bows>`
      - 0.2486
      - 0.0000


Systems
-----------------------------------------------

We test 13 different combinations of :ref:`BoW` and :ref:`DenseBoW` models. These models include the use of the two procedures to select the vocabulary (parameter voc_selection), the use of pre-trained :ref:`BoW`, and the creation of the :ref:`BoW` representation with the given training set. Additionally, we create text representations tailored to the problem at hand. That is the words with more discriminant power in a :ref:`BoW` classifier, trained on the training set, are selected as the labels in self-supervised problems. 

.. code-block:: python

    from EvoMSA import BoW, DenseBoW, StackGeneralization
    from EvoMSA.utils import Linear, b4msa_params
    from text_models.dataset import SelfSupervisedDataset    
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

.. _func_bow:

:ref:`BoW` default parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pre-trained :ref:`BoW` where the tokens are selected based on a normalized frequency w.r.t. its type, i.e., bigrams, words, and q-grams of characters.

.. code-block:: python

    def bow(lang, tr, vs, **kwargs):
        bow = BoW(lang=lang).fit(tr)
        hy = bow.predict(vs)
        return hy

.. _func_bow_voc_selection:

:ref:`BoW` using :py:attr:`voc_selection` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pre-trained :ref:`BoW` where the tokens correspond to the most frequent ones.

.. code-block:: python

    def bow_voc_selection(lang, tr, vs, **kwargs):
        bow = BoW(lang=lang, voc_selection='most_common').fit(tr)
        hy = bow.predict(vs)
        return hy

.. _func_bow_training_set:

:ref:`BoW` trained on the training set 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`BoW` trained with the training set; the number of tokens corresponds to all the tokens in the set. 

.. code-block:: python

    def bow_training_set(lang, tr, vs, **kwargs):
        params = b4msa_params(lang=lang)
        del params['token_max_filter']
        del params['max_dimension']
        bow_no_pre = BoW(lang=lang, pretrain=False, b4msa_kwargs=params).fit(tr)
        return bow_no_pre.predict(vs)

.. _func_stack_bow_keywords_emojis:

:ref:`StackGeneralization` with :ref:`BoW` and :ref:`DenseBoW` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stack generalization approach where the base classifiers are the :ref:`BoW`, the emojis, and the keywords dense BoW (i.e., :ref:`densebow`).

.. code-block:: python

    def stack_bow_keywords_emojis(lang, tr, vs, **kwargs):
        bow = BoW(lang=lang)
        keywords = DenseBoW(lang=lang, 
                            emoji=False, 
                            dataset=False).select(D=tr)
        emojis = DenseBoW(lang=lang, 
                          keyword=False, 
                          dataset=False).select(D=tr)
        stack = StackGeneralization(decision_function_models=[bow, 
                                                              keywords, 
                                                              emojis]).fit(tr)
        X = bow.transform(vs)
        for x in [bow, keywords, emojis]:
            x.cache = X    
        return stack.predict(vs)

.. _func_stack_bow_keywords_emojis_voc_selection:

:ref:`StackGeneralization` with :ref:`BoW` and :ref:`DenseBoW` using :py:attr:`voc_selection` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stack generalization approach where the base classifiers are the :ref:`BoW`,  the emojis, and the keywords dense BoW (i.e., :ref:`densebow`). The tokens in these models were selected based on a normalized frequency w.r.t. its type, i.e., bigrams, words, and q-grams of characters.

.. code-block:: python

    def stack_bow_keywords_emojis_voc_selection(lang, tr, vs, **kwargs):
        bow = BoW(lang=lang, voc_selection='most_common')
        keywords = DenseBoW(lang=lang, voc_selection='most_common',
                            emoji=False, 
                            dataset=False).select(D=tr)
        emojis = DenseBoW(lang=lang, voc_selection='most_common',
                          keyword=False, 
                          dataset=False).select(D=tr)
        stack = StackGeneralization(decision_function_models=[bow, 
                                                              keywords, 
                                                              emojis]).fit(tr)
        X = bow.transform(vs)
        for x in [bow, keywords, emojis]:
            x.cache = X    
        return stack.predict(vs)

.. _func_stack_bows:

:ref:`StackGeneralization` with two :ref:`BoW` models 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stack generalization approach where the base classifiers are BoW with the two token selection procedures described previously (i.e., :ref:`func_bow` and :ref:`func_bow_voc_selection`).

.. code-block:: python

    def stack_bows(lang, tr, vs, **kwargs):
        bow = BoW(lang=lang)
        bow2 = BoW(lang=lang, voc_selection='most_common')
        stack = StackGeneralization(decision_function_models=[bow, bow2]).fit(tr)
        return stack.predict(vs)

.. _func_stack_2_bow_keywords:

:ref:`StackGeneralization` using :ref:`BoW` and :ref:`DenseBoW` with and without :py:attr:`voc_selection` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^		

Stack generalization approach where with four base classifiers. These correspond to two :ref:`BoW` and two :ref:`dense BoW <densebow>` (emojis and keywords), where the difference in each is the procedure used to select the tokens, i.e., the most frequent or normalized frequency. 

.. code-block:: python

    def stack_2_bow_keywords(lang, tr, vs, **kwargs):
        bow = BoW(lang=lang)      
        keywords = DenseBoW(lang=lang, dataset=False).select(D=tr)
        bow2 = BoW(lang=lang, voc_selection='most_common')
        keywords2 = DenseBoW(lang=lang, 
                             voc_selection='most_common',
                             dataset=False).select(D=tr)
        stack = StackGeneralization(decision_function_models=[bow, bow2,
                                                              keywords,
                                                              keywords2]).fit(tr)
        X = bow.transform(vs)
        for x in [bow, keywords]:
            x.cache = X
        X = bow2.transform(vs)
        for x in [bow2, keywords2]:
            x.cache = X
        return stack.predict(vs)

.. _func_stack_2_bow_tailored_keywords:

:ref:`StackGeneralization` using :ref:`BoW` and tailored :ref:`DenseBoW` with and without :py:attr:`voc_selection` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stack generalization approach where with four base classifiers. These correspond to two :ref:`BoW` and two :ref:`dense BoW <densebow>` (emojis and keywords), where the difference in each is the procedure used to select the tokens, i.e., the most frequent or normalized frequency.  The second difference is that the dense representation with normalized frequency also includes models for the most discriminant words selected by a :ref:`BoW` classifier in the training set. We refer to these latter representations as :ref:`tailored keywords <tailored-keywords>`.

.. code-block:: python

    def stack_2_bow_tailored_keywords(lang, tr, vs, keywords=None, **kwargs):
        models = [Linear(**kwargs)
                  for kwargs in tweet_iterator(keywords)]    
        bow = BoW(lang=lang)      
        keywords = DenseBoW(lang=lang, dataset=False)
        keywords.text_representations_extend(models)
        keywords.select(D=tr)
        bow2 = BoW(lang=lang, voc_selection='most_common')
        keywords2 = DenseBoW(lang=lang, 
                             voc_selection='most_common',
                             dataset=False).select(D=tr)
        stack = StackGeneralization(decision_function_models=[bow, bow2,
                                                              keywords,
                                                              keywords2]).fit(tr)
        X = bow.transform(vs)
        for x in [bow, keywords]:
            x.cache = X
        X = bow2.transform(vs)
        for x in [bow2, keywords2]:
            x.cache = X
        return stack.predict(vs)

.. _func_stack_2_bow_all_keywords:

:ref:`StackGeneralization` using :ref:`BoW` and all :ref:`DenseBoW` with and without :py:attr:`voc_selection` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stack generalization approach where with four base classifiers equivalently to :ref:`func_stack_2_bow_keywords` where the difference is that the dense representations include the models created with the :ref:`human-annotated datasets <dense_parameters>`.

.. code-block:: python

    def stack_2_bow_all_keywords(lang, tr, vs, **kwargs):
        bow = BoW(lang=lang)      
        keywords = DenseBoW(lang=lang)
        sel = [k for k, v in enumerate(keywords.names)
               if v not in ['davincis2022_1'] or 'semeval2023' not in v]
        keywords.select(sel).select(D=tr)
        bow2 = BoW(lang=lang, voc_selection='most_common')
        keywords2 = DenseBoW(lang=lang,
                             voc_selection='most_common').select(sel).select(D=tr)
        stack = StackGeneralization(decision_function_models=[bow, 
                                                              bow2, 
                                                              keywords, 
                                                              keywords2]).fit(tr)
        X = bow.transform(vs)
        for x in [bow, keywords]:
            x.cache = X
        X = bow2.transform(vs)
        for x in [bow2, keywords2]:
            x.cache = X
        return stack.predict(vs)

.. _func_stack_2_bow_tailored_all_keywords:

:ref:`StackGeneralization` using :ref:`BoW` tailored and datasets :ref:`DenseBoW` with and without :py:attr:`voc_selection` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is a stack generalization approach with four base classifiers equivalent to :ref:`func_stack_2_bow_all_keywords`, where the difference is that the dense representation with normalized frequency also includes the tailored keywords.

.. code-block:: python

    def stack_2_bow_tailored_all_keywords(lang, tr, vs, keywords=None, **kwargs):
        models = [Linear(**kwargs)
                  for kwargs in tweet_iterator(keywords)]    
        bow = BoW(lang=lang)      
        keywords = DenseBoW(lang=lang)
        sel = [k for k, v in enumerate(keywords.names)
               if v not in ['davincis2022_1'] or 'semeval2023' not in v]
        keywords.select(sel)
        keywords.text_representations_extend(models)
        keywords.select(D=tr)
        bow2 = BoW(lang=lang, voc_selection='most_common')
        keywords2 = DenseBoW(lang=lang,
                             voc_selection='most_common').select(sel).select(D=tr)
        stack = StackGeneralization(decision_function_models=[bow, 
                                                              bow2, 
                                                              keywords, 
                                                              keywords2]).fit(tr)
        X = bow.transform(vs)
        for x in [bow, keywords]:
            x.cache = X
        X = bow2.transform(vs)
        for x in [bow2, keywords2]:
            x.cache = X
        return stack.predict(vs)

.. _func_stack_3_bows:

:ref:`StackGeneralization` with three :ref:`BoW` models 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^	

Stack generalization approach with three base classifiers. All of them are :ref:`BoW`; the first two correspond pre-trained BoW with the two token selection procedures described previously (i.e., :ref:`func_bow` and :ref:`func_bow_voc_selection`), and the latest is a :ref:`func_bow_training_set`.

.. code-block:: python

    def stack_3_bows(lang, tr, vs, **kwargs):
        params = b4msa_params(lang=lang)
        del params['token_max_filter']
        del params['max_dimension']
        bow_no_pre = BoW(lang=lang, pretrain=False, b4msa_kwargs=params)
        bow = BoW(lang=lang)
        bow2 = BoW(lang=lang, voc_selection='most_common')
        stack = StackGeneralization(decision_function_models=[bow_no_pre, 
                                                              bow, 
                                                              bow2]).fit(tr)
        return stack.predict(vs)

.. _func_stack_3_bows_tailored_keywords:

:ref:`StackGeneralization` using :ref:`BoW` and all :ref:`DenseBoW` with and without :py:attr:`voc_selection` plus :ref:`BoW` trained on the training set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stack generalization approach with five base classifiers. The first corresponds to a :ref:`func_bow_training_set`, and the rest are used in :ref:`func_stack_2_bow_tailored_keywords`.

.. code-block:: python

    def stack_3_bows_tailored_keywords(lang, tr, vs, keywords=None, **kwargs):
        models = [Linear(**kwargs)
                  for kwargs in tweet_iterator(keywords)]
        params = b4msa_params(lang=lang)
        del params['token_max_filter']
        del params['max_dimension']
        bow_no_pre = BoW(lang=lang, pretrain=False, b4msa_kwargs=params)
        bow = BoW(lang=lang)      
        keywords = DenseBoW(lang=lang, dataset=False)
        keywords.text_representations_extend(models)
        keywords.select(D=tr)
        bow2 = BoW(lang=lang, voc_selection='most_common')
        keywords2 = DenseBoW(lang=lang, 
                             voc_selection='most_common',
                             dataset=False).select(D=tr)
        stack = StackGeneralization(decision_function_models=[bow_no_pre,
                                                              bow, 
                                                              bow2, 													
                                                              keywords,
                                                              keywords2]).fit(tr)
        X = bow.transform(vs)
        for x in [bow, keywords]:
            x.cache = X
        X = bow2.transform(vs)
        for x in [bow2, keywords2]:
            x.cache = X
        return stack.predict(vs)

.. _func_stack_3_bow_tailored_all_keywords:

:ref:`StackGeneralization` using :ref:`BoW` and all :ref:`DenseBoW` with and without :py:attr:`voc_selection` plus :ref:`BoW` trained on the training set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stack generalization approach with five base classifiers. It is comparable to :ref:`func_stack_3_bows_tailored_keywords` being the difference in the use of the :ref:`tailored keywords <tailored-keywords>`.

.. code-block:: python

    def stack_3_bow_tailored_all_keywords(lang, tr, vs, keywords=None, **kwargs):
        params = b4msa_params(lang=lang)
        del params['token_max_filter']
        del params['max_dimension']
        bow_no_pre = BoW(lang=lang, pretrain=False, b4msa_kwargs=params)
        models = [Linear(**kwargs)
                  for kwargs in tweet_iterator(keywords)]    
        bow = BoW(lang=lang)      
        keywords = DenseBoW(lang=lang)
        sel = [k for k, v in enumerate(keywords.names)
               if v not in ['davincis2022_1'] or 'semeval2023' not in v]
        keywords.select(sel)
        keywords.text_representations_extend(models)
        keywords.select(D=tr)
        bow2 = BoW(lang=lang, voc_selection='most_common')
        keywords2 = DenseBoW(lang=lang,
                             voc_selection='most_common').select(sel).select(D=tr)
        stack = StackGeneralization(decision_function_models=[bow_no_pre, 
                                                              bow, 
                                                              bow2,
                                                              keywords, 
                                                              keywords2]).fit(tr)
        X = bow.transform(vs)
        for x in [bow, keywords]:
            x.cache = X
        X = bow2.transform(vs)
        for x in [bow2, keywords2]:
            x.cache = X
        return stack.predict(vs)

.. _tailored-keywords:

Tailored Keywords
-----------------------------

.. code-block:: python

    bow = BoW(lang=LANG, pretrain=False).fit(D)
    keywords = DenseBoW(lang=LANG, emoji=False, dataset=False).names
    tokens = [(name, np.median(np.fabs(w * v)))
              for name, w, v in zip(bow.names, bow.weights, bow.estimator_instance.coef_.T) 
              if name[:2] != 'q:' and '~' not in name and name not in keywords]
    tokens.sort(key=lambda x: x[1], reverse=True)
    semi = SelfSupervisedDataset([k for k, _ in tokens[:2048]],
                                 tempfile=f'{MODEL}.gz',
                                 bow=BoW(lang=LANG), capacity=1, n_jobs=63)
    semi.process(PATH_DATASET, output=MODEL)

Predictions
------------------------------
