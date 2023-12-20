.. _competition:
.. py:currentmodule:: EvoMSA.competitions

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

.. list-table:: :ref:`v2` Performance in different competitions. 
    :header-rows: 1

    * - Competitions
      - Edition
      - Score
      - Winner 
      - :ref:`v2` 
      - Difference
    * - :ref:`HaSpeeDe3 (textual) <haspeede>`
      - 2023 
      - macro-:math:`f_1`
      - 0.9128
      - 0.8845 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_2_bow_tailored_keywords>`)
      - 3.2%
    * - :ref:`HaSpeeDe3 (XReligiousHate) <haspeede>`
      - 2023 
      - macro-:math:`f_1`
      - 0.6525
      - 0.5522 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_2_bow_tailored_keywords>`)
      - 18.2% 
    * - :ref:`HODI <hodi>` 
      - 2023
      - macro-:math:`f_1`
      - 0.81079 
      - 0.71527 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_3_bows_tailored_keywords>`)
      - 13.4%
    * - :ref:`ACTI <acti>`
      - 2023
      - Accuracy
      - 0.85712
      - 0.78207 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_3_bows_tailored_keywords>`)
      - 9.6%
    * - :ref:`PoliticIT (Global) <politicit>`
      - 2023
      - 
      - 0.824057
      - 0.762001
      - 8.1%
    * - :ref:`PoliticIT (Gender) <politicit>`
      - 2023
      - macro-:math:`f_1`
      - 0.824287
      - 0.732259 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_3_bows>`)
      - 12.6%
    * - :ref:`PoliticIT (Ideology Binary) <politicit>`
      - 2023
      - macro-:math:`f_1`
      - 0.928223
      - 0.848525 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_3_bows>`)
      - 9.4%
    * - :ref:`PoliticIT (Ideology Multiclass) <politicit>`
      - 2023
      - macro-:math:`f_1`
      - 0.751477
      - 0.705220 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_3_bows>`)
      - 6.6%
    * - :ref:`PoliticEs (Global) <politicses>`
      - 2023
      - 
      - 0.811319
      - 0.777584
      - 4.3%
    * - :ref:`PoliticEs (Gender) <politicses>`
      - 2023
      - macro-:math:`f_1`
      - 0.829633
      - 0.711549 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_3_bows>`)
      - 16.6%
    * - :ref:`PoliticEs (Profession) <politicses>`
      - 2023
      - macro-:math:`f_1`
      - 0.860824
      - 0.837945 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_3_bows>`)
      - 2.7%
    * - :ref:`PoliticEs (Ideology Binary) <politicses>`
      - 2023
      - macro-:math:`f_1`
      - 0.896715
      - 0.891394 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_3_bows>`)
      - 0.6%
    * - :ref:`PoliticEs (Ideology Multiclass) <politicses>`
      - 2023
      - macro-:math:`f_1`
      - 0.691334
      - 0.669448 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_3_bows>`)
      - 3.3%
    * - :ref:`DA-VINCIS <davincis>`
      - 2023
      - :math:`f_1`
      - 0.9264
      - 0.8903 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_2_bow_tailored_all_keywords>`)
      - 4.1%
    * - :ref:`DA-VINCIS <davincis-2022>`
      - 2022
      - :math:`f_1`
      - 0.7817
      - 0.7510 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_2_bow_all_keywords>`)
      - 4.1%
    * - :ref:`Rest-Mex (Global) <restmex>`
      - 2023
      - see overview
      - 0.7790190145
      - 0.7375714730
      - 5.6%
    * - :ref:`Rest-Mex (Polarity) <restmex>`
      - 2023
      - see overview
      - 0.621691991
      - 0.554880778 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_bows>`)
      - 12.0%
    * - :ref:`Rest-Mex (Type) <restmex>`
      - 2023
      - see overview
      - 0.99032231
      - 0.980539122 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.bow_training_set>`)
      - 1.0%
    * - :ref:`Rest-Mex (Country) <restmex>`
      - 2023
      - see overview
      - 0.942028113
      - 0.927052594 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.bow_training_set>`)
      - 1.6%    
    * - :ref:`HOMO-MEX <homo-mex>`
      - 2023
      - macro-:math:`f_1`
      - 0.8847
      - 0.8050 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_3_bow_tailored_all_keywords>`)
      - 9.9%
    * - :ref:`HOPE (ES) <hope>`
      - 2023
      - macro-:math:`f_1`
      - 0.9161
      - 0.5214 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_bow_keywords_emojis_voc_selection>`)
      - 75.7%
    * - :ref:`HOPE (EN) <hope>`
      - 2023
      - macro-:math:`f_1`
      - 0.5012
      - 0.4651 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_bow_keywords_emojis>`)
      - 7.8%
    * - :ref:`DIPROMATS (ES) <dipromats>`
      - 2023
      - :math:`f_1`
      - 0.8089
      - 0.7485 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_3_bows>`)
      - 8.1%
    * - :ref:`DIPROMATS (EN) <dipromats>`
      - 2023
      - :math:`f_1`
      - 0.8090
      - 0.7255 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_3_bow_tailored_all_keywords>`)
      - 11.5%
    * - :ref:`HUHU <huhu>`
      - 2023
      - :math:`f_1`
      - 0.820
      - 0.775 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_2_bow_all_keywords>`)
      - 5.8%
    * - :ref:`TASS <tass>`
      - 2017
      - macro-:math:`f_1`
      - 0.577  
      - 0.525 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_2_bow_tailored_all_keywords>`)
      - 9.9%
    * - :ref:`EDOS (A) <edos>`
      - 2023
      - macro-:math:`f_1`
      - 0.8746
      - 0.7890 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_2_bow_keywords>`)
      - 10.8%
    * - :ref:`EDOS (B) <edos>`
      - 2023
      - macro-:math:`f_1`
      - 0.7326
      - 0.5413 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_bow_keywords_emojis>`)
      - 35.3%
    * - :ref:`EDOS (C) <edos>`
      - 2023
      - macro-:math:`f_1`
      - 0.5606
      - 0.3388 (:py:func:`Conf. <EvoMSA.competitions.Comp2023.stack_2_bow_all_keywords>`)
      - 65.5%

Competitions
------------------------------

.. toctree::
  :maxdepth: 1

  haspeede
  hodi
  acti
  politicit
  politices
  davincis
  restmex
  homo-mex
  hope
  dipromats
  huhu
  tass
  edos
  comp_systems
