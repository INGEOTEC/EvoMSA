.. _edos:

`Explainable Detection of Online Sexism (EDOS) <https://aclanthology.org/2023.semeval-1.305/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`The EDOS <https://aclanthology.org/2023.semeval-1.305/>`_ task presented at SemEval 2023 aims at the detection of sexism. 

The following code can generate an instance of the system used in the competition.

.. code-block:: python

  >>> from EvoMSA.competitions import Comp2023
  >>> D = # Training set
  >>> comp2023 = Comp2023(lang='en')
  >>> ins = comp2023.stack_2_bow_keywords(D)


.. list-table:: Performance in Cross-validation (A)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.7622
      - 1.0000
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.7580
      - 0.2220
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.7567
      - 0.0960
    * - :py:func:`Comp2023.stack_2_bow_tailored_all_keywords`
      - 0.7532
      - 0.1100
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.7517
      - 0.0720
    * - :py:func:`Comp2023.stack_2_bow_all_keywords`
      - 0.7503
      - 0.0600
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.7502
      - 0.0280
    * - :py:func:`Comp2023.stack_3_bow_tailored_all_keywords`
      - 0.7487
      - 0.0300
    * - :py:func:`Comp2023.stack_bows`
      - 0.7486
      - 0.0540
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.7478
      - 0.0100
    * - :py:func:`Comp2023.bow`
      - 0.7398
      - 0.0060
    * - :py:func:`Comp2023.bow_training_set`
      - 0.7354
      - 0.0020
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.7350
      - 0.0000


The following code can generate an instance of the system used in the competition.

.. code-block:: python

  >>> from EvoMSA.competitions import Comp2023
  >>> D = # Training set
  >>> comp2023 = Comp2023(lang='en')
  >>> ins = comp2023.stack_bow_keywords_emojis(D)


.. list-table:: Performance in Cross-validation (B)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.5247
      - 1.0000
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.5123
      - 0.1580
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.5088
      - 0.1540
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.5064
      - 0.1040
    * - :py:func:`Comp2023.stack_2_bow_all_keywords`
      - 0.5002
      - 0.1440
    * - :py:func:`Comp2023.stack_2_bow_tailored_all_keywords`
      - 0.4969
      - 0.1000
    * - :py:func:`Comp2023.stack_3_bow_tailored_all_keywords`
      - 0.4950
      - 0.0960
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.4929
      - 0.0760
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.4924
      - 0.0080
    * - :py:func:`Comp2023.stack_bows`
      - 0.4909
      - 0.1000
    * - :py:func:`Comp2023.bow`
      - 0.4597
      - 0.0340
    * - :py:func:`Comp2023.bow_training_set`
      - 0.4450
      - 0.0140
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.4427
      - 0.0140


The following code can generate an instance of the system used in the competition.

.. code-block:: python

  >>> from EvoMSA.competitions import Comp2023
  >>> D = # Training set
  >>> comp2023 = Comp2023(lang='en')
  >>> ins = comp2023.stack_2_bow_all_keywords(D)


.. list-table:: Performance in Cross-validation (C)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_2_bow_all_keywords`
      - 0.3236
      - 1.0000
    * - :py:func:`Comp2023.stack_2_bow_tailored_all_keywords`
      - 0.3145
      - 0.0980
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.3123
      - 0.2760
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.3069
      - 0.1460
    * - :py:func:`Comp2023.stack_3_bow_tailored_all_keywords`
      - 0.3035
      - 0.0020
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.2943
      - 0.0580
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.2924
      - 0.0240
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.2870
      - 0.0120
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.2700
      - 0.0140
    * - :py:func:`Comp2023.bow`
      - 0.2685
      - 0.0140
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.2556
      - 0.0000
    * - :py:func:`Comp2023.bow_training_set`
      - 0.2530
      - 0.0080
    * - :py:func:`Comp2023.stack_bows`
      - 0.2486
      - 0.0000


