.. _tass:

`Workshop on Sentiment Analysis at SEPLN (TASS) <https://ceur-ws.org/Vol-1896/p0_overview_tass2017.pdf>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`The TASS <https://ceur-ws.org/Vol-1896/p0_overview_tass2017.pdf>`_ workshop aims at the identification of polarity in Spanish tweets. 

.. code-block:: python

  >>> from EvoMSA.competitions import Comp2023
  >>> D = # Training set
  >>> tailored = 'tass2016'  
  >>> comp2023 = Comp2023(lang='es', tailored=tailored)
  >>> ins = comp2023.stack_2_bow_tailored_all_keywords(D)

.. list-table:: Performance in Cross-validation (A)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_2_bow_tailored_all_keywords`
      - 0.5461
      - 1.0000
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.5452
      - 0.2640
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.5431
      - 0.2280
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.5428
      - 0.0720
    * - :py:func:`Comp2023.stack_3_bow_tailored_all_keywords`
      - 0.5418
      - 0.0200
    * - :py:func:`Comp2023.stack_2_bow_all_keywords`
      - 0.5417
      - 0.0220
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.5405
      - 0.0080
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.5386
      - 0.0140
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.5186
      - 0.0000
    * - :py:func:`Comp2023.stack_bows`
      - 0.5078
      - 0.0000
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.4950
      - 0.0000
    * - :py:func:`Comp2023.bow`
      - 0.4933
      - 0.0000
    * - :py:func:`Comp2023.bow_training_set`
      - 0.4919
      - 0.0000
