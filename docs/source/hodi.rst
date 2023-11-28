.. _hodi:

========================================================================================================
`Homotransphobia Detection in Italian (HODI) <https://hodi-evalita.github.io>`_
========================================================================================================

`The HODI <https://ceur-ws.org/Vol-3473/paper26.pdf>`_ task presented at EVALITA 2023 focused on the detection of homotransphobia in Italian tweets.

The following code can generate an instance of the system used in the competition.

.. code-block:: python

  >>> from EvoMSA.competitions import Comp2023
  >>> D = # Training set
  >>> tailored = 'evalita2023_HODI'
  >>> comp2023 = Comp2023(lang='it', tailored=tailored)
  >>> ins = comp2023.stack_3_bows_tailored_keywords(D)


.. list-table:: Performance in Cross-validation
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.7883
      - 1.0000
    * - :py:func:`Comp2023.stack_bows`
      - 0.7875
      - 0.3900
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.7864
      - 0.2360
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.7862
      - 0.2640
    * - :py:func:`Comp2023.bow`
      - 0.7842
      - 0.1100
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.7838
      - 0.0060
    * - :py:func:`Comp2023.bow_training_set`
      - 0.7833
      - 0.0620
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.7830
      - 0.0000
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.7765
      - 0.0000
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.7758
      - 0.0000
