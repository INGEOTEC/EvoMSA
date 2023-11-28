.. _haspeede:

========================================================================================================
`Hate Speech Detection (HaSpeeDe3) <https://ceur-ws.org/Vol-3473/paper22.pdf>`_
========================================================================================================


The `HaSpeeDe3 <https://ceur-ws.org/Vol-3473/paper22.pdf>`_ task presented at EVALITA 2023 focused at the detection of hateful content written in Italian tweets. 

The following code can generate an instance of the system used in the competition.

.. code-block:: python

  >>> from EvoMSA.competitions import Comp2023
  >>> D = # Training set
  >>> tailored = 'evalita2023_HaSpeeDe3'
  >>> comp2023 = Comp2023(lang='it', tailored=tailored)
  >>> ins = comp2023.stack_2_bow_tailored_keywords(D)


.. list-table:: Performance in Cross-validation
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.8778
      - 1.0000
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.8769
      - 0.2720
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.8756
      - 0.0500
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.8754
      - 0.1380
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.8751
      - 0.1600
    * - :py:func:`Comp2023.stack_bows`
      - 0.8745
      - 0.1020
    * - :py:func:`Comp2023.bow`
      - 0.8740
      - 0.0780
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.8727
      - 0.0080
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.8723
      - 0.0260
    * - :py:func:`Comp2023.bow_training_set`
      - 0.8688
      - 0.0000
