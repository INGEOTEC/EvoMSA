.. _haspeede:

========================================================================================================
`Hate Speech Detection (HaSpeeDe3) <http://www.di.unito.it/~tutreeb/haspeede-evalita23/index.html>`_
========================================================================================================

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
