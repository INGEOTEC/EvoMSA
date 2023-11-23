.. _hope:

`Multilingual Hope Speech Detection (HOPE) <https://codalab.lisn.upsaclay.fr/competitions/10215>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation (Spanish)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.8224
      - 1.0000
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.8217
      - 0.3580
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.8192
      - 0.3680
    * - :py:func:`Comp2023.stack_2_bow_all_keywords`
      - 0.8192
      - 0.3040
    * - :py:func:`Comp2023.stack_3_bow_tailored_all_keywords`
      - 0.8192
      - 0.3680
    * - :py:func:`Comp2023.stack_2_bow_tailored_all_keywords`
      - 0.8159
      - 0.1740
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.8152
      - 0.1500
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.8119
      - 0.1020
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.7998
      - 0.0460
    * - :py:func:`Comp2023.bow`
      - 0.7966
      - 0.0260
    * - :py:func:`Comp2023.stack_bows`
      - 0.7966
      - 0.0260
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.7902
      - 0.0040
    * - :py:func:`Comp2023.bow_training_set`
      - 0.7807
      - 0.0000


.. list-table:: Performance in Cross-validation (English)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.7807
      - 1.0000
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.7801
      - 0.4600
    * - :py:func:`Comp2023.stack_bows`
      - 0.7753
      - 0.2860
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.7691
      - 0.0300
    * - :py:func:`Comp2023.stack_2_bow_tailored_all_keywords`
      - 0.7690
      - 0.0260
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.7680
      - 0.0220
    * - :py:func:`Comp2023.stack_2_bow_all_keywords`
      - 0.7662
      - 0.0120
    * - :py:func:`Comp2023.stack_3_bow_tailored_all_keywords`
      - 0.7661
      - 0.0200
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.7640
      - 0.0120
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.7575
      - 0.0020
    * - :py:func:`Comp2023.bow_training_set`
      - 0.7379
      - 0.0020
    * - :py:func:`Comp2023.bow`
      - 0.7300
      - 0.0000
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.7221
      - 0.0000
