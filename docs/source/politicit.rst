.. _politicit:

`Political Ideology Detection in Italian Texts (PoliticIT) <https://codalab.lisn.upsaclay.fr/competitions/8507>`_ 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance in Cross-validation (Gender)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.9792
      - 1.0000
    * - :py:func:`Comp2023.stack_bows`
      - 0.9583
      - 0.2120
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.9583
      - 0.2340
    * - :py:func:`Comp2023.bow_training_set`
      - 0.9375
      - 0.1260
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.8748
      - 0.0200
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.8748
      - 0.0200
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.8536
      - 0.0160
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.8536
      - 0.0160
    * - :py:func:`Comp2023.bow`
      - 0.8307
      - 0.0000
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.8307
      - 0.0000


.. list-table:: Performance in Cross-validation (Ideology Binary)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.bow_training_set`
      - 1.0000
      - 1.0000
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.9714
      - 0.1580
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.9714
      - 0.1580
    * - :py:func:`Comp2023.stack_bows`
      - 0.8712
      - 0.0200
    * - :py:func:`Comp2023.bow`
      - 0.8487
      - 0.0120
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.8487
      - 0.0120
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.8271
      - 0.0060
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.8271
      - 0.0060
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.7856
      - 0.0040
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.7856
      - 0.0040


.. list-table:: Performance in Cross-validation (Ideology Multiclass)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.9834
      - 1.0000
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.9834
      - 1.0000
    * - :py:func:`Comp2023.bow_training_set`
      - 0.9823
      - 0.4100
    * - :py:func:`Comp2023.stack_bows`
      - 0.7756
      - 0.0020
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.7271
      - 0.0000
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.7271
      - 0.0000
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.7111
      - 0.0000
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.7111
      - 0.0000
    * - :py:func:`Comp2023.bow`
      - 0.5308
      - 0.0000
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.5308
      - 0.0000
