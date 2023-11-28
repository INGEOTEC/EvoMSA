.. _homo-mex:

`Hate Speech Detection towards the Mexican Spanish Speaking LGBT+ Population (HOMO-MEX) <https://codalab.lisn.upsaclay.fr/competitions/10019>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 `The HOMO-MEX <http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6566>`_ task presented at IberLEF 2023 focused on detecting LGBTQ+ phobic content in Spanish tweets.

.. list-table:: Performance in Cross-validation
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_3_bow_tailored_all_keywords`
      - 0.7914
      - 1.0000
    * - :py:func:`Comp2023.stack_2_bow_tailored_all_keywords`
      - 0.7912
      - 0.4460
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.7908
      - 0.3420
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.7904
      - 0.2980
    * - :py:func:`Comp2023.stack_2_bow_all_keywords`
      - 0.7903
      - 0.2700
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.7901
      - 0.0740
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.7885
      - 0.1300
    * - :py:func:`Comp2023.stack_bows`
      - 0.7880
      - 0.1460
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.7871
      - 0.0660
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.7861
      - 0.0160
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.7689
      - 0.0000
    * - :py:func:`Comp2023.bow`
      - 0.7669
      - 0.0000
    * - :py:func:`Comp2023.bow_training_set`
      - 0.7553
      - 0.0000
