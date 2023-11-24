.. _acti:

`Automatic Conspiracy Theory Identification (ACTI) <https://russogiuseppe.github.io/ACTI>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`The ACTI <https://ceur-ws.org/Vol-3473/paper36.pdf>`_ task presented at EVALITA 2023 focused on the automatic identification of conspiracy theories.

.. code-block:: python

  >>> from EvoMSA.competitions import Comp2023
  >>> D = # Training set
  >>> tailored = 'evalita2023_ACTI_It.json.gz'
  >>> comp2023 = Comp2023(lang='it', tailored=tailored)
  >>> ins = comp2023.stack_3_bows_tailored_keywords(D)

.. list-table:: Performance in Cross-validation
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.7914
      - 1.0000
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.7872
      - 0.1180
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.7855
      - 0.1700
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.7855
      - 0.0740
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.7804
      - 0.0240
    * - :py:func:`Comp2023.stack_bows`
      - 0.7785
      - 0.0260
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.7780
      - 0.0200
    * - :py:func:`Comp2023.bow_training_set`
      - 0.7765
      - 0.0040
    * - :py:func:`Comp2023.bow`
      - 0.7758
      - 0.0040
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.7755
      - 0.0000
