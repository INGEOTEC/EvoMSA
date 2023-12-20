.. _dipromats:

`Automatic Detection and Characterization of Propaganda Techniques from Diplomats (DIPROMATS) <http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6569/3969>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`The DIPROMATS <http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6569/3969>`_ task (`webpage <https://sites.google.com/view/dipromats2023>`_)presented at IberLEF 2023 focused on indentifying tweets that have a propaganda technique. 

The following code can generate an instance of the system used in the competition.

.. code-block:: python

  >>> from EvoMSA.competitions import Comp2023
  >>> D = # Training set
  >>> comp2023 = Comp2023(lang='es')
  >>> ins = comp2023.stack_3_bows(D)

.. list-table:: Performance in Cross-validation (Spanish)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.6551
      - 1.0000
    * - :py:func:`Comp2023.stack_3_bow_tailored_all_keywords`
      - 0.6544
      - 0.4180
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.6515
      - 0.2200
    * - :py:func:`Comp2023.stack_2_bow_tailored_all_keywords`
      - 0.6514
      - 0.2500
    * - :py:func:`Comp2023.stack_bows`
      - 0.6488
      - 0.1120
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.6486
      - 0.1360
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.6486
      - 0.1360
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.6486
      - 0.1520
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.6485
      - 0.1480
    * - :py:func:`Comp2023.stack_2_bow_all_keywords`
      - 0.6484
      - 0.1180
    * - :py:func:`Comp2023.bow_training_set`
      - 0.6290
      - 0.0080
    * - :py:func:`Comp2023.bow`
      - 0.6136
      - 0.0000
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.6123
      - 0.0000


The following code can generate an instance of the system used in the competition.

.. code-block:: python

  >>> from EvoMSA.competitions import Comp2023
  >>> D = # Training set
  >>> tailored = 'IberLEF2023_DIPROMATS_task1'
  >>> comp2023 = Comp2023(lang='en', tailored=tailored)
  >>> ins = comp2023.stack_3_bow_tailored_all_keywords(D)

.. list-table:: Performance in Cross-validation (English)
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_3_bow_tailored_all_keywords`
      - 0.6498
      - 1.0000
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.6489
      - 0.2260
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.6471
      - 0.1280
    * - :py:func:`Comp2023.stack_2_bow_all_keywords`
      - 0.6448
      - 0.0440
    * - :py:func:`Comp2023.stack_2_bow_tailored_all_keywords`
      - 0.6446
      - 0.0140
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.6443
      - 0.0240
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.6386
      - 0.0080
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.6381
      - 0.0000
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.6377
      - 0.0040
    * - :py:func:`Comp2023.stack_bows`
      - 0.6327
      - 0.0000
    * - :py:func:`Comp2023.bow_training_set`
      - 0.6043
      - 0.0000
    * - :py:func:`Comp2023.bow`
      - 0.5961
      - 0.0000
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.5922
      - 0.0000
