.. _huhu:

`HUrtful HUmour (HUHU) <http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6568>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`The HUHU <http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6568>`_ task (`webpage <https://sites.google.com/view/huhuatiberlef23>`_) presented at IberLEF 2023 aims at the identification of prejudiced tweets in Spanish that have the additional characteristic of presenting humour. 


The following code can generate an instance of the system used in the competition.

.. code-block:: python

  >>> from EvoMSA.competitions import Comp2023
  >>> D = # Training set
  >>> comp2023 = Comp2023(lang='es')
  >>> ins = comp2023.stack_2_bow_all_keywords(D)


.. list-table:: Performance in Cross-validation 
    :header-rows: 1

    * - Configuration
      - Performance
      - p-value
    * - :py:func:`Comp2023.stack_2_bow_all_keywords`
      - 0.7417
      - 1.0000
    * - :py:func:`Comp2023.stack_2_bow_tailored_all_keywords`
      - 0.7416
      - 0.4700
    * - :py:func:`Comp2023.stack_2_bow_tailored_keywords`
      - 0.7379
      - 0.1700
    * - :py:func:`Comp2023.stack_3_bow_tailored_all_keywords`
      - 0.7377
      - 0.1340
    * - :py:func:`Comp2023.stack_3_bows_tailored_keywords`
      - 0.7358
      - 0.1000
    * - :py:func:`Comp2023.stack_2_bow_keywords`
      - 0.7347
      - 0.0180
    * - :py:func:`Comp2023.stack_bow_keywords_emojis`
      - 0.7334
      - 0.0620
    * - :py:func:`Comp2023.stack_bow_keywords_emojis_voc_selection`
      - 0.7298
      - 0.0100
    * - :py:func:`Comp2023.stack_3_bows`
      - 0.7127
      - 0.0000
    * - :py:func:`Comp2023.stack_bows`
      - 0.7103
      - 0.0000
    * - :py:func:`Comp2023.bow_voc_selection`
      - 0.7034
      - 0.0000
    * - :py:func:`Comp2023.bow`
      - 0.6969
      - 0.0000
    * - :py:func:`Comp2023.bow_training_set`
      - 0.6833
      - 0.0000
