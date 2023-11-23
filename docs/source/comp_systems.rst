.. _comp_systems:

========================
Competition Systems
========================

We test 13 different combinations of :ref:`BoW` and :ref:`DenseBoW` models. These models include the use of the two procedures to select the vocabulary (parameter voc_selection), the use of pre-trained :ref:`BoW`, and the creation of the :ref:`BoW` representation with the given training set. Additionally, we create text representations tailored to the problem at hand. That is the words with more discriminant power in a :ref:`BoW` classifier, trained on the training set, are selected as the labels in self-supervised problems. 

.. autoclass:: EvoMSA.competitions.Comp2023
   :members:

.. _tailored-keywords:

Tailored Keywords
-----------------------------

.. code-block:: python

    bow = BoW(lang=LANG, pretrain=False).fit(D)
    keywords = DenseBoW(lang=LANG, emoji=False, dataset=False).names
    tokens = [(name, np.median(np.fabs(w * v)))
              for name, w, v in zip(bow.names, bow.weights, bow.estimator_instance.coef_.T) 
              if name[:2] != 'q:' and '~' not in name and name not in keywords]
    tokens.sort(key=lambda x: x[1], reverse=True)
    semi = SelfSupervisedDataset([k for k, _ in tokens[:2048]],
                                 tempfile=f'{MODEL}.gz',
                                 bow=BoW(lang=LANG), capacity=1, n_jobs=63)
    semi.process(PATH_DATASET, output=MODEL)
