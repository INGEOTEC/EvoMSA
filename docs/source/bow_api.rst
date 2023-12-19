.. _bow_api:

================================
:py:class:`BoW` API
================================
.. image:: https://github.com/INGEOTEC/EvoMSA/actions/workflows/test.yaml/badge.svg
		:target: https://github.com/INGEOTEC/EvoMSA/actions/workflows/test.yaml

.. image:: https://coveralls.io/repos/github/INGEOTEC/EvoMSA/badge.svg?branch=develop
		:target: https://coveralls.io/github/INGEOTEC/EvoMSA?branch=develop

.. image:: https://badge.fury.io/py/EvoMSA.svg
		:target: https://badge.fury.io/py/EvoMSA

.. image:: https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/evomsa-feedstock?branchName=main
	    :target: https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=16466&branchName=main

.. image:: https://img.shields.io/conda/vn/conda-forge/evomsa.svg
		:target: https://anaconda.org/conda-forge/evomsa

.. image:: https://img.shields.io/conda/pn/conda-forge/evomsa.svg
		:target: https://anaconda.org/conda-forge/evomsa

.. image:: https://readthedocs.org/projects/evomsa/badge/?version=docs
		:target: https://evomsa.readthedocs.io/en/docs/?badge=docs

.. autoclass:: EvoMSA.text_repr.BoW
   :members: fit, transform, predict, decision_function, bow, names, weights, pretrain, lang, voc_selection, voc_size_exponent, v1, b4msa_fit, train_predict_decision_function, dependent_variable, cache, label_key, key, decision_function_name, kfold_class, kfold_kwargs, estimator_class, estimator_kwargs, estimator_instance, b4msa_kwargs, mixer_func, n_jobs

.. autoclass:: EvoMSA.back_prop.BoWBP
	:members: evolution, batches, deviation, parameters, model