# Copyright 2020 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from .base import EvoMSA
from queue import LifoQueue
from microtc.utils import save_model


class Node(object):
    """
    Base class to perform model selection on the first-stage models

    :param model: Models used in the node - List of keys from :py:attr:`models`
    :type model: list
    :param models: Dictionary of pairs (see :py:attr:`EvoMSA.base.EvoMSA.models`)
    :type model: dict
    """

    def __init__(self, model, models=None):
        self._models = models
        self._model = [x for x in model]
        _ = self._model.copy()
        _.sort()
        self._repr = "-".join(map(str, _))

    def __repr__(self):
        return self._repr

    def __eq__(self, other):

        return isinstance(other, Node) and str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __iter__(self):
        variables = set(self._models.keys())
        model = self._model
        for x in variables - set(model):
            yield self.__class__(model + [x],
                                 models=self._models)

    @property
    def model(self):
        """Models as received by :py:class:`EvoMSA.base.EvoMSA`"""

        models = self._models
        return [models[x] for x in self._model]

    def fit(self, X, y, TR=False, test_set=None, **kwargs):
        """Create an EvoMSA's instance

        :param X: Training set - independent variables
        :type X: list
        :param y: Training set - dependent variable
        :type y: list or np.array
        :param TR: EvoMSA's default model
        :type TR: bool
        :param test_set: Dataset to perform transductive learning
        :type test_set: list
        :rtype: self
        """

        self._evo = EvoMSA(TR=TR, models=self.model,
                           **kwargs)
        if test_set is not None:
            self._evo.fit(X, y, test_set=test_set)
        else:
            self._evo.fit(X, y)
        return self

    @property
    def perf(self):
        """Performance"""
        return self._perf

    def performance(self, X, y, **kwargs):
        """Compute the performance on the dataset

        :param X: Test set - independent variables
        :type X: list
        :param y: Test set - dependent variable
        :type y: list or np.array
        :rtype: float
        """
        from sklearn import metrics

        try:
            return self._perf
        except AttributeError:
            hy = self._evo.predict(X, **kwargs)
            self._evo = None
            self._perf = metrics.f1_score(y, hy, average="macro")
        return self._perf

    def __cmp__(self, other):
        x = self.perf
        y = other.perf
        return (x > y) - (x < y)

    def __gt__(self, other):
        return self.perf > other.perf


class NodeNB(Node):
    """
    Using as stacked-method Naive Bayes with Gaussian distribution
    """

    def fit(self, X, y, stacked_method="sklearn.naive_bayes.GaussianNB",
            **kwargs):
        """
        :param stacked_method: Stacked method used in :py:class:`EvoMSA.base.EvoMSA`
        :type stacked_method: str
        """

        return super(NodeNB, self).fit(X, y, stacked_method=stacked_method,
                                       **kwargs)


class ForwardSelection(object):
    """Forward Selection on the models

    >>> from EvoMSA import base
    >>> from EvoMSA.utils import download
    >>> from EvoMSA.model_selection import ForwardSelection, NodeNB
    >>> from microtc.utils import tweet_iterator
    >>> import os

    Read the dataset

    >>> tweets = os.path.join(os.path.dirname(base.__file__), 'tests', 'tweets.json')
    >>> D = [[x['text'], x['klass']] for x in tweet_iterator(tweets)]
    
    Models

    >>> models = dict()
    >>> models[0] = ["EvoMSA.model.EmoSpaceEs", "sklearn.svm.LinearSVC"]
    >>> models[1] = ["EvoMSA.model.ThumbsUpDownEs", "sklearn.svm.LinearSVC"]
    >>> models[2] = [download("spanish.evoha"), "sklearn.svm.LinearSVC"]
    >>> X = [x for x, y in D]
    >>> y = [y for x, y in D]
    >>> fwdSel = ForwardSelection(models, node=NodeNB)
    >>> cache = os.path.join("cache", "train")
    >>> fwdSel.fit(X[:500], y[:500], cache=cache)

    Using the latest 500 elements to guide the search

    >>> cache = os.path.join("cache", "test")
    >>> best = fwdSel.run(X[500:], y[500:], cache=test)
    0-2

    :param models: Dictionary of pairs (see :py:attr:`EvoMSA.base.EvoMSA.models`)
    :type models: dict
    :param node: Node use to perform the search
    :type node: :py:class:`EvoMSA.model_selection.Node`
    :param output: Filename to store intermediate models
    :type output: str
    :param verbose: Level to inform the user
    :type verbose: int
    """

    def __init__(self, models, node=NodeNB, output=None, verbose=logging.INFO):
        self._models = models
        self._nodes = [node([k], models=models) for k in models.keys()]
        self._output = output
        self._logger = logging.getLogger("EvoMSA.model_selection")
        self._logger.setLevel(verbose)

    def fit(self, X, y, **kwargs):
        """Train the nodes having only one model

        :param X: Training set - independent variables
        :type X: list
        :param y: Training set - dependent variable
        :type y: list or np.array
        :rtype: self
        """

        self._X = X
        self._y = y
        self._kwargs = kwargs
        self._logger.info("Training the initial models")
        [x.fit(X, y, **kwargs) for x in self._nodes]
        return self

    def run(self, X, y, **kwargs):
        """Start the search using X and y to guide it

        :param X: Test set - independent variables
        :type X: list
        :param y: Test set - dependent variable
        :type y: list or np.array
        :rtype: :py:class:`EvoMSA.model_selection.Node`
        """

        self._logger.info("Starting the search")
        r = [(node.performance(X, y, **kwargs), node) for node in self._nodes]
        node = max(r, key=lambda x: x[0])[1]
        while True:
            self._logger.info("Model: %s perf: %0.4f" % (node, node.perf))
            nodes = [xx.fit(self._X, self._y, **self._kwargs) for xx in node]
            if len(nodes) == 0:
                return node
            r = [(xx.performance(X, y, **kwargs), xx) for xx in nodes]
            perf, comp = max(r, key=lambda x: x[0])
            if perf < node.perf:
                break
            node = comp
        return node


class BeamSelection(ForwardSelection):
    """
    Select the models using Beam Search.
    """

    def perf(self, node):
        """
        Node's performance

        :param node: Node
        :type node: :py:class:`EvoMSA.model_selection.Node`
        :rtype: float
        """

        return node.perf
        # try:
        #     perf = self._perf
        # except AttributeError:
        #     self._perf = dict()
        #     perf = self._perf
        # depth = len(node.model)
        # value = perf.get(depth, node.perf)
        # value = value if value > node.perf else node.perf
        # perf[depth] = value
        # return value

    def run(self, X, y, early_stopping=1000, **kwargs):
        """

        :param early_stopping: Number of rounds to perform early stopping
        :type early_stopping: int
        :rtype: :py:class:`EvoMSA.model_selection.Node`
        """
        visited = set([(node.performance(X, y, **kwargs), node) for
                       node in self._nodes])
        _ = max(visited, key=lambda x: x[0])[1]
        best = None
        nodes = LifoQueue()
        nodes.put(_)
        index = len(visited)
        while not nodes.empty() and (len(visited) - index) < early_stopping:
            node = nodes.get()
            if best is None or node > best:
                index = len(visited)
                best = node
                if self._output:
                    save_model(best, self._output)
            self._logger.info("Model: %s perf: %0.4f " % (best, best.perf) +
                              "visited: %s " % len(visited) +
                              "size: %s " % nodes.qsize() +
                              "Rounds: %s" % (len(visited) - index))
            nn = [(xx, xx.fit(self._X, self._y,
                              **self._kwargs).performance(X, y, **kwargs)) for
                  xx in node if xx not in visited]
            [visited.add(x) for x, _ in nn]
            nn = [xx for xx, perf in nn if perf >= self.perf(node)]
            if len(nn) == 0:
                continue
            nn.sort()
            [nodes.put(x) for x in nn]
        return best
