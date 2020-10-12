from EvoMSA.model_selection import Node, NodeNB
from EvoMSA.model_selection import ForwardSelection, BeamSelection
from EvoMSA.tests.test_base import get_data
import os


def test_node():
    n = Node([1, 2], models={k: None for k in range(4)})
    n2 = Node([1, 2], models={k: None for k in range(4)})
    s = set()
    s.add(n)
    s.add(n2)
    assert len(s) == 1
    neighbors = set(n)
    assert Node([1, 2, 3]) in neighbors
    assert Node([1, 2, 0]) in neighbors
    str(Node("213", None) == "1-2-3")


def test_node_model():
    n = Node([1, 2], models={1: ["a", "b"], 2: ["c", "d"]})
    for a, b in zip(n.model, ["ab", "cd"]):
        assert "".join(a) == b


def test_NB():
    X, y = get_data()
    models = {0: ["EvoMSA.model.Corpus", "sklearn.svm.LinearSVC"],
              1: ["b4msa.textmodel.TextModel", "EvoMSA.model.Bernoulli"],
              2: ["EvoMSA.model.AggressivenessEs", "EvoMSA.model.Identity"]}
    a = NodeNB([0], models)
    a.fit(X[:500], y[:500])
    perf = a.performance(X[500:], y[500:])
    print(perf)
    assert perf < 1 and perf > 0.1


def test_ForwardSelection():
    X, y = get_data()
    models = {0: ["EvoMSA.model.Corpus", "sklearn.svm.LinearSVC"],
              1: ["b4msa.textmodel.TextModel", "EvoMSA.model.Bernoulli"],
              2: ["EvoMSA.model.AggressivenessEs", "EvoMSA.model.Identity"]}

    a = ForwardSelection(models,
                         node=NodeNB).fit(X[:500], y[:500],
                                          cache=os.path.join("tm", "fw"))
    a.run(X[500:], y[500:], cache=os.path.join("tm", "fw-test"))


def test_BeamSelection():
    X, y = get_data()
    models = {0: ["EvoMSA.model.Corpus", "sklearn.svm.LinearSVC"],
              1: ["b4msa.textmodel.TextModel", "EvoMSA.model.Bernoulli"],
              2: ["EvoMSA.model.AggressivenessEs", "EvoMSA.model.Identity"]}

    a = BeamSelection(models,
                      node=NodeNB).fit(X[:500], y[:500],
                                       cache=os.path.join("tm", "fw"))
    a.run(X[500:], y[500:], cache=os.path.join("tm", "fw-test"),
          early_stopping=2)
