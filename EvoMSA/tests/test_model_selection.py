from model_selection import Node, NodeNB, ForwardSelection, BeamSelection
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
    from microtc.utils import tweet_iterator
    train = "data/train_all_en_B.json"
    test = "data/gold_en_B.json"

    models = {0: ["distant_A.tm", "sklearn.linear_model.LogisticRegression"],
              1: ["distant_B.tm", "EvoMSA.model.Identity"],
              2: ["distant_C.tm", "EvoMSA.model.Identity"]}
    a = NodeNB([0], models)
    D = list(tweet_iterator(train))
    a.fit(D, [x['klass'] for x in D])
    Dtest = list(tweet_iterator(test))
    perf = a.performance(Dtest, [x['klass'] for x in Dtest])
    assert perf < 1 and perf > 0.3


def test_ForwardSelection():
    from microtc.utils import tweet_iterator
    train = "data/train_all_en_B.json"
    test = "data/gold_en_B.json"

    models = {0: ["distant_A.tm", "sklearn.linear_model.LogisticRegression"],
              1: ["distant_B.tm", "EvoMSA.model.Identity"],
              2: ["distant_C.tm", "EvoMSA.model.Identity"]}
    
    D = list(tweet_iterator(train))
    a = ForwardSelection(models, node=NodeNB).fit(D, [x['klass'] for x in D],
                                                  cache=os.path.join("tm", "fw"))
    Dtest = list(tweet_iterator(test))
    a.run(Dtest, [x['klass'] for x in Dtest], cache=os.path.join("tm", "fw-test"))
    
    
def test_BeamSelection():
    from microtc.utils import tweet_iterator
    train = "data/train_all_en_B.json"
    test = "data/gold_en_B.json"

    models = {0: ["distant_A.tm", "sklearn.linear_model.LogisticRegression"],
              1: ["distant_B.tm", "EvoMSA.model.Identity"],
              2: ["distant_C.tm", "EvoMSA.model.Identity"]}
    
    D = list(tweet_iterator(train))
    a = BeamSelection(models, node=NodeNB).fit(D, [x['klass'] for x in D],
                                               cache=os.path.join("tm", "fw"))
    Dtest = list(tweet_iterator(test))
    a.run(Dtest, [x['klass'] for x in Dtest], cache=os.path.join("tm", "fw-test"))

