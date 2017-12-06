# EvoMSA
Sentiment Analysis System based on B4MSA and EvoDAG

## Quick Start ##

There are two options to use EvoMSA, one is as a library
and the other is using the command line interface.

### Using EvoMSA as library ###

```python
# Importing EvoMSA
from EvoMSA.base import EvoMSA

# Reading data
from EvoMSA import base
from b4msa.utils import tweet_iterator
import os
tweets = os.path.join(os.path.dirname(base.__file__), 'tests', 'tweets.json')
D = [[x['text'], x['klass']] for x in tweet_iterator(tweets)]

#train the model
evo = EvoMSA(n_jobs=4).fit([x[0] for x in D], [x[1] for x in D])

#predict X using the model
hy = evo.predict([x[0] for x in D])
```

### Using EvoMSA from the command line

Let us assume one wants to create a classifier using a
set of label tweets, i.e. `tweets.json`.


#### Training EvoDAG


```bash   
EvoMSA-train -n4 -o model.evomsa tweets.json 
```

#### Predict 

Once the model is obtained, it is time to use it:

```bash   
EvoMSA-predict -m model.evomsa -o predictions.json tweets.json
```

where `-o` indicates the file name used to store the predictions, `-m`
contains the model, and `tweets.json` is the test set.


# Performance #

|Competition | Macro-F1 | Macro-Recall|Competition Results|
|----------:|--------:|-----------:|---------------:|
|TASS 2016 | 0.5081 | 0.5639| [Task 1 General Corpus](http://ceur-ws.org/Vol-1896/p0_overview_tass2017.pdf)|
|SemEval 2017 - Arabic| 0.5039 |0.5293|[Results](https://competitions.codalab.org/competitions/15887/results/27549/data)|
|SemEval 2017 - English | 0.5817 | 0.6171|[Results](https://competitions.codalab.org/competitions/15885/results/27545/data)|
