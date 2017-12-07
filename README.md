[![Build Status](https://travis-ci.org/INGEOTEC/EvoMSA.svg?branch=master)](https://travis-ci.org/INGEOTEC/EvoMSA)
[![Build status](https://ci.appveyor.com/api/projects/status/wg01w00evm7pb8po?svg=true)](https://ci.appveyor.com/project/mgraffg/evomsa)
[![Anaconda-Server Badge](https://anaconda.org/ingeotec/evomsa/badges/version.svg)](https://anaconda.org/ingeotec/evomsa)
[![Anaconda-Server Badge](https://anaconda.org/ingeotec/evomsa/badges/installer/conda.svg)](https://anaconda.org/ingeotec/evomsa)
[![Coverage Status](https://coveralls.io/repos/github/INGEOTEC/EvoMSA/badge.svg?branch=master)](https://coveralls.io/github/INGEOTEC/EvoMSA?branch=master)

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

|Competition | Scheme | Macro-F1 | Macro-Recall|Competition Results|
|----------:|-------:|--------:|-----------:|---------------:|
|TASS 2016 | Training set (T.S.) |0.5081 | 0.5639| [Task 1 General Corpus](http://ceur-ws.org/Vol-1896/p0_overview_tass2017.pdf)|
|TASS 2016 | T.S. + distant supervision dataset | 0.5152 | 0.5769 | Average of 30 datasets|
|TASS 2016 | T.S. + external dataset | 0.5101 | 0.5656 | |
|SemEval 2017 - Arabic| Training set (T.S.) |0.5039 |0.5293|[Results](https://competitions.codalab.org/competitions/15887/results/27549/data)|
|SemEval 2017 - Arabic| T.S. + distant supervision dataset |0.5041 | 0.5295 | Average of 30 datasets|
|SemEval 2017 - Arabic| T.S. + external dataset |0.5168 |0.5408| |
|SemEval 2017 - English | Training set (T.S.) | 0.5817 | 0.6171|[Results](https://competitions.codalab.org/competitions/15885/results/27545/data)|
|SemEval 2017 - English| T.S. + distant supervision dataset |0.5888 |0.6212| Average of 30 datasets |
|SemEval 2017 - English| T.S.  + external dataset | 0.6054 | 0.6354 | |
