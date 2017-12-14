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

|Dataset|SemEval|2017|Arabic|[Results](https://competitions.codalab.org/competitions/15887/results/27549/data) |SemEval|2017|English|[Results](https://competitions.codalab.org/competitions/15885/results/27545/data)|  
|-----:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|  
|Performance|Macro|F1|Macro|Recall |Macro|F1|Macro|Recall| 
|Fitness| Recall| F1 | Recall | F1 | Recall | F1 | Recall | F1 |   
|Baseline (B4MSA)|0.5062 | 0.5062 | 0.5091 | 0.5091 | 0.5894 | 0.5894 | 0.6003 | 0.6003 |  
|Training Set (TS)|0.5040 | 0.5189 | 0.5294 | 0.5252 | 0.5817 | 0.6113 | 0.6171 | 0.6248 |  
|TS + Distant supervision (DS)|0.5041 | 0.5202 | 0.5295 | 0.5284 | 0.5888 | 0.6114 | 0.6212 | 0.6261 |  
|TS + External|0.5169 | 0.5370 | 0.5408 | 0.5407 | 0.6054 | 0.6228 | 0.6355 | 0.6361 |  
|TS + DS + External|0.5184 | **0.5350** | **0.5414** | 0.5386 | 0.6074 | **0.6277** | 0.6364 | **0.6390** |  
|DS|0.3820 | 0.4022 | 0.4121 | 0.4066 | 0.4251 | 0.4418 | 0.4591 | 0.4573 |  
|External|0.4643 | 0.4632 | 0.4697 | 0.4598 | 0.5812 | 0.5799 | 0.5871 | 0.5773 |  
|DS + External|0.4618 | 0.4659 | 0.4747 | 0.4694 | 0.5829 | 0.5827 | 0.5907 | 0.5775 |  



|Dataset|TASS|2016|Task 1|[G. Corpus](http://ceur-ws.org/Vol-1896/p0_overview_tass2017.pdf)|    |TASS|2017|Spanish|  
|-----:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|  
|Performance|Macro|F1|Macro|Recall |Macro|F1|Macro|Recall|  
|Fitness| Recall| F1 | Recall | F1 | Recall | F1 | Recall | F1 |   
|Baseline (B4MSA)|0.4969 | 0.4969 | 0.5197 | 0.5197 | 0.4012 | 0.4012 | 0.4129 | 0.4129 |  
|Training Set (TS)|0.5082 | 0.5068 | 0.5640 | 0.5657 | 0.4259 | 0.4523 | 0.4519 | **0.4584** |  
|TS + Distant supervision (DS)|0.5152 | 0.5161 | 0.5769 | 0.5789 | 0.4259 | 0.4324 | 0.4486 | 0.4520 |  
|TS + External|0.5102 | 0.5107 | 0.5657 | 0.5714 | 0.4195 | 0.4346 | 0.4294 | 0.4479 |  
|TS + DS + External|0.5173 | **0.5178** | 0.5786 | **0.5827** | 0.4497 | **0.4437** | 0.4575 | 0.4498 |  
|DS|0.4118 | 0.4164 | 0.4168 | 0.4186 | 0.3599 | 0.3663 | 0.3970 | 0.3815 |  
|External|0.3734 | 0.3726 | 0.3991 | 0.4010 | 0.4004 | 0.3994 | 0.4052 | 0.3999 |  
|DS + External|0.4370 | 0.4360 | 0.4565 | 0.4622 | 0.4102 | 0.4265 | 0.4174 | 0.4313 |  
