[![Build Status](https://travis-ci.org/INGEOTEC/EvoMSA.svg?branch=master)](https://travis-ci.org/INGEOTEC/EvoMSA)
[![Build Status](https://travis-ci.org/INGEOTEC/EvoMSA.svg?branch=develop)](https://travis-ci.org/INGEOTEC/EvoMSA)
[![Build status](https://ci.appveyor.com/api/projects/status/wg01w00evm7pb8po?svg=true)](https://ci.appveyor.com/project/mgraffg/evomsa)
[![Build status](https://ci.appveyor.com/api/projects/status/wg01w00evm7pb8po/branch/master?svg=true)](https://ci.appveyor.com/project/mgraffg/evomsa/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/INGEOTEC/EvoMSA/badge.svg?branch=master)](https://coveralls.io/github/INGEOTEC/EvoMSA?branch=master)
[![Anaconda-Server Badge](https://anaconda.org/ingeotec/evomsa/badges/version.svg)](https://anaconda.org/ingeotec/evomsa)
[![Anaconda-Server Badge](https://anaconda.org/ingeotec/evomsa/badges/latest_release_date.svg)](https://anaconda.org/ingeotec/evomsa)
[![Anaconda-Server Badge](https://anaconda.org/ingeotec/evomsa/badges/platforms.svg)](https://anaconda.org/ingeotec/evomsa)
[![Anaconda-Server Badge](https://anaconda.org/ingeotec/evomsa/badges/installer/conda.svg)](https://anaconda.org/ingeotec/evomsa)
[![PyPI version](https://badge.fury.io/py/EvoMSA.svg)](https://badge.fury.io/py/EvoMSA)
[![Anaconda-Server Badge](https://anaconda.org/ingeotec/evomsa/badges/license.svg)](https://anaconda.org/ingeotec/evomsa)
[![Documentation Status](https://readthedocs.org/projects/evomsa/badge/?version=latest)](https://evomsa.readthedocs.io/en/latest/?badge=latest)

# EvoMSA
Sentiment Analysis System based on [B4MSA](https://github.com/ingeotec/b4msa) and [EvoDAG](https://github.com/mgraffg/EvoDAG).

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
