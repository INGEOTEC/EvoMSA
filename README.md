# EvoMSA
Sentiment Analysis System based on B4MSA and EvoDAG

## Quick Start ##

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
