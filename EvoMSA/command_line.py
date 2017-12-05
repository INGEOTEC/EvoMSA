# Copyright 2017 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import EvoMSA
from EvoMSA import base
from b4msa.utils import tweet_iterator
import gzip
import pickle
import json


class CommandLine(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='EvoMSA')
        pa = self.parser.add_argument
        pa('--version',
           action='version', version='EvoMSA %s' % EvoMSA.__version__)
        pa('--verbose', dest='verbose', default=logging.NOTSET, type=int)
        pa('-n', '--n_jobs', help='Number of cores',
           dest='n_jobs', default=1, type=int)
        pa('-o', '--output-file',
           help='File / directory to store the result(s)', dest='output_file',
           default=None, type=str)

    def parse_args(self):
        self.data = self.parser.parse_args()
        if hasattr(self.data, 'verbose'):
            logging.basicConfig()
            for k in ['EvoDAG', 'EvoMSA']:
                logger = logging.getLogger(k)
                logger.setLevel(self.data.verbose)
                logger.info('Logging to: %s', self.data.verbose)
        self.main()


class CommandLineTrain(CommandLine):
    def __init__(self):
        super(CommandLineTrain, self).__init__()
        self.training_set()
        pa = self.parser.add_argument
        pa('--evodag-kw', dest='evo_kwargs', default=None, type=str,
           help='Parameters in json that overwrite EvoDAG default parameters')
        pa('--b4msa-kw', dest='b4msa_kwargs', default=None, type=str,
           help='Parameters in json that overwrite B4MSA default parameters')

    def training_set(self):
        cdn = 'File containing the training set.'
        pa = self.parser.add_argument
        pa('training_set',  nargs='+',
           default=None, help=cdn)

    def main(self):
        fnames = self.data.training_set
        if not isinstance(fnames, list):
            fnames = [fnames]
        D = []
        Y = []
        for fname in fnames:
            _ = [[x['text'], x['klass']] for x in tweet_iterator(fname)]
            D.append([x[0] for x in _])
            Y.append([x[1] for x in _])
        evo_kwargs = dict(tmpdir=self.data.output_file + '_dir')
        if self.data.evo_kwargs is not None:
            _ = json.loads(self.data.evo_kwargs)
            evo_kwargs.update(_)
        b4msa_kwargs = {}
        if self.data.b4msa_kwargs is not None:
            _ = json.loads(self.data.b4msa_kwargs)
            b4msa_kwargs.update(_)
        evo = base.EvoMSA(n_jobs=self.data.n_jobs, b4msa_args=b4msa_kwargs,
                          evodag_args=evo_kwargs).fit(D, Y)
        with gzip.open(self.data.output_file, 'w') as fpt:
            pickle.dump(evo, fpt)


class CommandLinePredict(CommandLine):
    def __init__(self):
        super(CommandLinePredict, self).__init__()
        pa = self.parser.add_argument
        pa('predict_file',  nargs='+',
           default=None, help='File to predict.')
        pa('-m', '--model', dest='model', default=None,
           help='Model')

    def main(self):
        predict_file = self.data.predict_file[0]
        D = [x['text'] for x in tweet_iterator(predict_file)]
        print(self.data.model)
        with gzip.open(self.data.model, 'r') as fpt:
            evo = pickle.load(fpt)
        hy = evo.predict(D)
        with open(self.data.output_file, 'w') as fpt:
            for x, y in zip(tweet_iterator(predict_file), hy):
                x.update(dict(klass=y))
                fpt.write(json.dumps(x) + '\n')


def train(output=False):
    c = CommandLineTrain()
    c.parse_args()
    if output:
        return c


def predict(output=False):
    c = CommandLinePredict()
    c.parse_args()
    if output:
        return c
    
