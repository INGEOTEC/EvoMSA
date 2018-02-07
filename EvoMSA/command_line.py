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
import os
import numpy as np


class CommandLine(object):
    def __init__(self):
        self._klass = os.getenv('KLASS', default='klass')
        self._text = os.getenv('TEXT', default='text')
        self._decision_function = os.getenv('DECISION_FUNCTION',
                                            default='decision_function')
        self.parser = argparse.ArgumentParser(description='EvoMSA')
        self._logger = logging.getLogger('EvoMSA')
        pa = self.parser.add_argument
        pa('--version',
           action='version', version='EvoMSA %s' % EvoMSA.__version__)
        pa('--verbose', dest='verbose', default=logging.NOTSET, type=int)
        pa('-n', '--n_jobs', help='Number of cores',
           dest='n_jobs', default=1, type=int)
        pa('-o', '--output-file',
           help='File / directory to store the result(s)', dest='output_file',
           default=None, type=str)
        pa('--exogenous',
           help='Exogenous variables', dest='exogenous',
           default=None, type=str, nargs='*')

    def parse_args(self):
        self.data = self.parser.parse_args()
        if hasattr(self.data, 'verbose'):
            logging.basicConfig()
            for k in ['EvoDAG', 'EvoMSA']:
                logger = logging.getLogger(k)
                logger.setLevel(self.data.verbose)
                logger.info('Logging to: %s', self.data.verbose)
        self.exogenous()
        self.main()

    def exogenous(self):
        self._exogenous = None
        if self.data.exogenous is None:
            return
        D = None
        for fname in self.data.exogenous:
            d = [base.EvoMSA.tolist(x['decision_function']) for x in tweet_iterator(fname)]
            if D is None:
                D = d
            else:
                [v.__iadd__(w) for v, w in zip(D, d)]
        self._exogenous = np.array(D)


class CommandLineTrain(CommandLine):
    def __init__(self):
        super(CommandLineTrain, self).__init__()
        self.training_set()
        pa = self.parser.add_argument
        pa('--kw', dest='kwargs', default=None, type=str,
           help='Parameters in json that overwrite EvoMSA default parameters')
        pa('--evodag-kw', dest='evo_kwargs', default=None, type=str,
           help='Parameters in json that overwrite EvoDAG default parameters')
        pa('--b4msa-kw', dest='b4msa_kwargs', default=None, type=str,
           help='Parameters in json that overwrite B4MSA default parameters')
        pa('--logistic-regression-kw', dest='logistic_regression_kwargs', default=None, type=str,
           help='Parameters in json that overwrite Logistic Regression default parameters')
        pa('--test_set', dest='test_set', default=None, type=str,
           help='Test set to do transductive learning')
        pa('-P', '--parameters', dest='parameters', type=str,
           help='B4MSA parameters')
        pa('--no-use-ts', dest='use_ts', default=True, action='store_false')

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
            _ = [[x[self._text], x[self._klass]] for x in tweet_iterator(fname)]
            D.append([x[0] for x in _])
            Y.append([x[1] for x in _])
        if self.data.test_set is not None:
            test_set = [x[self._text] for x in tweet_iterator(self.data.test_set)]
        else:
            test_set = None
        kwargs = dict(use_ts=self.data.use_ts, n_jobs=self.data.n_jobs)
        if self.data.kwargs is not None:
            _ = json.loads(self.data.kwargs)
            kwargs.update(_)
        evo_kwargs = dict(tmpdir=self.data.output_file + '_dir')
        if self.data.evo_kwargs is not None:
            _ = json.loads(self.data.evo_kwargs)
            _['fitness_function'] = 'macro-F1'
            evo_kwargs.update(_)
        b4msa_kwargs = {}
        if self.data.b4msa_kwargs is not None:
            _ = json.loads(self.data.b4msa_kwargs)
            b4msa_kwargs.update(_)
        logistic_regression_kwargs = None
        if self.data.logistic_regression_kwargs is not None:
            logistic_regression_kwargs = json.loads(self.data.logistic_regression_kwargs)
        evo = base.EvoMSA(b4msa_params=self.data.parameters,
                          b4msa_args=b4msa_kwargs, evodag_args=evo_kwargs,
                          logistic_regression_args=logistic_regression_kwargs,
                          **kwargs)
        evo.exogenous = self._exogenous
        evo.fit(D, Y, test_set=test_set)
        evo.exogenous = None
        with gzip.open(self.data.output_file, 'w') as fpt:
            evo._logger = None
            pickle.dump(evo, fpt)


class CommandLineUtils(CommandLineTrain):
    def __init__(self):
        super(CommandLineUtils, self).__init__()
        pa = self.parser.add_argument
        pa('--b4msa-df', dest='b4msa_df', default=False, action='store_true')
        pa('--transform', dest='transform', default=False, action='store_true')
        pa('-m', '--model', dest='model', default=None, help='Model')
        pa('--fitness', dest='fitness', default=False,
           help='Fitness in the validation set', action='store_true')

    def transform(self):
        predict_file = self.data.training_set[0]
        D = [x[self._text] for x in tweet_iterator(predict_file)]
        with gzip.open(self.data.model, 'r') as fpt:
            evo = pickle.load(fpt)
        evo.exogenous = self._exogenous
        D = evo.transform(D)
        with open(self.data.output_file, 'w') as fpt:
            for x, v in zip(tweet_iterator(predict_file), D):
                _ = dict(vec=v.tolist())
                x.update(_)
                fpt.write(json.dumps(x) + '\n')

    def fitness(self):
        model_file = self.data.training_set[0]
        with gzip.open(model_file, 'r') as fpt:
            evo = pickle.load(fpt)
        print("Median fitness: %0.4f" % (evo._evodag_model.fitness_vs * -1))

    def main(self):
        if self.data.transform:
            return self.transform()
        elif self.data.fitness:
            return self.fitness()
        if not self.data.b4msa_df:
            return
        fnames = self.data.training_set
        if not isinstance(fnames, list):
            fnames = [fnames]
        D = []
        Y = []
        for fname in fnames:
            _ = [[x[self._text], x[self._klass]] for x in tweet_iterator(fname)]
            D.append([x[0] for x in _])
            Y.append([x[1] for x in _])
        self._logger.info('Reading test_set %s' % self.data.test_set)
        if self.data.test_set is not None:
            test_set = [x[self._text] for x in tweet_iterator(self.data.test_set)]
        else:
            test_set = None
        b4msa_kwargs = {}
        if self.data.b4msa_kwargs is not None:
            _ = json.loads(self.data.b4msa_kwargs)
            b4msa_kwargs.update(_)
        evo = base.EvoMSA(b4msa_params=self.data.parameters,
                          n_jobs=self.data.n_jobs, b4msa_args=b4msa_kwargs)
        evo.fit_svm(D, Y)
        output = self.data.output_file
        if self.data.test_set is None:
            hy = evo.transform(D[0])
            with open(output, 'w') as fpt:
                for x, y in zip(tweet_iterator(fnames[0]), hy):
                    x.update(dict(vec=y.tolist()))
                    fpt.write(json.dumps(x) + '\n')
        else:
            if not os.path.isdir(output):
                os.mkdir(output)
            train = os.path.join(output, 'train.json')
            hy = evo.transform(D[0])
            with open(train, 'w') as fpt:
                for x, y in zip(tweet_iterator(fnames[0]), hy):
                    x.update(dict(vec=y.tolist()))
                    fpt.write(json.dumps(x) + '\n')
            test = os.path.join(output, 'test.json')
            hy = evo.transform(test_set)
            with open(test, 'w') as fpt:
                for x, y in zip(tweet_iterator(self.data.test_set), hy):
                    x.update(dict(vec=y.tolist()))
                    fpt.write(json.dumps(x) + '\n')


class CommandLinePredict(CommandLine):
    def __init__(self):
        super(CommandLinePredict, self).__init__()
        pa = self.parser.add_argument
        pa('predict_file',  nargs='+',
           default=None, help='File to predict.')
        pa('-m', '--model', dest='model', default=None,
           help='Model')
        pa('--raw-outputs', dest='raw_outputs', default=False,
           action='store_true',
           help='Raw decision function')
        pa('--decision-function', dest='decision_function', default=False,
           action='store_true',
           help='Ensemble decision function')

    def raw_outputs(self, evo, D):
        predict_file = self.data.predict_file[0]
        X = evo.raw_decision_function(D)
        with open(self.data.output_file, 'w') as fpt:
            for x, df in zip(tweet_iterator(predict_file), X):
                _ = {self._decision_function: df.tolist()}
                x.update(_)
                fpt.write(json.dumps(x) + '\n')

    def decision_function(self, evo, D):
        predict_file = self.data.predict_file[0]
        X = evo.decision_function(D)
        with open(self.data.output_file, 'w') as fpt:
            for x, df in zip(tweet_iterator(predict_file), X):
                _ = {self._decision_function: df.tolist()}
                x.update(_)
                fpt.write(json.dumps(x) + '\n')

    def main(self):
        predict_file = self.data.predict_file[0]
        D = [x[self._text] for x in tweet_iterator(predict_file)]
        with gzip.open(self.data.model, 'r') as fpt:
            evo = pickle.load(fpt)
        evo.exogenous = self._exogenous
        if self.data.raw_outputs:
            return self.raw_outputs(evo, D)
        elif self.data.decision_function:
            return self.decision_function(evo, D)
        pr = evo.predict_proba(D)
        hy = evo._le.inverse_transform(pr.argmax(axis=1))
        with open(self.data.output_file, 'w') as fpt:
            for x, y, df in zip(tweet_iterator(predict_file), hy, pr):
                _ = {self._klass: str(y), self._decision_function: df.tolist()}
                x.update(_)
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


def utils(output=False):
    c = CommandLineUtils()
    c.parse_args()
    if output:
        return c

