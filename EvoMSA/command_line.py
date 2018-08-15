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
import importlib
import logging
import EvoMSA
from EvoMSA import base
from b4msa.utils import tweet_iterator
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from scipy.stats import pearsonr
import gzip
import pickle
import json
import os
import numpy as np
from scipy.stats import wilcoxon
from multiprocessing import Pool
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def fitness_vs(k_model):
    k, model = k_model
    if model == '-':
        return k, '-'
    model = CommandLine.load_model(model)
    return k, np.mean([x.fitness_vs for x in model._evodag_model._m.models])


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
            d = [base.EvoMSA.tolist(x[self._decision_function]) for x in tweet_iterator(fname)]
            if D is None:
                D = d
            else:
                [v.__iadd__(w) for v, w in zip(D, d)]
        self._exogenous = np.array(D)

    @staticmethod
    def get_class(m):
        if isinstance(m, str):
            a = m.split('.')
            p = importlib.import_module('.'.join(a[:-1]))
            return getattr(p, a[-1])
        return m

    @staticmethod
    def load_model(fname):
        if os.path.isfile(fname):
            with gzip.open(fname, 'r') as fpt:
                return pickle.load(fpt)
        else:
            cls = CommandLine.get_class(fname)
            ins = cls()
            return ins


class CommandLineTrain(CommandLine):
    def __init__(self):
        super(CommandLineTrain, self).__init__()
        self.training_set()
        pa = self.parser.add_argument
        pa('--ieee-cim', dest='ieee_cim', default=None, type=str,
           help='Model used in Computational Intelligence Magazine avaiable language AR, EN, and ES')
        pa('--kw', dest='kwargs', default=None, type=str,
           help='Parameters in json that overwrite EvoMSA default parameters')
        pa('--evodag-kw', dest='evo_kwargs', default=None, type=str,
           help='Parameters in json that overwrite EvoDAG default parameters')
        pa('--b4msa-kw', dest='b4msa_kwargs', default=None, type=str,
           help='Parameters in json that overwrite B4MSA default parameters')
        pa('--logistic-regression-kw', dest='logistic_regression_kwargs',
           default=None, type=str,
           help='Parameters in json that overwrite Logistic Regression default parameters')
        pa('--test_set', dest='test_set', default=None, type=str,
           help='Test set to do transductive learning')
        pa('-P', '--parameters', dest='parameters', type=str,
           help='B4MSA parameters')
        pa('--exogenous-model', help='Exogenous model(s) - pickle.dump with gzip',
           dest='exogenous_model', default=None, type=str, nargs='*')

    def training_set(self):
        cdn = 'File containing the training set.'
        pa = self.parser.add_argument
        pa('training_set',  nargs='+',
           default=None, help=cdn)

    def params_ieee_cim(self):
        lang = self.data.ieee_cim.lower()
        assert lang in ['ar', 'en', 'es']
        if lang == 'ar':
            emo = 'EvoMSA.model.EmoSpaceAr'
            th = 'EvoMSA.model.ThumbsUpDownAr'
            # ha = 'EvoMSA.model.HaSpaceAr'
        elif lang == 'en':
            emo = 'EvoMSA.model.EmoSpaceEn'
            th = 'EvoMSA.model.ThumbsUpDownEn'
            # ha = 'EvoMSA.model.HaSpaceEn'
        elif lang == 'es':
            emo = 'EvoMSA.model.EmoSpace'
            th = 'EvoMSA.model.ThumbsUpDownEs'
            # ha = 'EvoMSA.model.HaSpace'
        kw = json.loads(self.data.kwargs) if self.data.kwargs is not None else dict()
        models = kw.get('models', list())
        h = {":".join(tt_cl) for tt_cl in models}
        for tt_cl in [['EvoMSA.model.B4MSATextModel', 'sklearn.svm.LinearSVC'],
                      [emo, 'sklearn.svm.LinearSVC'],
                      [th, 'EvoMSA.model.Identity']]:
            if ":".join(tt_cl) in h:
                continue
            models.append(tt_cl)
        kw.update(models=models)
        self.data.kwargs = json.dumps(kw)

    def main(self):
        if self.data.ieee_cim is not None:
            self.params_ieee_cim()
        fnames = self.data.training_set
        if not isinstance(fnames, list):
            fnames = [fnames]
        D = []
        Y = []
        for fname in fnames:
            _ = [[x, x[self._klass]] for x in tweet_iterator(fname)]
            D.append([x[0] for x in _])
            Y.append([x[1] for x in _])
        if self.data.test_set is not None:
            if os.path.isfile(self.data.test_set):
                test_set = [x for x in tweet_iterator(self.data.test_set)]
            else:
                test_set = self.data.test_set
        else:
            test_set = None
        kwargs = dict(n_jobs=self.data.n_jobs)
        if self.data.kwargs is not None:
            _ = json.loads(self.data.kwargs)
            kwargs.update(_)
        evo_kwargs = dict(tmpdir=self.data.output_file + '_dir', fitness_function='macro-F1')
        if self.data.evo_kwargs is not None:
            _ = json.loads(self.data.evo_kwargs)
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
        if self.data.exogenous_model is not None:
            evo.exogenous_model = [self.load_model(x) for x in self.data.exogenous_model]
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
        D = [x for x in tweet_iterator(predict_file)]
        evo = self.load_model(self.data.model)
        evo.exogenous = self._exogenous
        D = evo.transform(D)
        with open(self.data.output_file, 'w') as fpt:
            for x, v in zip(tweet_iterator(predict_file), D):
                _ = dict(vec=v.tolist())
                x.update(_)
                fpt.write(json.dumps(x) + '\n')

    def fitness(self):
        model_file = self.data.training_set[0]
        evo = self.load_model(model_file)
        print("Median fitness: %0.4f" % (evo._evodag_model._m.fitness_vs * -1))

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
            _ = [[x, x[self._klass]] for x in tweet_iterator(fname)]
            D.append([x[0] for x in _])
            Y.append([x[1] for x in _])
        self._logger.info('Reading test_set %s' % self.data.test_set)
        if self.data.test_set is not None:
            test_set = [x for x in tweet_iterator(self.data.test_set)]
        else:
            test_set = None
        kwargs = dict(n_jobs=self.data.n_jobs)
        if self.data.kwargs is not None:
            _ = json.loads(self.data.kwargs)
            kwargs.update(_)
        b4msa_kwargs = {}
        if self.data.b4msa_kwargs is not None:
            _ = json.loads(self.data.b4msa_kwargs)
            b4msa_kwargs.update(_)
        evo = base.EvoMSA(b4msa_params=self.data.parameters, b4msa_args=b4msa_kwargs, **kwargs)
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
        pa('--max-lines', dest='max_lines', default=10000, type=int,
           help='Maximum number of lines to predict in an instance')

    def raw_outputs(self, evo, D):
        predict_file = self.data.predict_file[0]
        pr = []
        max_lines = self.data.max_lines
        while len(D):
            pr.append(evo.raw_decision_function(D[:max_lines]))
            del D[:max_lines]
        X = np.concatenate(pr)
        with open(self.data.output_file, 'w') as fpt:
            for x, df in zip(tweet_iterator(predict_file), X):
                _ = {self._decision_function: df.tolist()}
                x.update(_)
                fpt.write(json.dumps(x) + '\n')

    def decision_function(self, evo, D):
        predict_file = self.data.predict_file[0]
        pr = []
        max_lines = self.data.max_lines
        while len(D):
            pr.append(evo.decision_function(D[:max_lines]))
            del D[:max_lines]
        X = np.concatenate(pr)
        with open(self.data.output_file, 'w') as fpt:
            for x, df in zip(tweet_iterator(predict_file), X):
                _ = {self._decision_function: df.tolist()}
                x.update(_)
                fpt.write(json.dumps(x) + '\n')

    def main(self):
        predict_file = self.data.predict_file[0]
        D = [x for x in tweet_iterator(predict_file)]
        evo = self.load_model(self.data.model)
        evo.exogenous = self._exogenous
        if self.data.raw_outputs:
            return self.raw_outputs(evo, D)
        elif self.data.decision_function:
            return self.decision_function(evo, D)
        max_lines = self.data.max_lines
        pr = []
        while len(D):
            pr.append(evo.predict_proba(D[:max_lines]))
            del D[:max_lines]
        pr = np.concatenate(pr)
        hy = evo._le.inverse_transform(pr.argmax(axis=1)).tolist()
        with open(self.data.output_file, 'w') as fpt:
            for x, y, df in zip(tweet_iterator(predict_file), hy, pr):
                _ = {self._klass: y, self._decision_function: df.tolist()}
                x.update(_)
                fpt.write(json.dumps(x) + '\n')


class CommandLinePerformance(CommandLine):
    def __init__(self):
        super(CommandLinePerformance, self).__init__()
        g = self.parser.add_mutually_exclusive_group(required=True)
        pa = self.parser.add_argument
        pa('predictions', nargs='*', default=None)
        ga = g.add_argument
        ga('-m', '--model', help='Model(s) - pickle.dump with gzip', dest='model',
           default=None, type=str, nargs='*')
        pa('--score', help='Score - default macroF1', dest='score',
           default='macroF1', type=str)
        ga('-y', help='Output measured', dest='output', default=None, type=str)
        self._macroF1 = lambda x, y: f1_score(x, y, average='macro')
        self._macroRecall = lambda x, y: recall_score(x, y, average='macro')
        self._macroPrecision = lambda x, y: precision_score(x, y, average='macro')
        self._accuracy = lambda x, y: accuracy_score(x, y)
        self._pearsonr = lambda x, y: pearsonr(x, y)[0]

    def output(self):
        y = [x[self._klass] for x in tweet_iterator(self.data.output)]
        le = base.LabelEncoderWrapper().fit(y)
        perf = getattr(self, "_%s" % self.data.score)
        y = le.transform(y)
        D = []
        I = []
        for fname in self.data.predictions:
            if fname == '-':
                D.append(I)
                I = []
                continue
            hy = le.transform([x[self._klass] for x in tweet_iterator(fname)])
            I.append(perf(y, hy))
        if len(I):
            D.append(I)
        D = np.array(D).T
        p, alpha = self.compute_p(D)
        self._p = p
        self._alpha = alpha
        for _p, _alpha, mu in zip(p, alpha, D.mean(axis=0)):
            cdn = ''
            if np.isfinite(_alpha):
                cdn = " *"
            print("%0.4f" % mu, cdn)

    def main(self):
        if self.data.output is not None:
            return self.output()
        if len([x for x in self.data.model if x == '-']):
            args = [(k, x) for k, x in enumerate(self.data.model)]
            if self.data.n_jobs > 1:
                p = Pool(self.data.n_jobs)
                res = [x for x in tqdm(p.imap_unordered(fitness_vs, args), total=len(args))]
                res.sort(key=lambda x: x[0])
                p.close()
            else:
                res = [fitness_vs(x) for x in tqdm(args)]
            D = []
            I = []
            for _, x in res:
                if x == '-':
                    D.append(I)
                    I = []
                    continue
                I.append(x)
            if len(I):
                D.append(I)
            D = np.array(D).T
        else:
            models = [self.load_model(d) for d in self.data.model]
            D = np.array([[x.fitness_vs for x in m._evodag_model._m.models] for m in models]).T
        # print(D, '***')
        p, alpha = self.compute_p(D)
        self._p = p
        self._alpha = alpha
        for m, _p, _alpha, mu in zip(self.data.model, p, alpha, D.mean(axis=0)):
            cdn = ''
            if np.isfinite(_alpha):
                cdn = " *"
            print("%0.4f" % mu, m, cdn)

    @staticmethod
    def compute_p(syss):
        p = []
        mu = syss.mean(axis=0)
        best = mu.argmax()
        for i in range(syss.shape[1]):
            if i == best:
                p.append(np.inf)
                continue
            try:
                pv = wilcoxon(syss[:, best], syss[:, i])[1]
                p.append(pv)
            except ValueError:
                p.append(np.inf)
        ps = np.argsort(p)
        alpha = [np.inf for _ in ps]
        m = ps.shape[0] - 1
        for r, i in enumerate(ps[:-1]):
            alpha_c = (0.05 / (m + 1 - (r + 1)))
            if p[i] > alpha_c:
                break
            alpha[i] = alpha_c
        return p, alpha


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


def performance(output=False):
    c = CommandLinePerformance()
    c.parse_args()
    if output:
        return c



