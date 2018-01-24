"""Calibration of predicted probabilities."""

# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Balazs Kegl <balazs.kegl@gmail.com>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Mario Graff <mgraffg@ieee.org>
# License: BSD 3 clause
import numpy as np
from math import log
from scipy.optimize import fmin_bfgs
from sklearn.preprocessing import label_binarize
from SparseArray import SparseArray


class Calibration(object):
    def __init_(self):
        self._coef = None

    def predict_proba(self, X):
        df = [self.normalize(d) for d in X]
        coef = self._coef
        if self._nclasses > 2:
            _ = [[1. / (1. + np.exp(x * c[0] +
                                    c[1])) for x, c in zip(xx.T, cc)] for xx, cc in zip(df, coef)]
            _ = [x / np.sum(x, axis=0) for x in _]
        else:
            _ = [1. / (1. + np.exp(x[:, 1] * c[0] + c[1])) for x, c in zip(df, coef)]
            _ = [np.vstack([1 - x, x]) for x in _]
            # proba = np.array([np.vstack([1 - x, x]) for x in _]).T
            # proba = np.mean(proba, axis=-1)
            # proba /= np.sum(proba, axis=1)[:, np.newaxis]
        proba = np.mean(np.array(_).T, axis=-1)
        proba /= np.sum(proba, axis=1)[:, np.newaxis]
        proba[np.isnan(proba)] = 1. / self._nclasses
        return proba

    def fit(self, X, y):
        df = [self.normalize(d) for d in X]
        self._classes = np.unique(y)
        Y = label_binarize(y, self._classes)
        self._nclasses = self._classes.shape[0]
        if self._nclasses > 2:
            _ = [[_sigmoid_calibration(x, k) for x, k in zip(xx.T, Y.T)] for xx in df]
        else:
            _ = [_sigmoid_calibration(xx[:, 1], Y[:, 0]) for xx in df]
        self._coef = _
        return self

    @staticmethod
    def sp2array(d):
        if isinstance(d, SparseArray):
            return d.full_array()
        return d

    @staticmethod
    def normalize(df):
        df = np.array([Calibration.sp2array(x) for x in df])
        mu = df.mean(axis=0)
        df = (df - mu).T
        return df


def _sigmoid_calibration(df, y, sample_weight=None):
    """Probability Calibration with sigmoid method (Platt 2000)

    Parameters
    ----------
    df : ndarray, shape (n_samples,)
        The decision function or predict proba for the samples.

    y : ndarray, shape (n_samples,)
        The targets.

    sample_weight : array-like, shape = [n_samples] or None
        Sample weights. If None, then samples are equally weighted.

    Returns
    -------
    a : float
        The slope.

    b : float
        The intercept.

    References
    ----------
    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    F = df  # F follows Platt's notations
    tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning

    # Bayesian priors (see Platt end of section 2.2)
    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    T = np.zeros(y.shape)
    T[y > 0] = (prior1 + 1.) / (prior1 + 2.)
    T[y <= 0] = 1. / (prior0 + 2.)
    T1 = 1. - T

    def objective(AB):
        # From Platt (beginning of Section 2.2)
        E = np.exp(AB[0] * F + AB[1])
        P = 1. / (1. + E)
        l = -(T * np.log(P + tiny) + T1 * np.log(1. - P + tiny))
        if sample_weight is not None:
            return (sample_weight * l).sum()
        else:
            return l.sum()

    def grad(AB):
        # gradient of the objective function
        E = np.exp(AB[0] * F + AB[1])
        P = 1. / (1. + E)
        TEP_minus_T1P = P * (T * E - T1)
        if sample_weight is not None:
            TEP_minus_T1P *= sample_weight
        dA = np.dot(TEP_minus_T1P, F)
        dB = np.sum(TEP_minus_T1P)
        return np.array([dA, dB])

    AB0 = np.array([0., log((prior0 + 1.) / (prior1 + 1.))])
    AB_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False)
    return AB_[0], AB_[1]
