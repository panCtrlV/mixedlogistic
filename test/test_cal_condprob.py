__author__ = 'panc'

import unittest

import numpy as np
import scipy as sp

from mixedlogistic.cal_condprob import *
from data.tribolium import *
from mixedlogistic.helper import *


class TestCalCondProb(unittest.TestCase):
    def setUp(self):
        self.n = 100  # sample size
        # self.p = 2  # number of covariates
        self.x = np.matrix(np.random.randn(self.n, 1))

        # Prepare tribolium data
        # read data
        df = import_data()
        xm = pd.get_dummies(df.Replicate).ix[:, 2:]
        xr = pd.get_dummies(df.Species).ix[:, 1:]
        self.m = np.matrix(df.Total).T
        self.y = np.matrix(df.Remaining).T
        # pre-processing data
        self.xm = addIntercept(xm)  # add a leading 1s column
        self.xr = addIntercept(xr)  # add a leading 1s column
        self.c = 3  # three replicates as three components

    def test_cal_sigmoid(self):
        np.array_equal(cal_sigmoid(self.x), sp.special.expit(self.x))

    def test_cal_sigmoid_on_matrix_input(self):
        x = np.random.randn(100).reshape((20,5))
        x = np.matrix(x)
        b = np.random.random(10).reshape((5,2))
        b = np.matrix(b)

        xb = x*b

        f1 = cal_sigmoid(x*b)

        f2 = []
        for j in range(2):
            for i in range(20):
               f2.append(cal_sigmoid(xb[i,j]))

        f2 = np.matrix(f2).reshape((20,2), order='F')

        np.array_equal(f1, f2)

    def test_cal_binomProbForData(self):
        n = 10
        p = 2
        y = np.matrix(np.random.choice(np.arange(1, 10), n)).T
        x = np.matrix(np.random.randn(n * p).reshape((n, p)))
        m = np.matrix(np.random.choice(np.arange(20, 30), n)).T
        a = np.matrix(np.random.randn(2)).T
        binomprob = cal_binomProbForData(y, x, m, a)

        isinstance(binomprob, np.matrix)

        # Calculate binomial prob element-wise
        binomprob2 = []
        for i in range(n):
            prob = cal_sigmoid(x[i,:] * a)
            binomprob2.append( sp.stats.binom.pmf(y[i], m[i], prob)[0,0] )

        binomprob2 = np.matrix(binomprob2).T

        np.array_equal(binomprob, binomprob2)


    def test_cal_binomProbForData_on_matrix_input(self):
        x = np.matrix(np.random.randn(100).reshape((20,5)))
        a = np.matrix(np.random.random(25).reshape((5,5)))
        # logitxa = cal_sigmoidForData(x, a)
        m = np.matrix(np.random.choice(np.arange(10,20), 20)).T
        y = np.matrix(np.random.choice(np.arange(0,10), 20)).T

        binomp1 = cal_binomProbForData(y, x, m, a)

        binomp2 = []
        logitxa = cal_sigmoidForData(x, a)
        for j in range(5):
            for i in range(20):
                binomp2.append(sp.stats.binom.pmf(y[i], m[i], logitxa[i,j])[0,0])

        binomp2 = np.matrix(binomp2).reshape((20,5), order='F')

        np.array_equal(binomp1, binomp2)

    def test_cal_condProbOfComponentForData_sum1(self):
        pr = 2
        pm = 2
        n = 1  # number of observation
        c = 3  # three components
        y = np.matrix(np.random.choice(np.arange(1,10),n)).T

        _xr = np.matrix(np.random.randn(n*pr)).reshape((n, pr))
        xr = addIntercept(_xr)
        lxr = pr + 1

        _xm = np.matrix(np.random.randn(n*pm)).reshape((n, pm))
        xm = addIntercept(_xm)
        lxm = pm + 1

        beta = np.matrix(np.random.random(lxm * (c-1))).reshape((lxm, c-1))
        alpha = np.matrix(np.random.random(lxr * c)).reshape((lxr, c))
        m = np.matrix(np.random.choice(np.arange(10,20), n)).T

        # p, condprob = cal_condProbOfComponentForData(y, xm, xr, m, alpha, beta)
        # self.assertEqual(condprob.sum(), 1.)

        cal_condProbOfComponentForData(y, xm, xr, m, beta, alpha)

    def test_cal_condProbOfComponentForData_correctness(self):
        pr = 2
        pm = 2
        n = 10  # number of observation
        c = 3  # three components
        y = np.matrix(np.random.choice(np.arange(1,10),n)).T
        xr = np.matrix(np.random.randn(n*pr)).reshape((n, pr))
        xm = np.matrix(np.random.randn(n*pm)).reshape((n, pm))
        beta = np.matrix(np.random.random(pm * (c-1))).reshape((pm, c-1))
        alpha = np.matrix(np.random.random(pr*c)).reshape((pr, c))
        m = np.matrix(np.random.choice(np.arange(10,20), n)).T

        p, condprob1 = cal_condProbOfComponentForData(y, xm, xr, m, alpha, beta)

        beta = np.hstack((beta, np.zeros((pm, 1))))
        pxb = cal_softmaxForData(xm, beta)
        # xb = xm * beta
        # pxb = cal_sigmoid(xb)
        # pxb = [pxb[0,0], pxb[0,1], 1. - pxb.sum()]

        xa = xr * alpha
        pxa = cal_sigmoid(xa)
        binomp = []
        for i in range(n):
            for j in range(c):
                binomp.append(sp.stats.binom.pmf(y[i,0], m[i,0], pxa[i,j]))
        binomp = np.matrix(binomp).reshape((n, c))

        jointprob = np.multiply(pxb, binomp)
        jointprob_scaled = jointprob / jointprob.max(axis=1)
        condprob2 = jointprob_scaled / jointprob_scaled.sum(axis=1)
        condprob2 = np.matrix(condprob2)

        np.array_equal(condprob1, condprob2)

    def test_cal_softmax_smallInput(self):
        x = np.random.uniform(0, 1e-5, 5)
        print cal_softmax(x)

    def test_cal_softmax_largeInput(self):
        x = np.random.uniform(1e5, 1e10, 5)
        print cal_softmax(x)

if __name__ == '__main__':
    unittest.main()