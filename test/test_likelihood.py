__author__ = 'panc'

from likelihood import *
from mixedLogistic import mapOptimizationMatrices2Vector
import numpy as np
import unittest

class TestLikelihood(unittest.TestCase):
    def setUp(self):
        pass

    def test_ellk_evaluation(self):
        # data simulation
        pr = 2
        pm = 2
        n = 10 # number of observation
        c = 3 # three components
        y = np.matrix(np.random.choice(np.arange(1,10),n)).T
        xr = np.matrix(np.random.randn(n*pr)).reshape((n, pr))
        xm = np.matrix(np.random.randn(n*pm)).reshape((n, pm))
        beta = np.matrix(np.random.random(pm * (c-1))).reshape((pm, c-1))
        alpha = np.matrix(np.random.random(pr*c)).reshape((pr, c))
        m = np.matrix(np.random.choice(np.arange(10,20), n)).T

        q = ellk(y, xm, xr, m, beta, alpha, beta, alpha)
        print '\n'
        print q

    def test_ollk_evaluation(self):
        # data simulation
        pr = 2
        pm = 2
        n = 10 # number of observation
        c = 3 # three components
        y = np.matrix(np.random.choice(np.arange(1,10),n)).T
        xr = np.matrix(np.random.randn(n*pr)).reshape((n, pr))
        xm = np.matrix(np.random.randn(n*pm)).reshape((n, pm))
        beta = np.matrix(np.random.random(pm * (c-1))).reshape((pm, c-1))
        alpha = np.matrix(np.random.random(pr*c)).reshape((pr, c))
        m = np.matrix(np.random.choice(np.arange(10,20), n)).T

        q = ollk(y, xm, xr, m, beta, alpha)
        print '\n'
        print 'Observed log-likelihood = ', q

    def test_nollkForOptimization(self):
        # data simulation
        pr = 2
        pm = 2
        n = 10 # number of observation
        c = 3 # three components
        y = np.matrix(np.random.choice(np.arange(1,10),n)).T
        xr = np.matrix(np.random.randn(n*pr)).reshape((n, pr))
        xm = np.matrix(np.random.randn(n*pm)).reshape((n, pm))
        beta = np.matrix(np.random.random(pm * (c-1))).reshape((pm, c-1))
        alpha = np.matrix(np.random.random(pr*c)).reshape((pr, c))
        m = np.matrix(np.random.choice(np.arange(10,20), n)).T

        nollk = nollkForOptimization(c, y, xm, xr, m)
        vBetaAlpha = mapOptimizationMatrices2Vector(beta, alpha)
        q1 = nollk(vBetaAlpha)
        q2 = ollk(y, xm, xr, m, beta, alpha)
        self.assertEqual(q1, q2)


    # def test_logLikelihood_1(self):
    #     """
    #     Evaluate log-likelihood while no predictor is used.
    #
    #     :return:
    #     """
    #     # --- Generate data ---
    #     np.random.seed(25)
    #     ## Setting
    #     n = 100  # sample size
    #     c = 3  # number of components
    #     # p_comp = np.matrix(np.repeat(1./3, c))
    #     p_binom = [1./2, 1./3, 1./6]
    #     m = 50  # fixed number of trials for all observations
    #     ## Generation
    #     membership = np.random.choice([0,1,2], 100) # equal membership probabilities
    #     pxr = [p_binom[i] for i in membership]
    #     y = np.matrix([np.random.binomial(m, p) for p in pxr]).T
    #     xm = xr = np.ones((n,1))
    #
    #     # --- Evaluate log-likelihood given the initial guess ---
    #     beta0 = np.matrix(np.repeat(1., c-1))
    #     alpha0 = np.matrix(np.repeat(1., c))
    #     m = np.matrix(np.repeat(m, n)).T
    #
    #     ellk = expectedLogLikelihood(y, xm, xr, m, beta0, alpha0, beta0, alpha0)
    #
    #     # --- Check ---
    #     pi = cal_softmax(appendZeros(beta0))  # membership probabilities
    #     p = cal_sigmoid(alpha0)  # binomial probabilities


if __name__ == '__main__':
    unittest.main()