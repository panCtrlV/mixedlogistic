__author__ = 'panc'

import unittest

from mixedlogistic.set_initial import *
from data.tribolium import *
from mixedlogistic.helper import *


class TestMixedLogistic(unittest.TestCase):
    def setUp(self):
        pass


    def test_nellkForOptimization_evaluation(self):
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

        vBetaAlpha = mapOptimizationMatrices2Vector(beta, alpha)
        nellk = nellkForOptimization(vBetaAlpha, vBetaAlpha, c, y, xm, xr, m)
        print '\n', 'nellk = ', nellk, '\n'


    def test_minimize_negLogLikelihoodForOptimize_tribolium(self):
        # read data
        df = import_data()
        xm = pd.get_dummies(df.Replicate).ix[:, 2:]
        xr = pd.get_dummies(df.Species).ix[:, 1:]
        m = np.matrix(df.Total).T
        y = np.matrix(df.Remaining).T
        # pre-process data
        xm = addIntercept(xm)  # add a leading 1s column
        xr = addIntercept(xr)  # add a leading 1s column
        c = 3  # three replicates as three components
        # set initial
        beta0, alpha0 = set_initial(c, xm.shape[1], xr.shape[1])
        vBetaAlpha0 = mapOptimizationMatrices2Vector(beta0, alpha0)
        res = sp.optimize.minimize(nellkForOptimization, x0=vBetaAlpha0,
                                   args=(vBetaAlpha0,c,y,xm,xr,m), method='BFGS')

    def test_mixedLogistic_Estep_evaluation(self):
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

        vBetaAlpha = mapOptimizationMatrices2Vector(beta, alpha)
        qfn = mixedLogistic_Estep(vBetaAlpha, c, y, xm, xr, m)
        print '\n', qfn, '\n'

    def test_mixedLogistic_Estep_correctness(self):
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
        vBetaAlpha = mapOptimizationMatrices2Vector(beta, alpha)

        qfn = mixedLogistic_Estep(vBetaAlpha, c, y, xm, xr, m)
        enllk1 = qfn(vBetaAlpha)

        enllk2 = nellkForOptimization(vBetaAlpha, vBetaAlpha, c, y, xm, xr, m)

        self.assertEqual(enllk1, enllk2)

    def test_mixedLogistic_Mstep_evaluation_tribolium(self):
        # read data
        df = import_data()
        xm = pd.get_dummies(df.Replicate).ix[:, 2:]
        xr = pd.get_dummies(df.Species).ix[:, 1:]
        m = np.matrix(df.Total).T
        y = np.matrix(df.Remaining).T
        # pre-process data
        xm = addIntercept(xm)  # add a leading 1s column
        xr = addIntercept(xr)  # add a leading 1s column
        c = 3  # three replicates as three components
        # set initial
        beta0, alpha0 = set_initial(c, xm.shape[1], xr.shape[1])
        vBetaAlpha0 = mapOptimizationMatrices2Vector(beta0, alpha0)
        # E-step
        qfn = mixedLogistic_Estep(vBetaAlpha0, c, y, xm, xr, m)
        # M-step
        param, fval = mixedLogistic_Mstep(qfn, vBetaAlpha0)
        print '\n'
        print 'parameter estimates = \n', param
        print 'function value = ', fval
        print '\n'

    def test_mixedLogistic_EMtrain_evaluation_tribolium(self):
        # --- prepare data ---
        data = prepareTriboliumData()
        # --- set initial ---
        beta0, alpha0 = set_initial(data['c'], data['xm'].shape[1], data['xr'].shape[1])
        vBetaAlpha0 = mapOptimizationMatrices2Vector(beta0, alpha0)
        # vBetaAlpha0 = np.repeat(1., 15)
        # --- EM ---
        # param, fval, nIter = mixedLogistic_EMtrain(vBetaAlpha0, c, y, xm, xr, m)
        res = mixedLogistic_EMtrain(vBetaAlpha0, data['c'], data['y'], data['xm'], data['xr'], data['m'])

        # param in matrix form
        beta, alpha = mapOptimizationVector2Matrices(res['param'], data['c'], data['xm'].shape[1])
        print 'beta ='
        print beta
        print 'alpha ='
        print alpha

    def test_mixedLogistic_QNtrain(self):
        # prepare data
        data = prepareTriboliumData()
        # set initial
        beta0, alpha0 = set_initial(data['c'], data['xm'].shape[1], data['xr'].shape[1])
        vBetaAlpha0 = mapOptimizationMatrices2Vector(beta0, alpha0)

        nollk = nollkForOptimization(data['c'], data['y'], data['xm'], data['xr'], data['m'])
        res = mixedLogistic_QNtrain(nollk, vBetaAlpha0)
        print '\n'
        print 'parameter estimates =\n', res.x
        print 'minimum fval =', res.fun

        mapOptimizationVector2Matrices(res.x, data['c'], data['xm'].shape[1])

    def test_mixedLogistic_train_tribolium(self):
        # --- prepare data ---
        data = prepareTriboliumData()
        # --- set initial ---
        beta0, alpha0 = set_initial(data['c'], data['xm'].shape[1], data['xr'].shape[1])
        vBetaAlpha0 = mapOptimizationMatrices2Vector(beta0, alpha0)
        # --- Estimation ---
        res = mixedLogistic_train(vBetaAlpha0, data['c'], data['y'], data['xm'], data['xr'], data['m'])
        mapOptimizationVector2Matrices(res.x, data['c'], data['xm'].shape[1])


if __name__ == "__main__":
    unittest.main()