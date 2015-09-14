# __author__ = 'panc'

import scipy as sp
from likelihood import *
from set_initial import *


eps1 = 1e-3
eps2 = 1e-3


def mixedLogistic_Estep(vBetaAlpha_old, c, y, xm, xr, m):
    """
    Wrapper for nellkForOptimization. The intent is to make the function call easier in M-step.

    :param vBetaAlpha_old: 1d numpy array
        Parameter estimates from the previous step.
        It used to evaluate the conditional probabilities used in the E-step.
    :param c: int
        Number of mixing components.
    :param y: numpy matrix
        Column matrix containing the regression response.
    :param xm: numpy matrix
        It contains the regression covariates for the mixing probabilities
        before pre-processing, i.e. no leading 1s column to accommodate intercept.
    :param xr: numpy matrix
        It contains the regression covariates for the component probabilities
        before pre-processing, i.e. no leading 1s column to accommodate intercept.
    :param m: numpy matrix
        A column matrix containing the number of trials for each observation.
    :return nellk: callable
        The negative expected log-likelihood function given the parameter estimates from
        the last step. It will be used as an input in the subsequent M-step.

    :type nellk: callable
    """
    # TODO: If the number of trials for all observations remains the same, the program should be able to
    # todo handle m as an integer.
    def nellk(vBetaAlpha):
        return nellkForOptimization(vBetaAlpha, vBetaAlpha_old, c, y, xm, xr, m)
    return nellk


def mixedLogistic_Mstep(nellk, vBetaAlpha0):
    """
    M-step in an EM iteration. BFGS method is chosen for the minimization.

    :param nellk:
        callable.
        The negative expected log-likelihood to be minimized.
        It is a returned object from `mixedLogistic_Estep` function. The function already encapsulates the
        information of data, nellk functional form, and parameter estimates from the previous step.
    :param vBetaAlpha0:
        1d numpy array.
        Initial values of the parameters for the optimization.
    :return:
    """
    res = sp.optimize.minimize(nellk, x0=vBetaAlpha0, method='BFGS')
    return res.x, res.fun


def mixedLogistic_EMtrain(param0, c, y, xm, xr, m):
    """
    Training a mixed logistic model using EM algorithm.

    :param param0:
    :param c:
    :param y:
    :param xm:
    :param xr:
    :param m:
    :return:
    """
    lxm = xm.shape[1]  # xm is obtained after pre-processing
    param_old = param0  # initial guess of the parameters

    # nellk0 = mixedLogistic_Estep(param_old, c, y, xm, xr, m)
    # optfval_old = nellk0(param_old)

    # --- Using negative observed log-likelihood as one of the convergence criteria ---
    nollk = nollkForOptimization(c, y, xm, xr, m) # negative observed log-likelihood
    nollkfval_old = nollk(param_old)

    nIter = 0  # counter for number of EM iterations

    print '--- Training via EM ---'
    while True:
        nIter += 1

        print '\n', 'Iter', nIter, ':'

        beta_old, alpha_old = mapOptimizationVector2Matrices(param_old, c, lxm)
        #--- E-step ---
        nellk = mixedLogistic_Estep(param_old, c, y, xm, xr, m)
        #--- M-step ---
        param, optfval = mixedLogistic_Mstep(nellk, param_old)
        beta, alpha = mapOptimizationVector2Matrices(param, c, lxm)

        print 'parameters update ='
        print param
        print 'expected nLogLikelihood =', optfval

        betaNorm = np.linalg.norm(beta - beta_old)
        alphaNorm = np.linalg.norm(alpha - alpha_old)
        # optfvaldiff = np.abs(optfval - optfval_old)

        nollkfval = nollk(param)
        print 'negative observed loglikelihood =', nollkfval
        nollkfvaldiff = nollkfval - nollkfval_old
        print 'negative observed loglikelihood change =', nollkfvaldiff
        nollkfvalabsdiff = np.abs(nollkfvaldiff)

        # print 'expected nLogLikelihood change =', fval - fval_old

        # if betaNorm < eps1 and alphaNorm < eps1 and fvaldiff < eps2:
        if betaNorm < eps1 and alphaNorm < eps1 and nollkfvalabsdiff < eps2:
            break
        else:
            param_old = param
            # fval_old = fval
            nollkfval_old = nollkfval

    # return {'param': param, 'fval': fval, 'nit': nIter}
    return {'param':param, 'nollk':nollkfval, 'nit':nIter}


def mixedLogistic_QNtrain(nollk, param0):
    """
    Minimizing negative observed log-likelihood using quasi-Newton (BFGS) method directly.

    :param nollk: callable
    :param param0:
    :return:
    """
    print '--- Training via Quasi-Newton ---'
    res = sp.optimize.minimize(nollk, x0 = param0, method='BFGS')
    return res


def mixedLogistic_train(param0, c, y, xm, xr, m):
    resEM = mixedLogistic_EMtrain(param0, c, y, xm, xr, m)
    nollk = nollkForOptimization(c, y, xm, xr, m)
    # res = sp.optimize.minimize(nollk, x0=resEM['param'], method='BFGS')
    res = mixedLogistic_QNtrain(nollk, resEM['param'])
    return res
