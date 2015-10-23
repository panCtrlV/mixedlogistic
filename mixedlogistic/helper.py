__author__ = 'panc'

"""
Helper functions.
"""

import math
import numpy as np


def combination(n, r):
    return math.factorial(n) / math.factorial(r) / math.factorial(n-2)


def mapMatrix2Vector(M):
    return M.flatten(order='F')


def mapVector2Matrix(v, nrow):
    try:
        return v.reshape(nrow, len(v) / nrow, order='F')
    except:
        raise ValueError(" Dimension don't align. "
                         "'v' has %d values, which cannot be reshaped into "
                         "%d rows." % (len(v), nrow))


def mapOptimizationMatrices2Vector(beta, alpha):
    """
    Converting two matrices, beta (of size lxm * (c-1)) and alpha (of size lxr * c),x to a vector,
    because Scipy.optimize.minimize requires a 1D vector to optimize.

    lxm and lxr (after pre-processing, i.e. 1 is added for the intercept) are the dimensions
    for the covariates for the mixing probabilities and component logistic regressions respectively.

    The reference can be seen at:
        http://stackoverflow.com/questions/31292374/how-do-i-put-2-matrix-into-scipy-optimize-minimize

    :param beta:
        numpy matrix, which contains the logistic regression coefficients.
        Each column contains the coefficients for the corresponding group, including those for the c^th group.
        If there are p covariates, and c groups, then beta is a p*c matrix.
        For estimation, we will force the coefficients for the c^th group to be all 0, to ensure identifiability.
    :param alpha:
        numpy matrix, of size (length of x^(r)) * c, where x^(r) is the component covariates.
        Each column of alpha contains the logistic regression coefficients for the corresponding component (i.e.
        hidden group).
    :return:
        1d numpy array, which contains the vectorization of the two parameter matrices.
    """
    vbeta = beta.flatten(order='F')  # column stack of beta, length = lxm * (c-1)
    valpha = alpha.flatten(order='F')  # column stack of alpha, length = lxr * c
    vbetaAlpha = np.concatenate((vbeta, valpha), axis=1)
    return np.squeeze(np.asarray(vbetaAlpha))


def mapOptimizationVector2Matrices(vBetaAlpha, c, lxm):
    """
    Inverse of mapOptimizationMatrices2Vector.

    :param vBetaAlpha:
        1d numpy array.
    :param c:
        integer, number of components / hidden groups
    :param lxm:
        integer, dimension of covariates for component probabilities (after pre-processing)
    :return:
        tuple of two matrices, (beta, alpha)
    """
    n = len(vBetaAlpha)
    d = lxm * (c-1)
    vbeta = vBetaAlpha[:d]
    valpha = vBetaAlpha[d:]
    beta = np.matrix(vbeta).reshape((lxm, c-1), order='F')
    alpha = np.matrix(valpha).reshape(((n-d)/c, c), order='F')
    return beta, alpha


def addBaselineCoefficientsAsZeros(coefMtx):
    """
    Append a 0 column to the input, which would be the coefficients for the baseline group.

    :param coefMtx:
        numpy matrix. If the covariate dimension is p, and number of components is c, then
        the matrix is of size p * (c-1).
    :return:
        numpy matrix, of size p * c, where the last column contains all 0s.
    """
    m, n = coefMtx.shape
    fullMtx = np.zeros((m, n+1))
    fullMtx[:,:-1] = coefMtx
    return fullMtx

def addIntercept(x):
    """
    Adding the leading 1's column for a given data matrix.

    :param x: numpy matrix. data matrix.
    :return: numpy matrix, x augmented with a leading column of ones.
    """
    # TODO: if there is no covariate, then return a one's column

    n, p = x.shape
    xfull = np.ones((n, p+1))
    xfull[:, 1:] = x
    return np.matrix(xfull)


def appendZeros(beta):
    """
    Appending a zero column at the ending of the parameter matrix.

    :param beta:
        numpy matrix, parameter matrix. If there are c hidden groups / components, and the
        dimension of the covariates for the mixing probabilities is pm, then the matrix is
        of size pr*(c-1). That's why we need to append a zero column at the end as the parameters
        for the c^th component. This is one way to ensure identifiability for the logistic model.
    :return:
    """
    return np.hstack((beta, np.zeros((beta.shape[0], 1))))
