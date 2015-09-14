__author__ = 'panc'

"""
Helper functions.
"""

import math
import numpy as np


def combination(n, r):
    return math.factorial(n) / math.factorial(r) / math.factorial(n-2)


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
