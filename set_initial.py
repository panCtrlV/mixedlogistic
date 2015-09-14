__author__ = 'panc'

"""
Setting initial parameters for mixed logistic
"""

import numpy as np


def set_initial(c, lxm, lxr):
    """
    Setting initial values for beta and alpha for optimization in M-step.

    :param c:
        integer, number of pre-determined hidden groups
    :param lxm:
        number of covariates for mixing probabilities (after pre-processing data)
    :param lxr:
        number of covariates for component probabilities (after pre-processing data)
    :return: (c, beta, alpha).
        beta is a (nxm+1)*(c-1) numpy matrix, where "+1" is for the intercept.
        alpha is a (nxr+1)*c numpy matrix, where "+1" is for the intercepts.
    """
    # TODO: check identifiability
    beta0 = np.random.random(lxm * (c-1)).reshape((lxm, c-1))
    alpha0 = np.random.random(lxr * c).reshape((lxr, c))
    return beta0, alpha0
