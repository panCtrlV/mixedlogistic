__author__ = 'panc'

"""
Pre-processing data and parameters.
"""
import numpy as np


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
