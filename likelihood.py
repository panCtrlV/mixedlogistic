__author__ = 'panc'

"""
Evaluating the expectation of the complete log-likelihood function.
The expectation is taken wrt the conditional probability of the membership
given the current estimate of the parameters.
This corresponds to the M-step in the EM algorithm.
"""

from cal_condprob import *
from helper import mapOptimizationVector2Matrices


# TODO: need the log-likelihood function (for observed data) for the quasi-Newton optimization phase.
def ollk(y, xm, xr, m, beta, alpha):
    """
    Observed log-likelihood function.
        L(c, xm, xr, y, m | p, \pi) = \sum_{j=1}^c {p_j * Bin(y|m, \pi_j)}

    :param y:
    :param xm:
    :param xr:
    :param m:
    :param beta:
    :param alpha:
    :return:
    """
    pxb = cal_softmaxForData(xm, appendZeros(beta))
    pbinom = cal_binomProbForData(y, xr, m, alpha)
    return np.log(np.multiply(pxb, pbinom).sum(axis=1)).sum()
    # return np.log(np.multiply(pxb, pbinom)).sum()  # this is WRONG !


def nollkForOptimization(c, y, xm, xr, m):
    """
    Wrapper for the observed log-likelihood function for optimization (minimization) purpose.
    It returns a NEGATIVE log-likelihood function callable.

    :param c:
    :param y:
    :param xm:
    :param xr:
    :param m:
    :return: callable
    """
    def nollk(vBetaAlpha):
        """
        :param vBetaAlpha: 1d numpy array
        """
        lxm = xm.shape[1]
        beta, alpha = mapOptimizationVector2Matrices(vBetaAlpha, c, lxm)
        return  - ollk(y, xm, xr, m, beta, alpha)
    return nollk


def ellk(y, xm, xr, m, beta_old, alpha_old, beta, alpha):
    """
    Expected log-likelihood function for complete data, given the parameter estimates from
    the previous. This is the optimization target for M-step in an EM iteration.

    :param y:
        numpy matrix, a column matrix containing the binomial responses
    :param xm:
        numpy matrix, data matrix (after pre-processing) for modelling mixing probabilities.
    :param xr:
        numpy matrix, data matrix (after pre-processing) for modelling component probabilities.
    :param m:
        numpy matrix, a column matrix containing the number of trials for Binomial distribution,
        with size = y's size
    :param beta_old:
        numpy matrix.
        It contains the coefficient estimates for the mixing covariates from the previous step.
        The size of the matrix is

            (length of x^(m)) * (c-1),

        where x^(m) is the covariates for the mixing probabilities.
        The matrix does NOT contain the 0s for the baseline group.
        Each column of beta contains the (multinomial) logistic regression coefficients for the corresponding
        mixing probability (i.e. the probability that an observation is in the corresponding hidden group).
    :param alpha_old:
        numpy matrix.
        It contains the coefficient estimates for the component covariates from the previous step.
        The size of the matrix is

            (length of x^(r)) * c,

        where x^(r).
        Each column of alpha contains the logistic regression coefficients for the corresponding component (i.e.
        hidden group).
    :param beta:
        numpy matrix.
        It contains the current coefficient values for the mixing covariates.
        The matrix does NOT contain the 0s for the baseline group.
    :param alpha:
        numpy matrix.
        It contains the current coefficient values for the component covariates.
    :return:
    """
    # pz depends on the previous estimates
    pz = cal_condProbOfComponentForData(y, xm, xr, m, beta_old, alpha_old)
    # the following two set of probabilities depend on the current parameter inputs
    pxb = np.matrix(cal_softmaxForData(xm, appendZeros(beta)))
    pbinom = cal_binomProbForData(y, xr, m, alpha)

    q1 = np.multiply(pz, np.log(pxb)).sum()
    q2 = np.multiply(pz, np.log(pbinom)).sum()
    return q1 + q2

def nellkForOptimization(vBetaAlpha, vBetaAlpha_old, c, y, xm, xr, m):
    """
    Wrapper of eval_logLikelihood.expectedLogLikelihood function, for being called in optimization.

    :param vBetaAlpha:
        1d numpy array.
        It is the vectorization of the two parameter matrices beta and alpha.
        For optimization, it will be given an initial value, which is the coefficient estimates from
        the previous step.
    :param vBetaAlpha_old:
        1d numpy array.
        It is the vectorization of the two parameter matrices, beta and alpha, estimated
        from the previous step.
    :param c:
        integer. Number of mixing components.
    :param y:
        numpy matrix. A column matrix containing the binomial responses
    :param xm:
        numpy matrix. Data matrix (after pre-processing) for modelling mixing probabilities.
    :param xr:
        numpy matrix, data matrix (after pre-processing) for modelling component probabilities.
    :param m:
        numpy matrix, a column matrix containing the number of trials for Binomial distribution,
        with size = y's size
    :return:
    """
    lxm = xm.shape[1]
    beta_old, alpha_old = mapOptimizationVector2Matrices(vBetaAlpha_old, c, lxm)
    beta, alpha = mapOptimizationVector2Matrices(vBetaAlpha, c, lxm)
    return -ellk(y, xm, xr, m, beta_old, alpha_old, beta, alpha)