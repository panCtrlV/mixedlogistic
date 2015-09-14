"""
Calculating a conditional probability for the membership of an observation
given its covariates, current estimates of paramters.
This is the E-step for the EM algorithm.
"""

from scipy.special import expit
from scipy.stats import binom
from preprocess import *


def cal_sigmoid(x):
    """
    Evaluating a sigmoid function.

    :param x:
        scalar or a list or a matrix, at which sigmoid function is evaluated.
    :return:
        scalar or a list or a matrix, depending on the input,
        containing the evaluation of the sigmoid function at given x.
    """
    return expit(x)


def cal_sigmoidForData(x, b):
    """
    Calculating sigmoid function for a given data matrix (after pre-processing) and parameters
    under logistic model.

        sigmoid(x,b) = 1 / (1 + exp(- x^T * b))

    :param x:
        numpy matrix, data matrix where each column is the observations of a covariate
    :param b:
        numpy matrix, a column matrix where the length = number of columns of x
    :return:
        numpy matrix, a column matrix which contains
    """
    xb = x * b  # a column matrix
    return cal_sigmoid(xb)

def cal_softmax(x):
    """
    Evaluating softmax function for each element in x.

    :param x:
        1-d numpy array, a softmax is evaluated for each element.

            x = (x1, x2, ..., xp)
            softmax(x1) = exp(x1) / sum{exp(xi)}
            softmax(x2) = exp(x2) / sum{exp(xi)}
            .
            .
            .
            softmax(xp) = exp(xp) / sum{exp(xi)}
    :return:
        1-d numpy array
    """
    e = np.exp(x)
    e = e / e.max()  # to avoid overflow
    return e / e.sum()


def cal_softmaxForData(x, beta):
    """

    :param x:
        numpy matrix, data matrix of size n*p.
    :param beta:
        numpy matrix, logistic regression coefficient. Each column contains the coefficients for the
        corresponding group, including those for the c^th group.
        If there are p covariates, and c groups, then beta is a p*c matrix.
        For estimation, we will force the coefficients for the c^th group to be all 0, to ensure identifiability.
    :return:
    """
    xb = x * beta
    return np.apply_along_axis(cal_softmax, axis=1, arr=xb)


def cal_binomProbForData(y, x, m, a):
    """
    Calculating binomial probability for a given data set (after pre-processing), number of trials
    and parameters under logistic model.

        Yi ~ Bin(m_i, p_i)
        logit(p_i) = x_i.T * a

    :param y:
        numpy matrix, a column matrix containing the binomial responses
    :param x:
        numpy matrix, data matrix
    :param m:
        numpy matrix, a column matrix containing the number of trials for Binomial distribution, with
        size = y
    :param a:
        numpy matrix, a column matrix or a matrix with multiple columns.
        In either case, number of rows of a = number of columns of x.
        If a has multiple columns, each column contains the logistic regression coefficients for the
        corresponding component. Then, x*a is a matrix, so is the resulting sigmoid function evaluated
        at x*a.

        The `binom.pmf` can handle inputs with different dimensions with proper broadcasting. For example,
        with y, m, and p being n*1, n*1 and n*m matrices,

            binom.pmf(y, m, p)

        will be appropriately evaluating the Binomial probability for each possible combination of y, m, and p.

    :return:
    """
    # TODO: the probability is calculated for each observation, efficiency can be improved by calculating
    # todo the probability once for a set of observations which share the same values for covariates.

    p = cal_sigmoidForData(x, a)  # a column matrix with size = y
    return np.matrix(binom.pmf(y, m, p)) # y, m, p must be of same length


def cal_condProbOfComponentForData(y, xm, xr, m, beta, alpha):
    """
    Calculating the conditional probability of the membership for each observation,
    given the data and the **previous** estimate of the parameters

    :param y:
        numpy matrix, a column matrix containing the binomial responses
    :param xm:
        numpy matrix, data matrix (after pre-processing) for modelling mixing probabilities.
    :param xr:
        numpy matrix, data matrix (after pre-processing) for modelling component probabilities.
    :param m:
        numpy matrix, a column matrix containing the number of trials for Binomial distribution,
        with size = y's size
    :param alpha:
        numpy matrix, of size (length of x^(r)) * c, where x^(r) is the component covariates.
        Each column of alpha contains the logistic regression coefficients for the corresponding component (i.e.
        hidden group).
    :param beta:
        numpy matrix, which contains the regression coefficients for the mixing probabilities.
        The size of the matrix is

            (length of x^(m)) * (c-1),

        where x^(m) is the covariates for the mixing probabilities.
        Each column of beta contains the (multinomial) logistic regression coefficients for the corresponding
        mixing probability (i.e. the probability that an observation is in the corresponding hidden group).
    :return:
        numpy matrix, of size n*c.
    """

    beta = appendZeros(beta)

    # The following two sets of probabilities are based on previous estimates of the parameters.
    pxb = np.matrix(cal_softmaxForData(xm, beta))
    pbinom = cal_binomProbForData(y, xr, m, alpha)

    p_joint = np.multiply(pxb, pbinom)
    scale = p_joint.max(axis=1)
    p_joint_scaled = p_joint / scale

    return p_joint_scaled / p_joint_scaled.sum(axis=1)