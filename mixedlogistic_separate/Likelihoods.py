import numpy as np
from scipy.stats import binom

from mixedlogistic_separate.Math import *
from mixedlogistic_separate.Data import *


def posteriorHiddenMemberProbs(data, params_old):
    """
    :type data: Data
    :param data:
    :type params_old: Paramters
    :param params_old:
    :return:
    """
    n = data.n
    m = data.m

    memberProbs = softmaxForData(data.Xm, params_old.Alpha, params_old.a, True)
    choosingProbs = sigmoidForData(data.Xr, params_old.Beta, params_old.b)
    if isinstance(m, int):
        if m == 1:  # binary response
            observedLikelihoods = np.abs((1. - data.y).reshape(n, 1) - choosingProbs)
        else:
            observedLikelihoods = binom.pmf(data.y.reshape(n, 1),
                                            np.repeat(m, n).reshape(n, 1),
                                            choosingProbs)
    else:
        raise("Function is not yet implemented for non-integer 'm'.")

    jointProbs = np.multiply(memberProbs, observedLikelihoods)
    scales = jointProbs.max(axis=1).reshape(n, 1)
    scaledJointProbs = jointProbs / scales
    return scaledJointProbs / scaledJointProbs.sum(axis=1).reshape(n, 1)


def observedLogLikelihood(data, params):
    """
    :type data: Data
    :param data:
    :type params: Paramters
    :param params:
    :return:
    """
    n = data.n
    m = data.m

    memberProbs = softmaxForData(data.Xm, params.Alpha, params.a, True)
    choosingProbs = sigmoidForData(data.Xr, params.Beta, params.b)
    if isinstance(m, int):
        if m == 1:
            observedLikelihoods = np.abs((1. - data.y).reshape(n, 1) - choosingProbs)
        else:
            observedLikelihoods = binom.pmf(data.y.reshape(n, 1),
                                            np.repeat(m, n).reshape(n, 1),
                                            choosingProbs)
    else:
        raise("Function is not yet implemented for non-integer 'm'.")

    jointProbs = np.multiply(memberProbs, observedLikelihoods)
    return np.log(jointProbs.sum(axis=1)).sum()


def negObservedLogLikelihood(data, params):
    return -observedLogLikelihood(data, params)


def generateNegQAndGradientFunctions(data, params_old):
    """

    :type data: Data
    :param data:
    :type params_old: Paramters
    :param params_old:
    :return:
    """
    # number of leading terms in vparams are intercepts
    hasHiddenIntercepts = params_old.hasHiddenIntercepts

    if hasHiddenIntercepts:
        offset = params_old.Alpha.shape[1]
    else:
        offset = 0

    dxm = params_old.dxm  # number of hidden layer covariates

    hasObservedIntercepts = params_old.hasObservedIntercepts

    m = data.m
    n = data.n

    weights = posteriorHiddenMemberProbs(data, params_old)

    def negHiddenLayerQ(vparams):
        # map flattened paramters to matrix
        a, Alpha = params_old.getHiddenParametersFromFlatInput(vparams)  # with new flatten order

        logMemberProbs = softmaxForData(data.Xm, Alpha, a, returnLog=True)
        return -np.multiply(weights, logMemberProbs).sum()

    def negObservedLayerQj(vparams, j):
        bj, betaj = params_old.getObservedParametersFromFlatInput_j(vparams)

        choosingProbsj = sigmoidForData(data.Xr, betaj, bj)
        if isinstance(m, int):
            if m == 1:
                observedLikelihoods = np.abs((1. - data.y) - choosingProbsj)
            else:
                observedLikelihoods = binom.pmf(data.y, np.repeat(m, n), choosingProbsj)
        else:
            raise NotImplementedError("** Function is not yet implemented for non-integer 'm'. "
                                      "We cannot handle non-equal trial numbers. **")

        return -weights[:, j].dot(np.log(observedLikelihoods))

    def negHiddenLayerQ_grad(vparams):
        # map flattened paramters to matrix
        a, Alpha = params_old.getHiddenParametersFromFlatInput(vparams)

        memberProbs = softmaxForData(data.Xm, Alpha, a)
        qMinusPi = weights[:, :-1] - memberProbs[:, :-1]
        # gradAlpha = data.Xm.T.dot(qMinusPi).flatten(order='F')
        gradAlpha = data.Xm.T.dot(qMinusPi)  # new flatten order
        if hasHiddenIntercepts:
            gradIntercepts = qMinusPi.sum(axis=0)
            # return -np.hstack([gradIntercepts, gradAlpha])
            return -np.vstack([gradIntercepts, gradAlpha]).flatten(order='F')  # new flatten order
        else:
            # return -gradAlpha
            return -gradAlpha.flatten(order='F')  # new flatten order

    def negObservedLayerQj_grad(vparams, j):
        # prepare parameters
        bj, betaj = params_old.getObservedParametersFromFlatInput_j(vparams)

        choosingProbs = sigmoidForData(data.Xr, betaj, bj)
        # print choosingProbs
        weightsTimesYMinusMP = weights[:,j] * (data.y - m * choosingProbs)
        # print weightsTimesYMinusMP
        gradBetaj = data.Xr.T.dot(weightsTimesYMinusMP)
        if hasObservedIntercepts:
            gradInterceptj = weightsTimesYMinusMP.sum()
            return -np.hstack([gradInterceptj, gradBetaj])
        else:
            return -gradBetaj

    return negHiddenLayerQ, negObservedLayerQj, negHiddenLayerQ_grad, negObservedLayerQj_grad