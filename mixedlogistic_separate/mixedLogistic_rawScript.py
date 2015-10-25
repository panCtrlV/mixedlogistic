import numpy as np
from scipy.special import expit
from scipy.misc import logsumexp
from scipy.stats import binom
from scipy import optimize

from mixedlogistic_separate.Data import *
from mixedlogistic_separate.Parameters import *

"""
Data Formats:

c: number of hidden groups
k == 2: number of response categories
m: number of trials, same for all samples
Xm: hidden layer covariates matrix
Xr: observed layer covariates matrix
Alpha: coefficients for hidden layer covariates
Beta: coefficient for observed layer covariates
a: hidden layer intercepts (if needed)
b: observed layer intercepts (if needed)
y: response vector
"""

eps = 1e-6


##############################################
# Classes to encapsulate parameters and data #
##############################################

# Moved to Parameters.py and Data.py

##################
# Math functions #
##################
def sigmoid(x):
    return expit(x)


def sigmoidForData(X, coef, intercept=None, returnFull=False):
    if coef.ndim == 1:  # coef is 1d array
        if intercept is None:
            linearTerms = X.dot(coef)
        else:
            if isinstance(intercept, float):
                linearTerms = X.dot(coef) + intercept  # intercept should be a float
            else:
                raise("'coef' is 1-dimensional, which requires a scalar intercept."
                      "Non-scalar intercept is not yet implemented.")

        sigmoids = sigmoid(linearTerms)
        if returnFull:
            return np.vstack((sigmoids, 1-sigmoids)).T
        else:
            return sigmoids
    else:  # coef is 2d array
        if intercept is None:
            linearTerms = X.dot(coef)
        else:
            if len(intercept) == coef.shape[1]:
                linearTerms = X.dot(coef) + intercept
            else:
                ValueError("The shapes of 'coef' and 'intercept' are not conformable. "
                           "'coef' has shape %s and 'intercept' has shape %s." % (coef.shape, intercept.shape))
        return sigmoid(linearTerms)


def softmax(x, addDefaultBase=True, axis=1, returnLog=False):
    # x is a 1d or 2d array
    if x.ndim == 1:
        if addDefaultBase:
            x = np.append(x, 0)
        logSumExp = logsumexp(x)
        logSoftMax = x - logSumExp
        if returnLog:
            return logSoftMax
        else:
            return np.exp(logSoftMax)
    else:
        if addDefaultBase:
            x = np.append(x, np.zeros((x.shape[0], 1)), axis=1)
        logSumExp = logsumexp(x, axis, keepdims=True)
        logSoftMax = x - logSumExp
        if returnLog:
            return logSoftMax
        else:
            return np.exp(logSoftMax)


def softmaxForData(X, Coef, intercepts=None, addDefaultBase=True, returnLog=False):
    if intercepts is None:
        linearTerms = X.dot(Coef)
    else:
        linearTerms = X.dot(Coef) + intercepts

    return softmax(linearTerms, addDefaultBase, returnLog=returnLog)


##################
# Data Simulator #
##################

# Moved to dataSimulator.py


####################################################
# Posterior (Conditional) Membership Probabilities #
####################################################
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


########################
# Likelihood Functions #
########################
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
        # if offset:
        #     a = vparams[:offset]
        #     Alpha = vparams[offset:].reshape((dxm, offset), order='F')
        # else:
        #     a = None
        #     Alpha = vparams.reshape((dxm, offset), order='F')

        a, Alpha = params_old.getHiddenParametersFromFlatInput(vparams)  # with new flatten order

        logMemberProbs = softmaxForData(data.Xm, Alpha, a, returnLog=True)
        return -np.multiply(weights, logMemberProbs).sum()

    def negObservedLayerQj(vparams, j):
        # if hasObservedIntercepts:
        #     bj = vparams[0]  # first parameter is the coefficient
        #     betaj = vparams[1:]
        # else:
        #     betaj = vparams

        bj, betaj = params_old.getObservedParametersFromFlatInput_j(vparams)

        choosingProbsj = sigmoidForData(data.Xr, betaj, bj)
        if isinstance(m, int):
            if m == 1:
                observedLikelihoods = np.abs((1. - data.y) - choosingProbsj)
            else:
                observedLikelihoods = binom.pmf(data.y, np.repeat(m, n), choosingProbsj)
        else:
            raise("Function is not yet implemented for non-integer 'm'.")

        return -weights[:, j].dot(np.log(observedLikelihoods))

    def negHiddenLayerQ_grad(vparams):
        # map flattened paramters to matrix
        # if offset:
        #     a = vparams[:offset]
        #     Alpha = vparams[offset:].reshape((dxm, offset), order='F')
        # else:
        #     a = None
        #     Alpha = vparams.reshape((dxm, offset), order='F')

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
        # if hasObservedIntercepts:
        #     bj = vparams[0]  # first parameter is the coefficient
        #     betaj = vparams[1:]
        # else:
        #     bj = None
        #     betaj = vparams

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


###############
# EM Training #
###############
def trainEM(data, params0, optMethod='L-BFGS-B', maxIter=500):
    """

    :type data: Data
    :param data:
    :type params0: Paramters
    :param params0:
    :type optMethod: str or callable
    :param optMethod:
    :type maxIter: int
    :param maxIter:
    :return:
    """
    c = params0.c
    dxm = params0.dxm
    hasHiddenIntercepts = params0.hasHiddenIntercepts
    hasObservedIntercepts = params0.hasObservedIntercepts

    # if hasHiddenIntercepts:
    #     offset = params0.Alpha.shape[1]

    params_old = params0
    nollk_old = negObservedLogLikelihood(data, params_old)

    nIter = 0
    while nIter < maxIter:
        nIter += 1
        print("\nCurrent iteration: %d" % nIter)

        # --- E-step ---
        fNegQ1, fNegQ2j, gradNegQ1, gradNegQ2j = generateNegQAndGradientFunctions(data, params_old)
        # --- M-step ---
        # print("\t Optimizing hidden layer ...")
        vHiddenLayerParameters_old = params_old.flattenHiddenLayerParameters()
        updateNegQ1 = optimize.minimize(fNegQ1, vHiddenLayerParameters_old,
                                        method=optMethod, jac=gradNegQ1)

        # ... check fNegQ1 to accommodate the now order of the flattened hidden parameters ...

        # if hasHiddenIntercepts:
        #     a = updateNegQ1.x[:offset]
        #     Alpha = updateNegQ1.x[offset:].reshape((dxm, offset), order='F')
        # else:
        #     a = None
        #     Alpha = updateNegQ1.x[offset:].reshape((dxm, offset), order='F')

        a, Alpha = params_old.getHiddenParametersFromFlatInput(updateNegQ1.x)

        # updateNegQ2 = []
        bjs = []  # modification
        betajs = []  # modification
        # print("\t Optimizing observed layer ...")
        for j in xrange(c):
            # print("\t\t component %d ..." % j)
            vObservedLayerParameterj_old = params_old.flattenObservedLayerParametersj(j)
            updateNegQ2j = optimize.minimize(fNegQ2j, vObservedLayerParameterj_old,
                                             args=(j, ), method=optMethod, jac=gradNegQ2j)
            # updateNegQ2.append(updateNegQ2j)
            bjs.append(updateNegQ2j.x[0])  # modification
            betajs.append(updateNegQ2j.x[1])  # modification

        # if hasObservedIntercepts:
        #     b = np.array([params.x[0] for params in updateNegQ2])
        #     Beta = np.vstack([params.x[1:] for params in updateNegQ2]).T
        # else:
        #     b = None
        #     Beta = np.vstack([params.x for params in updateNegQ2]).T

        b = None if all(bj is None for bj in bjs) else np.array(bjs)
        Beta = np.vstack(betajs).T

        # print '\t updated parameters: '
        # print '\t\t Alpha = '
        # print Alpha
        # if hasHiddenIntercepts:
        #     print "\t\t with intercepts = "
        #     print a
        # print '\t\t Beta = '
        # print Beta
        # if hasObservedIntercepts:
        #     print "\t\t with intercepts = "
        #     print b

        params = Paramters(Alpha, Beta, a, b)
        paramsDiffNorm = params.normedDifference(params_old)
        print '\t Changes in parameters (L2 norm) = ', paramsDiffNorm

        nollk = negObservedLogLikelihood(data, params)
        print("\t Current negative observed loglikelihood = %f" % nollk)

        nollkDiff = nollk - nollk_old
        print '\t Change in negative observed loglikelihood =', nollkDiff

        if np.abs(nollkDiff) < eps and paramsDiffNorm < eps:
            break
        else:
            params_old = params
            nollk_old = nollk

    return dict(params=params, nit=nIter, nollk=nollk)


####################
# Simulation Study #
####################

# # === Random initials (simulation setting) ====
# # For the same sample, estimating with different initial values
# mixedLogisticSimulator = MixedLogisticDataSimulator(1000, 3, 3, 3)
# mixedLogisticSimulator.simulate(25)
# data = mixedLogisticSimulator.data
# params = mixedLogisticSimulator.parameters
#
# resCollection = []
# seeds = np.arange(100)
# for seed in seeds:
#     np.random.seed(seed)
#     Alpha0 = np.random.uniform(size=params.Alpha.shape)
#     Beta0 = np.random.uniform(size=params.Beta.shape)
#     a0 = np.random.uniform(size=params.a.shape)
#     b0 = np.random.uniform(size=params.b.shape)
#     params0 = Paramters(Alpha0, Beta0, a0, b0)
#
#     res = trainEM(data, params0, maxIter=10000)
#     resCollection.append(res)
#
# # Simulation was run on sever, and the results are pickled.
# # Now, unpickle the results.
# import pickle
#
# estParameters = pickle.load(open("/Users/panc25/sshfs_map2/results.pickle"))
# compiledParameters = np.vstack(x.flatten() for x in estParameters)
# # compiledParameters[0].shape
# # compiledParameters[:, [0,2]]
#
# # Plot the simulation results
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))
#
# # labels = list([r'$\a_1$', r'$\alpha_{11}$', r'$\alpha_{12}$', r'$\alpha_{13}$'])
# axes[0, 0].boxplot(compiledParameters[:, [0, 2, 3, 4]], showmeans=True)
# axes[0, 0].set_title('Hidden Layer, Group 1')
#
# axes[0, 1].boxplot(compiledParameters[:, [1, 5, 6, 7]], showmeans=True)
# axes[0, 1].set_title("Hidden Layer, Group 2")
#
# axes[1, 0].boxplot(compiledParameters[:, [8, 9, 10, 11]], showmeans=True)
# axes[1, 0].set_title('Observed Layer, Group 1')
#
# axes[1, 1].boxplot(compiledParameters[:, [12, 13, 14, 15]], showmeans=True)
# axes[1, 1].set_title('Observed Layer, Group 2')
#
# axes[1, 2].boxplot(compiledParameters[:, [16, 17, 18, 19]], showmeans=True)
# axes[1, 2].set_title('Observed Layer, Group 3')
#
# plt.show()


# # === Random samples (simulation setting) ===
# # Estimating for different sample using the same initial value
# simulator = MixedLogisticDataSimulator(1000, 3, 3, 3)
# simulator.simulate(25)
# params = simulator.parameters
#
# # Fix initial values
# np.random.seed(25)
# Alpha0 = np.random.uniform(size=params.Alpha.shape)
# Beta0 = np.random.uniform(size=params.Beta.shape)
# a0 = np.random.uniform(size=params.a.shape)
# b0 = np.random.uniform(size=params.b.shape)
# params0 = Paramters(Alpha0, Beta0, a0, b0)
#
# # Simulate random data and estimate
# collectResults = []
# # seeds = np.arange(100) + 1000
# seeds = np.arange(1000, step=10)
# for seed in seeds:
#     mixedLogisticSimulator = MixedLogisticDataSimulator(1000, 3, 3, 3)
#     mixedLogisticSimulator.simulate(seed)
#     data = mixedLogisticSimulator.data
#
#     res = trainEM(data, params0, maxIter=10000)
#     collectResults.append(res)
#
#
# # Simulation is fired up on server, the results are pickled.
# # Now, unpickle the results.
# import pickle
# collectedParameters = pickle.load(open('/Users/panc25/sshfs_map2/script2.pickle', 'rb'))
# flattenedParameters = np.vstack(x.flatten() for x in collectedParameters)
#
# # Plot the simulation results
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))
#
# # labels = list([r'$\a_1$', r'$\alpha_{11}$', r'$\alpha_{12}$', r'$\alpha_{13}$'])
# axes[0, 0].boxplot(flattenedParameters[:, [0, 2, 3, 4]], showmeans=True)
# axes[0, 0].set_title('Hidden Layer, Group 1')
#
# axes[0, 1].boxplot(flattenedParameters[:, [1, 5, 6, 7]], showmeans=True)
# axes[0, 1].set_title("Hidden Layer, Group 2")
#
# axes[1, 0].boxplot(flattenedParameters[:, [8, 9, 10, 11]], showmeans=True)
# axes[1, 0].set_title('Observed Layer, Group 1')
#
# axes[1, 1].boxplot(flattenedParameters[:, [12, 13, 14, 15]], showmeans=True)
# axes[1, 1].set_title('Observed Layer, Group 2')
#
# axes[1, 2].boxplot(flattenedParameters[:, [16, 17, 18, 19]], showmeans=True)
# axes[1, 2].set_title('Observed Layer, Group 3')
#
# plt.show()
#
# # Further explore each parameter
# alpha11 = flattenedParameters[:, 2]  # \alpha_{11}
# alpha13 = flattenedParameters[:, 4]  # \alpha_{13}
#
# beta10 = flattenedParameters[:, 8]
# beta13 = flattenedParameters[:, 11]
# beta21 = flattenedParameters[:, 13]
# beta22 = flattenedParameters[:, 14]
# beta31 = flattenedParameters[:, 17]
# beta30 = flattenedParameters[:, 16]
#
# plt.figure()
# plt.hist(alpha11)
# plt.show()
#
# plt.figure()
# plt.boxplot(np.sort(alpha11)[4:80])
# plt.show()
#
# plt.figure()
# plt.boxplot(beta10)
# plt.show()
#
# plt.figure()
# plt.boxplot(np.sort(beta13)[:-10])
# plt.show()
#
# plt.figure()
# plt.boxplot(np.sort(beta21)[:-5])
# plt.show()
#
# plt.figure()
# plt.boxplot(np.sort(beta22)[:-8])
# plt.show()
#
# plt.figure()
# plt.boxplot(np.sort(beta30)[3:90])
# plt.show()
#
# plt.figure()
# plt.boxplot(np.sort(beta31)[:-10])
# plt.show()
#
#
# collectedParameters = pickle.load(open())
#
#
# # === 10-19-2015 ====
# # Simulate data
# simulator = MixedLogisticDataSimulator(15000, 3, 3, 3)
# simulator.simulate(100)
# data = simulator.data
# params = simulator.parameters
#
# # Fix initial values
# np.random.seed(25)
# Alpha0 = np.random.uniform(size=params.Alpha.shape)
# Beta0 = np.random.uniform(size=params.Beta.shape)
# a0 = np.random.uniform(size=params.a.shape)
# b0 = np.random.uniform(size=params.b.shape)
# params0 = Paramters(Alpha0, Beta0, a0, b0)
#
# # train
# res = trainEM(data, params, maxIter=10000)
#
# res.a
# params.a
#
# res.Alpha
# params.Alpha
#
# res.b
# params.b
#
# res.Beta
# params.Beta
#
#
# # likelihood
# negObservedLogLikelihood(data, res)
# negObservedLogLikelihood(data, params)
#
# negObservedLogLikelihood(data, params0)
# # observedLogLikelihood(data, res)
# # observedLogLikelihood(data, params)
#
# res.a