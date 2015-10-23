import numpy as np
from scipy.special import expit
from scipy.misc import logsumexp
from scipy.stats import binom
from scipy import optimize


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
class Paramters(object):
    """
    Encapsulate the parameters needed for a mixed logistic regression model.
    In the hidden layer, the last group is assumed to be the base group whose
    parameters are forced to be 0s.
    In the observed layer, the response is assumed to be binomial.
    """
    def __init__(self, Alpha, Beta, a=None, b=None):
        """
        Construct a new Parameters object.

        :type Alpha: 2d numpy array
        :param Alpha: parameter matrix for hidden layer covariates. Since the last group is
                      assumed to be the base group, its parameters are forced to be 0s and
                      excluded from the parameter matrix. Each column is corresponding to one
                      group.

        :type Beta: 2d numpy array
        :param Beta: parameter matrix for observed layer covariates. Since the response is
                     binomial (resp. binary), there is only one binomial (resp. bernoulli)
                     probability to be modeled by a sigmoid function (equivalently logit as
                     the link function) for one hidden group. Each column is corresponding to
                     the parameters of a sigmoid.

        :type a: None or 1d numpy array
        :param a: hidden layer intercepts. The default value is None, when the intercept is
                  not used in modelling a logit as linear regression. When it is not None,
                  intercept is activated in the regression. Since the last group is assumed to
                  be the base group and its intercept is forced to be 0, the size of the array
                  should be one fewer than the number of hidden groups.

        :type b: None or 1d numpy array
        :param b: observed layer intercepts. The default value is None, when the intercept is
                  not used in a logistic regression. When it is not None, each logistic regression
                  in a hidden group requires one intercept, so the size of the array is the
                  same as the number of hidden groups.

        :return: nothing
        """
        self.Alpha = Alpha
        self.Beta = Beta
        self.a = a  # hidden layer intercepts
        self.b = b  # observed layer intercepts
        self.hasHiddenIntercepts = a is not None
        self.hasObservedIntercepts = b is not None
        self.dxm = Alpha.shape[0]
        self.dxr, self.c = Beta.shape  # number of hidden groups

    def getParameters(self, layer, groupNumber):
        """
        By specifying the layer and hidden group number, it returns the
        logistic regression coefficients for the specified layer and group.

        :type layer: int
        :param layer: layer number. 1 - hidden, 2 - observed.

        :type groupNumber: int
        :param groupNumber: hidden group number. Group index starts from 1.
                            The maximum group number equals the total number of

        :return: parameter list for the specified layer and group.
        """
        if groupNumber <= 0 or groupNumber > self.c:
            raise ValueError("** 'groupNumber' is out of bounds. **")
        else:
            if layer == 1:  # hidden layer
                if self.a is None:
                    print "** Hidden layer has no intercept. **"
                    if groupNumber == self.c:
                        return np.zeros(self.dxm)
                    else:
                        return self.Alpha[:, groupNumber - 1]
                else:
                    if groupNumber == self.c:
                        return np.zeros(self.dxm + 1)
                    else:
                        return np.hstack([self.a[groupNumber - 1], self.Alpha[:, groupNumber - 1]])

            elif layer == 2:  # observed layer
                if self.b is None:
                    print "** Observed layer has no intercept. **"
                    return self.Beta[:, groupNumber - 1]
                else:
                    return np.hstack([self.b[groupNumber - 1], self.Beta[:, groupNumber - 1]])
            else:
                raise ValueError("** Layer is out of bound. **")

    # This method is not intended to be inherited
    def flattenHiddenLayerParameters(self):
        """
        Squeeze / flatten hidden layer parameters such that the intercepts (if `a` is not None)
        comes before the coefficients (`Alpha`) which are stacked by columns. This is shown below:

            a, Alpha => [a_1, a_2, ..., Alpha_{11}, Alpha_{12}, ..., Alpha_{21}, Alpha_{22}, ...]

        :return: flattened hidden layer parameters
        """
        if self.hasHiddenIntercepts:
            return np.hstack([self.a, self.Alpha.flatten(order='F')])
        else:
            return self.Alpha.flatten(order='F')

    # This method is not intended to be inherited
    def flattenObservedLayerParametersj(self, j):
        """
        Squeeze / flatten observed layer parameters for the j_th hidden group such that
        the intercept comes before the coefficients for the covariates. This shown below:

            b, Beta, j_th group => [b_j, Beta_{1j}, Beta_{2j}, ...]

        :param j:
        :return: flattened observed layer parameters for j_th hidden group
        """
        if self.hasObservedIntercepts:
            return np.hstack([self.b[j], self.Beta[:, j]])
        else:
            return self.Beta[:, j]

    # This method is not intended to be inherited
    def flatten(self):
        """
        Create a flatten list of the parameters in the model. The concatenation order is given
        as follows:

            a, Alpha, b1, Beta[:,1], b2, Beta[:, 2], ...

        :return: flattened model parameters
        """
        vHiddenLayerParams = self.flattenHiddenLayerParameters()
        vObservedLayerParams = np.array([self.flattenObservedLayerParametersj(j) for j in range(self.c)]).flatten()
        return np.hstack([vHiddenLayerParams, vObservedLayerParams])

    def normedDifference(self, that):
        """
        Calculate the L2 norm of the difference between the current parameter list and another
        given parameter list. This is mainly used to check parameter convergence
        in the EM algorithm.

        :type that: Paramters
        :param that: another set of parameters to compare to.

        :return: L2 norm of the difference between two sets of parameters.
        """
        # if not isinstance(that, Paramters)
        #     raise("The argument is not of type Parameters.")
        return np.linalg.norm(self.flatten() - that.flatten())

    # This method is not intended to be inherited
    def getOrderedParameters(self, trueParameters):
        """
        Re-order the groups of parameters to be conformable to the given true parameters. After
        re-ordering, the L2 norm between the two sets of parameters belonging to the same group
        is minimized.

        This is mainly used in simulation studies, since the parameters of a mixed logistic
        regression are identifiable up to a permutation of groups.

        By re-ordering the groups, the base group parameters (i.e. 0s) are augmented.

        :type trueParameters: Paramters
        :param trueParameters: the true parameter set whose group order the current parameter
                               set is trying to be conformable to.

        :type return: OrderedParameters
        :return: an OrderedParameters object encapsulating the re-ordered parameters.
        """
        return OrderedParameters(self.Alpha, self.Beta, self.a, self.b, trueParameters)

    def __str__(self):
        line1 = "Hidden Layer:"
        line2 = "\tGroup\tIntercept\t" + '\t'.join(['alpha'+str(i) for i in np.arange(self.dxm) + 1])
        # Construct print strings for hidden layer parameters
        # by collecting as a list of string lines where each
        # line corresponds to one hidden group
        hidden_parameter_lines = []
        for j in range(self.c - 1):
            if self.hasHiddenIntercepts:
                parameter_line = \
                    ( '\t{' + '}\t\t{'.join(str(i) for i in range(2 + self.dxm)) + '}' ).format( *((j+1, self.a[j],) + tuple(self.Alpha[:, j])) )
            else:
                parameter_line = \
                    ( '\t{' + '}\t\t{'.join(str(i) for i in range(2 + self.dxm)) + '}' ).format( *( (j+1, None,) + tuple(self.Alpha[:, j]) ) )
            hidden_parameter_lines.append(parameter_line)
        # Base group parameters are forced to 0s
        base_hidden_parameter_line = ('\t{0}' + '\t\t{1}'*(1 + self.dxm)).format(self.c, 0.)
        hidden_parameter_lines.append(base_hidden_parameter_line)
        # Insert line breaks between lines
        line3 = '\n'.join(hidden_parameter_lines)

        line4 = "Observed Layer:"
        line5 = "\tGroup\tIntercept\t" + '\t'.join(['beta'+str(i) for i in np.arange(self.dxr) + 1])
        # Construct print strings for observed layer parameters
        # by collecting as a list of string lines where each
        # line corresponds to one hidden group
        observed_parameter_lines = []
        for j in range(self.c):
            if self.hasObservedIntercepts:
                parameter_line = \
                    ( '\t{' + '}\t\t{'.join(str(i) for i in range(2 + self.dxr)) + '}' ).format( *((j+1, self.b[j],) + tuple(self.Beta[:, j])) )
            else:
                parameter_line = \
                    ( '\t{' + '}\t\t{'.join(str(i) for i in range(2 + self.dxr)) + '}' ).format( *( (j+1, None,) + tuple(self.Beta[:, j]) ) )
            observed_parameter_lines.append(parameter_line)
        # Insert line breaks between lines
        line6 = '\n'.join(observed_parameter_lines)

        return '\n'.join([line1, line2, line3, line4, line5, line6])


class OrderedParameters(Paramters):
    def __init__(self, Alpha, Beta, a, b, trueParameters):
        Paramters.__init__(self, Alpha, Beta, a, b)
        self._reorderGroups(trueParameters)

    def _reorderGroups(self, trueParameters):
        """
        Mixture model is identifiable up to the permutation of groups.
        In order to ensure unique ordering, the estimated groups are
        ordered such that the L2 distance between a give group and the
        the corresponding true group is minimized.

        :type trueParameters: Paramters
        :param trueParameters:
        """
        if self.c != trueParameters.c:
            raise ValueError("** The number of groups for the two parameters are not conformable. **")
        if self.dxm != trueParameters.dxm:
            raise ValueError("** The hidden layer dimensions for the two parameters are not conformable. **")
        if self.dxr != trueParameters.dxr:
            raise ValueError("** The observed layer dimensions for the two parameters are not conformable. **")

        # Re-organize parameters as a (dimension * number_of_groups) matrix
        thisParamsMtx = np.vstack(
                            np.hstack(
                                [self.getParameters(1, i), self.getParameters(2, i)]
                            ) for i in (np.arange(self.c) + 1)
                        )
        trueParamsMtx = np.vstack(
                            np.hstack(
                                [trueParameters.getParameters(1, i), trueParameters.getParameters(2, i)]
                            ) for i in (np.arange(trueParameters.c) + 1)
                        )

        # For each estimated group, calculate L2 norm against all true groups
        l2_dist = []
        for i in range(self.c):
            l2_dist_i = []
            for j in range(self.c):
                l2_dist_i.append(np.linalg.norm(thisParamsMtx[i, :] - trueParamsMtx[j, :]))
            l2_dist.append(l2_dist_i)

        # Find groups each estimates should belong
        l2_dist_array = np.array(l2_dist)
        groupIndices = l2_dist_array.argmin(axis=0)

        # Re-order parameters among groups
        aWithBaseGroup = np.append(self.a, 0.)
        if self.hasHiddenIntercepts:
            self.a = aWithBaseGroup[groupIndices]
        if self.hasObservedIntercepts:
            self.b = self.b[groupIndices]
        AlphaWithBaseGroup = np.append(self.Alpha, np.zeros((self.dxm, 1)), axis=1)
        self.Alpha = AlphaWithBaseGroup[:, groupIndices]
        self.Beta = self.Beta[:, groupIndices]


class Data(object):
    def __init__(self, Xm, Xr, y, m):
        self.Xm = Xm
        self.Xr = Xr
        self.y = y
        self.m = m
        self.n = len(y)


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
        if offset:
            a = vparams[:offset]
            Alpha = vparams[offset:].reshape((dxm, offset), order='F')
        else:
            a = None
            Alpha = vparams.reshape((dxm, offset), order='F')

        logMemberProbs = softmaxForData(data.Xm, Alpha, a, returnLog=True)
        return -np.multiply(weights, logMemberProbs).sum()

    def negObservedLayerQj(vparams, j):
        if hasObservedIntercepts:
            bj = vparams[0]  # first parameter is the coefficient
            betaj = vparams[1:]
        else:
            betaj = vparams

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
        if offset:
            a = vparams[:offset]
            Alpha = vparams[offset:].reshape((dxm, offset), order='F')
        else:
            a = None
            Alpha = vparams.reshape((dxm, offset), order='F')

        memberProbs = softmaxForData(data.Xm, Alpha, a)
        qMinusPi = weights[:, :-1] - memberProbs[:, :-1]
        gradAlpha = data.Xm.T.dot(qMinusPi).flatten(order='F')
        if hasHiddenIntercepts:
            gradIntercepts = qMinusPi.sum(axis=0)
            return -np.hstack([gradIntercepts, gradAlpha])
        else:
            return -gradAlpha

    def negObservedLayerQj_grad(vparams, j):
        # prepare parameters
        if hasObservedIntercepts:
            bj = vparams[0]  # first parameter is the coefficient
            betaj = vparams[1:]
        else:
            bj = None
            betaj = vparams

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
    if hasHiddenIntercepts:
        offset = params0.Alpha.shape[1]

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
        if hasHiddenIntercepts:
            a = updateNegQ1.x[:offset]
            Alpha = updateNegQ1.x[offset:].reshape((dxm, offset), order='F')
        else:
            a = None
            Alpha = updateNegQ1.x[offset:].reshape((dxm, offset), order='F')

        updateNegQ2 = []
        # print("\t Optimizing observed layer ...")
        for j in xrange(c):
            # print("\t\t component %d ..." % j)
            vObservedLayerParameterj_old = params_old.flattenObservedLayerParametersj(j)
            updateNegQ2j = optimize.minimize(fNegQ2j, vObservedLayerParameterj_old,
                                             args=(j, ), method=optMethod, jac=gradNegQ2j)
            updateNegQ2.append(updateNegQ2j)

        if hasObservedIntercepts:
            b = np.array([params.x[0] for params in updateNegQ2])
            Beta = np.vstack([params.x[1:] for params in updateNegQ2]).T
        else:
            b = None
            Beta = np.vstack([params.x for params in updateNegQ2]).T

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