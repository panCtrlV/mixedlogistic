import numpy as np
from scipy.stats import binom
from scipy import optimize

from mixedlogistic_separate.Data import *
from mixedlogistic_separate.Parameters import *
from mixedlogistic_separate.Likelihoods import *

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

# Moved to Math.py


####################################################
# Posterior (Conditional) Membership Probabilities #
####################################################

# Moved to Likelihoods.py

########################
# Likelihood Functions #
########################

# Moved to Likelihoods.py


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
    # dxm = params0.dxm
    # hasHiddenIntercepts = params0.hasHiddenIntercepts
    # hasObservedIntercepts = params0.hasObservedIntercepts

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
        # format hidden layer parameters
        a, Alpha = params0.getHiddenParametersFromFlatInput(updateNegQ1.x)

        bjs = []
        betajs = []
        # print("\t Optimizing observed layer ...")
        for j in xrange(c):
            # print("\t\t component %d ..." % j)
            vObservedLayerParameterj_old = params_old.flattenObservedLayerParametersj(j)
            updateNegQ2j = optimize.minimize(fNegQ2j, vObservedLayerParameterj_old,
                                             args=(j, ), method=optMethod, jac=gradNegQ2j)
            bjs.append(updateNegQ2j.x[0])
            betajs.append(updateNegQ2j.x[1:])
        # format observed layer parameters
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

        # print params

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
