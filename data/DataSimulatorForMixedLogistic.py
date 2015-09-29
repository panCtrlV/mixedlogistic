from preprocess import *
from cal_condprob import *
from helper import *
import matplotlib.pyplot as plt
import random
from collections import Counter

"""
Generating data from mixed logistic model
"""


class DataSimulatorForMixedLogistic(object):
    def __init__(self, n, dxm, dxr, c, m):
        self.n = n
        self.dxm = dxm
        self.dxr = dxr
        self.c = c
        self.m = m

    def simpleSimulator(self):
        # generate xm from independent normal
        _xm = np.random.normal(0., 1., self.n * self.dxm).reshape((self.n, self.dxm))
        xm = addIntercept(_xm)
        # coefficients for mixing probabilities
        _beta = 0.1 * (np.arange(self.dxm + 1) + 1)
        beta = []
        for i in range(self.c - 1):
            _beta = np.delete(np.append(_beta, _beta[0]), 0)
            beta.append(_beta)
        beta = addBaselineCoefficientsAsZeros(np.vstack(beta).T)
        # Calculate the membership probabilities
        pMember = cal_softmaxForData(xm, beta)
        u = np.random.uniform(size=self.n)
        _member = []
        for i in range(self.n):
            _member.append(np.sum([x < u[i] for x in pMember[i, :].cumsum()]))
        member = np.array(_member)  # 0-index

        # generate xr from independent normal
        _xr = np.random.normal(0., 1., self.n * self.dxr).reshape((self.n, self.dxr))
        xr = addIntercept(_xr)
        # coefficient for the component probabilities
        _alpha = 0.2 * (np.arange(self.dxr + 1) + 1)
        alpha = []
        for i in range(self.c):
            _alpha = np.delete(np.append(_alpha, _alpha[0]), 0)
            alpha.append(_alpha)
        alpha = np.vstack(alpha).T
        # component probabilities given membership
        pComponent = [float(cal_sigmoid(xr[i, :].dot(alpha[:, member[i]]))) for i in range(self.n)]

        # Generating response from component probabilities
        y = np.random.binomial(self.m, pComponent)

        self.xm = xm
        self.xr = xr
        self.y = y
        self.beta = beta
        self.alpha = alpha
        self.pMember = pMember
        self.member = member
        self.pComponent = pComponent

    def plotHistogramForComponentProbabilities(self):
        plt.hist(self.pComponent)

    def plotResponse(self):
        counter = Counter(self.y)
        index = counter.keys()
        values = counter.values()
        plt.bar(index, values, 0.1)

    def plotMembership(self):
        counter = Counter(self.member)
        index = counter.keys()
        values = counter.values()
        plt.bar(index, values, 0.1)


# === Use class ====
data = DataSimulatorForMixedLogistic(100, 3, 3, 3, 1)
data.simpleSimulator()
# data.plotResponse()
# data.plotHistogramForComponentProbabilities()
data.plotMembership()


# # === Constants ===
# n = 100  # sample size
# dxm = 3  # number of variables for xm
# dxr = 3  # number of variables for xr
# c = 3  # number of components
# m = 1  # Binomial number of trials (same for all observations)
#
# # === Preparing data ===
# # generate xm from independent normal
# _xm = np.random.normal(0., 1., n*dxm).reshape((n, dxm))
# xm = addIntercept(_xm)
# # coefficients for mixing probabilities
# lxm = xm.shape[1]
# _beta = 0.1 * (np.arange(lxm) + 1)
# beta = []
# for i in range(c-1):
#     _beta = np.delete(np.append(_beta, _beta[0]), 0)
#     beta.append(_beta)
# beta = addBaselineCoefficientsAsZeros(np.vstack(beta).T)
# # Calculate the membership probabilities
# pMember = cal_softmaxForData(xm, beta)
# u = np.random.uniform(size=n)
# _member = []
# for i in range(n):
#     _member.append(np.sum([x < u[i] for x in pMember[i,:].cumsum()]))
# member = np.array(_member)  # 0-index
#
# # generate xr from independent normal
# _xr = np.random.normal(0., 1., n*dxr).reshape((n, dxr))
# xr = addIntercept(_xr)
# # coefficient for the component probabilities
# lxr = xr.shape[1]
# _alpha = 0.2 * (np.arange(lxr) + 1)
# alpha = []
# for i in range(c):
#     _alpha = np.delete(np.append(_alpha, _alpha[0]), 0)
#     alpha.append(_alpha)
# alpha = np.vstack(alpha).T
# # component probabilities given membership
# pComponent = [float(cal_sigmoid(xr[i,:].dot(alpha[:,member[i]]))) for i in range(n)]
#
# # Histogram of component probabilities
# plt.hist(pComponent)
#
# # Generating response from component probabilities
# y = np.random.binomial(m, pComponent)
# # Frequency of responses
# plt.hist(y)
