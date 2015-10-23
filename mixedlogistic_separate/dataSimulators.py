from mixedLogistic_rawScript import *


class MixedLogisticDataSimulator(object):
    def __init__(self, n, dxm, dxr, c, m=1, k=2,
                 hasHiddenIntercepts=True, hasObservedIntercepts=True):
        self.n = n  # sample size
        self.dxm = dxm  # hidden layer covariates dimension
        self.dxr = dxr  # observed layer covariates dimension
        self.c = c
        self.m = m  # number of trials
        self.k = k  # number of response categories
        self.parameters = None
        self.data = None
        self.hiddenMemberProbs = None
        self.hiddenMembers = None
        self.choosingProbs = None
        self.hasHiddenIntercepts = hasHiddenIntercepts
        self.hasObservedIntercepts = hasObservedIntercepts

    # set hidden layer coefficients
    def _setAlpha(self):
        _alpha = (np.arange(self.dxm) + 1) * np.power(10., np.arange(self.dxm) - self.dxm / 2)

        # _alpha = np.ones(self.dxm) * np.power(10., np.arange(self.dxm) - self.dxm / 2)
        alpha = []
        for i in range(self.c - 1):
            alpha.append(_alpha)
            _alpha = np.delete(np.append(_alpha, _alpha[0]), 0)
        return np.vstack(alpha).T

        # return np.array([[2., 2., 2.], [3., 3., 3.]]).T  # c == 3  10-19-2015

    # set observed layer coefficients
    def _setBeta(self):
        # _beta = (np.arange(self.dxr) + 1) * np.power(10., np.arange(self.dxr) - self.dxr / 2)

        # _beta = np.ones(self.dxr) * np.power(10., np.arange(self.dxr) - self.dxr / 2)
        # beta = []
        # for i in range(self.c):
        #     beta.append(_beta)
        #     _beta = np.delete(np.append(_beta, _beta[0]), 0)
        # return np.vstack(beta).T

        return np.array([[0.1, 5, 14], [14, 0.1, 5], [5, 14, 0.1]]).T  # 10-19-2015

    # set hidden layer intercepts
    def _set_a(self):
        # return (np.arange(self.c - 1) + 1) * 0.5
        return np.arange(self.c - 1) * 5.

    # set observed layer intercepts
    def _set_b(self):
        # return (np.arange(self.c) + 1) * -0.5
        return np.arange(self.c) * -5.

    def _setXm(self):
        return np.random.uniform(-2, 2, self.n * self.dxm).reshape((self.n, self.dxm), order='F')

    def _setXr(self):
        return np.random.uniform(-2, 2, self.n * self.dxr).reshape((self.n, self.dxr), order='F')

    def simulate(self, seed):
        np.random.seed(seed)

        Alpha = self._setAlpha()
        Beta = self._setBeta()
        if self.hasHiddenIntercepts: a = self._set_a()
        if self.hasObservedIntercepts: b = self._set_b()
        Xm = self._setXm()
        Xr = self._setXr()

        # Calculate membership probabilities
        hiddenMemberProbs = softmaxForData(Xm, Alpha, a)
        # Simulate hidden membership
        u = np.random.uniform(size=self.n)
        hiddenMembers = (hiddenMemberProbs.cumsum(axis=1) - u.reshape(len(u), 1) < 0).sum(axis=1)  # start from 0
        # Knowing hidden membership, calculate choosing probabilities (i.e. Binomial probabilities)
        choosingProbs = sigmoid(Xr.dot(Beta))[np.arange(self.n), hiddenMembers]
        # Simulate response
        if isinstance(self.m, int):
            if self.m == 1:  # binary
                u = np.random.uniform(size=self.n)
                # print np.vstack([choosingProbs, u]).T
                y = (choosingProbs > u)
            else:  # binomial
                y = np.random.binomial(self.m, choosingProbs)
        else:
            raise NotImplementedError("Not implemented yet!")

        self.parameters = Paramters(Alpha, Beta, a, b)
        self.hiddenMemberProbs = hiddenMemberProbs
        self.hiddenMembers = hiddenMembers
        self.choosingProbs = choosingProbs
        self.data = Data(Xm, Xr, y, self.m)
