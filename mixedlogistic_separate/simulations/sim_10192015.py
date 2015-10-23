from mixedlogistic_separate.mixedLogistic_rawScript import *


"""
When the largest parameter of Beta is 30, we saw the corresponding estimate
is around 17, which is almost the half of the true parameter.

We gradually reduce the value of the largest parameter as follows:

    30, 25, 20, 15, 10, 5

to investigate if the reduce-by-half effect persists.

The following settings are fixed:
   - c = 3
   - dxm = 3
   - dxr = 3
   - hasHiddenIntercepts = True
   - hadObservedIntercepts = True
"""


class MixedLogisticDataSimulator(object):
    def __init__(self, n, largestBeta):
        self.n = n  # sample size
        self.dxm = 3  # hidden layer covariates dimension
        self.dxr = 3  # observed layer covariates dimension
        self.c = 3
        self.m = 1  # number of trials
        self.k = 2  # number of response categories
        self.parameters = None
        self.data = None
        self.hiddenMemberProbs = None
        self.hiddenMembers = None
        self.choosingProbs = None
        self.hasHiddenIntercepts = True
        self.hasObservedIntercepts = True

        self.largestBeta = largestBeta

    # set hidden layer coefficients
    def _setAlpha(self):
        return np.array([[0.1, 2., 30], [30, 2., 0.1]]).T

    # set observed layer coefficients
    def _setBeta(self):
        return np.array([[0.1, 5, self.largestBeta],
                         [self.largestBeta, 0.1, 5],
                         [5, self.largestBeta, 0.1]]).T

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
            raise("Not implemented yet!")

        self.parameters = Paramters(Alpha, Beta, a, b)
        self.hiddenMemberProbs = hiddenMemberProbs
        self.hiddenMembers = hiddenMembers
        self.choosingProbs = choosingProbs
        self.data = Data(Xm, Xr, y, self.m)


# The simulation study use the same initial values
np.random.seed(25)
Alpha0 = np.random.uniform(2, 5, size=(3, 2))  # Unif(2, 5)
Beta0 = np.random.uniform(2, 5, size=(3, 3))
a0 = np.random.uniform(2, 5, size=(2,))
b0 = np.random.uniform(2, 5, size=(3,))
params0 = Paramters(Alpha0, Beta0, a0, b0)

# The seed for generating data is also fixed
collectResults = []
collectData = []
collectTrueParameters = []
# largestBetaValues = [30, 25, 20, 15, 10, 5]
largestBetaValues = [30]
for val in largestBetaValues:
    simulator = MixedLogisticDataSimulator(n=5000, largestBeta=val)
    simulator.simulate(125)
    data = simulator.data
    collectData.append(data)
    trueParameters = simulator.parameters
    collectTrueParameters.append(trueParameters)
    res = trainEM(data, params0, maxIter=10000)
    # res = trainEM(data, trueParameters, maxIter=10000)  # use true parameters as initials
    collectResults.append(res)


# estParameters = collectResults[0]['params']
# trueParameters = collectTrueParameters[0]
# orderedEstParameters = estParameters.getOrderedParameters(trueParameters)
# orderedEstParameters.getParameters(2,3)

# Compare true Beta and estimated Beta
for i in range(6):
    print '---', largestBetaValues[i], '---'
    print collectResults[i]['params'].a
    print collectTrueParameters[i].a

for i in range(6):
    print '---', largestBetaValues[i], '---'
    print collectResults[i]['params'].Alpha, '\n'
    print collectTrueParameters[i].Alpha

for i in range(6):
    print '---', largestBetaValues[i], '---'
    print collectResults[i]['params'].b
    print collectTrueParameters[i].b

for i in range(6):
    print '---', largestBetaValues[i], '---'
    print collectResults[i]['params'].Beta, '\n'
    print collectTrueParameters[i].Beta



# minimum of nollk
[res['nollk'] for res in collectResults]
# nollk at true parameters
[negObservedLogLikelihood(data, params) for (data, params) in zip(collectData, collectTrueParameters)]
