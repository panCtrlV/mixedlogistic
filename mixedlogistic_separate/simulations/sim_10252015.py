from mixedlogistic_separate.mixedLogistic_rawScript import *

"""
Investigate if the model is estimable when hidden and observed layers
share the same covariates.

Simulation setting:
    - c: =3  number of hidden groups
    - dxm: =3  number of hidden layer covariates
    - dxr: =3  number of observed layer covariates

hidden and observed layer covariates are the same.
"""

class MixedLogisticDataSimulator(object):
    def __init__(self, n, dxm=3, dxr=3):
        self.n = n  # sample size
        self.dxm = dxm  # hidden layer covariates dimension
        self.dxr = dxr  # observed layer covariates dimension
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

    # set hidden layer coefficients
    def _setAlpha(self):
        return np.array([[5., 0., -5.], [-5., 0., 5.]]).T

    # set observed layer coefficients
    def _setBeta(self):
        return np.array([[-1., 0., 10.], [10., 0., -1.], [0., 10., -1.]]).T

    # set hidden layer intercepts
    def _set_a(self):
        return np.array([-1., 1.])

    # set observed layer intercepts
    def _set_b(self):
        return np.array([-5., 0., 5.])

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
        # Xr = self._setXr()
        Xr = Xm  # share the same covariates

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

# Simulate data
simulator = MixedLogisticDataSimulator(5000)
simulator.simulate(25)
data = simulator.data
params = simulator.parameters

# Set initials
np.random.seed(25)
Alpha0 = np.random.uniform(0, 1, size=params.Alpha.shape)
Beta0 = np.random.uniform(0, 1, size=params.Beta.shape)
a0 = np.random.uniform(0, 1, size=params.a.shape)
b0 = np.random.uniform(0, 1, size=params.b.shape)
params0 = Paramters(Alpha0, Beta0, a0, b0)

# Train
res = trainEM(data, params0, maxIter=10000)

# Compare estimated and true parameters
params_est = res['params']
orderedParams_est = params_est.getOrderedParameters(params)
print orderedParams_est, '\n'
print params

# Compare nollk at estimated and true parameters
print "Nollk (estimated parameters) = ", negObservedLogLikelihood(data, params_est)
print "Nollk (true parameters) = ", negObservedLogLikelihood(data, params)


"""Observational Conclustion

Based on the limited number of simulation, there are a few observations:

1. It seems that by sharing all covariates between hidden and observed layers, the model can
   still be estimated (with 934 iterations). By running the above simulation, we compare the
   estimated and true parameters as follows:

    > Estimated Parameters:
    Hidden Layer:
        Group	Intercept	        alpha1	            alpha2	            alpha3
        1		-0.767171877393		5.05899284762		-0.0100760115059	-5.23449823337
        2		1.02743683528		-4.19840507748		0.0589443892047		4.29436128566
        3		0.0		            0.0		            0.0		            0.0
    Observed Layer:
        Group	Intercept	        beta1	            beta2	            beta3
        1		0.418155275061		-1.26993461266		0.0623359856431		10.354859806
        2		-0.453235582059		8.55699306989		-0.0111727117338	-0.223227978107
        3		-0.118321337146		0.612961553519		8.33264457874		-1.66535216434

    > True Parameters:
    Hidden Layer:
        Group	Intercept	alpha1	alpha2	alpha3
        1		-1.0		5.0		0.0		-5.0
        2		1.0		    -5.0	0.0		5.0
        3		0.0		    0.0		0.0		0.0
    Observed Layer:
        Group	Intercept	beta1	beta2	beta3
        1		-5.0		-1.0	0.0		10.0
        2		0.0		    10.0	0.0		-1.0
        3		5.0		    0.0		10.0	-1.0

    It can be seen that except the estimation for the observed layer intercepts, all other estimates
    are reasonably close to the true parameters.

    The sign of the estimates of the observed layer intercepts are not consistent with the true parameters.
"""

# In order to systemically study the estimation accuracy of the model where the two layers
# share the same covariates, we run 100 simulations.

collectResults = []

seeds = np.arange(100)
for seed in seeds:
    # Simulate data
    simulator = MixedLogisticDataSimulator(5000)
    simulator.simulate(seed)
    data = simulator.data
    params = simulator.parameters

    # Set initials (same for all simulations)
    np.random.seed(25)
    Alpha0 = np.random.uniform(0, 1, size=params.Alpha.shape)
    Beta0 = np.random.uniform(0, 1, size=params.Beta.shape)
    a0 = np.random.uniform(0, 1, size=params.a.shape)
    b0 = np.random.uniform(0, 1, size=params.b.shape)
    params0 = Paramters(Alpha0, Beta0, a0, b0)

    # Train
    res = trainEM(data, params0, maxIter=10000)

    collectResults.append(res)

# The simulation is running on mean ...