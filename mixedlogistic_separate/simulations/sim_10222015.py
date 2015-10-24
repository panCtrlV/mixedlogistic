from mixedlogistic_separate.mixedLogistic_rawScript import *

"""
We saw the problem of inaccurate estimation in the previous simulations.
In particular, when we ran the simulation with the following setting:

    - c = 3
    - dxm = 3
    - dxr = 3
    - hasHiddenIntercepts = True
    - hadObservedIntercepts = True

The estimates for the observed layer are not as good as those of the
hidden layer.

In this script, we try to reduce the parameter dimension to see if the
problem still persists. The following setting is used:

    - c = 3
    - dxm = 1
    - dxr = 1
    - hasHiddenIntercepts = True
    - hadObservedIntercepts = True

In the current setting, the parameter dimension is 10 instead of 20.
"""

dxr = 1
dxm = 1

class MixedLogisticDataSimulator(object):
    def __init__(self, n, dxm, dxr):
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
        # return np.array([[0., 0.]])  # hidden membership probabilities = 1/3
        return np.array([[5., 0.]])

    # set observed layer coefficients
    def _setBeta(self):
        return np.array([[-5., 0., 5.]])


    # set hidden layer intercepts
    def _set_a(self):
        return np.array([0., 0.])

    # set observed layer intercepts
    def _set_b(self):
        # return np.array([0., 0., 0.])
        return np.array([-1., 0., 1.])

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


# Simulate data
simulator = MixedLogisticDataSimulator(5000, dxr, dxm)
simulator.simulate(125)

data = simulator.data
params = simulator.parameters

# print params
# simulator.hiddenMemberProbs.mean(0)
# np.unique(simulator.hiddenMembers, return_counts=True)


# Set initial values
np.random.seed(25)
Alpha0 = np.random.uniform(2, 5, size=params.Alpha.shape)
Beta0 = np.random.uniform(2, 5, size=params.Beta.shape)
a0 = np.random.uniform(2, 5, size=params.a.shape)
b0 = np.random.uniform(2, 5, size=params.b.shape)
params0 = Paramters(Alpha0, Beta0, a0, b0)

# Train
res = trainEM(data, params0, maxIter=10000)

# Compare true and estimated parameters
est_parameters = res['params']
print params, '\n'
print est_parameters


# Compare negative log-likelihood for true and estimated parameters
print 'nollk (true parameters):\t\t', negObservedLogLikelihood(data, params)
print 'nollk (estimated parameters):\t', res['nollk']


"""=== Observational Conclusion ====

1. Wither fewer parameters to estimated, the negative log-likelihood calculated from the
   estimated parameters is close to that calculated from the true parameters. The nollk's
   are shown below:

    nollk (true parameters):		3465.7359028
    nollk (estimated parameters):	3463.53675009

2. The training process is very difficult. It is very hard for the parameters to converge.

3. Up on terminating the training by reaching the maximum number of iterations, the estimated
   parameters are not close to the true parameters. The followings are the true and estimated
   parameters:

    > True Parameters:
    Hidden Layer:
        Group	Intercept	alpha1
        1		0.0		    0.0
        2		0.0		    0.0
        3		0.0		    0.0
    Observed Layer:
        Group	Intercept	beta1
        1		-1.0		-5.0
        2		0.0		    0.0
        3		1.0		    5.0

    > Estimated Parameters:
    Hidden Layer:
        Group	Intercept	        alpha1
        1		1.77313016175		5.70128319044
        2		3.36328643388		2.78588206223
        3		0.0		            0.0
    Observed Layer:
        Group	Intercept	            beta1
        1		-0.0226726265897		0.0775792126973
        2		0.00242433809325		-0.0763059734443
        3		0.0713911584531		    0.0185411179339

4. The observations 1 - 3 are based on the balanced hidden memberships, i.e. \pi_1 = \pi_2 = \pi_3 = 1/3.
   **Balanced hidden memberships may cause the training process to be confused.** Then, we modified the
   simulator by force the hidden memberships to be unbalanced. The true parameters become:

    Hidden Layer:
        Group	Intercept	alpha1
        1		0.0		    5.0
        2		0.0		    0.0
        3		0.0		    0.0
    Observed Layer:
        Group	Intercept	beta1
        1		-1.0		-5.0
        2		0.0		    0.0
        3		1.0		    5.0

   where everything is the same except \alpha_{11} = 5 now. This modification causes more data to belong
   to the first hidden group. The training took less effort this time, which converged in 2950 iterations
   instead of being forcefully terminated at 10000. Compared to the true parameters, the estimated parameters
   are more reasonable:

    Hidden Layer:
        Group	Intercept	        alpha1
        1		0.488065473794		4.78604539551
        2		0.619662518278		0.179725687892
        3		0.0		            0.0
    Observed Layer:
        Group	Intercept	        beta1
        1		0.153804379195		-4.61571684143
        2		0.184863194958		0.221584461991
        3		-1.07362214031		8.380826987

   However, the estimates of the observed layer intercepts do not seem to be good, especially because
   b_1 and b_3 changed the signs.

   The negative log-likelihoods calculated from the true and estimated parameters are also very close:

    nollk (true parameters):		2406.27926243
    nollk (estimated parameters):	2320.42283123
"""