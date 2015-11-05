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
trueNollks = []

seeds = np.arange(100)
for seed in seeds:
    # Simulate data
    simulator = MixedLogisticDataSimulator(5000)
    simulator.simulate(seed)
    data = simulator.data
    params = simulator.parameters  # same true parameters
    trueNollks.append(negObservedLogLikelihood(data, params))

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

# The estimated parameters are prepared by using the following commands and pickled:

# estimatedParametersList = [res['params'] for res in collectResults]
# estimatedParametersMatrix = np.vstack(params.flatten() for params in estimatedParametersList)
# pickle.dump(estimatedParametersMatrix, open('estimatedParameters.pkl', 'wb'))
# estimatedNollks = [res['nollk'] for res in collectResults]
# pickle.dump(estimatedNollks, open('estimatedNollks_10252015.pkl', 'wb'))

# Unserialize the pickled data for analysis
import pickle

pkl_file = open('mixedlogistic_separate/simulations/estimatedNollks_10252015.pkl', 'rb')
estimatedNollks = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('mixedlogistic_separate/simulations/estimatedParameters_10252015.pkl', 'rb')
estimatedParametersMatrix = pickle.load(pkl_file)
pkl_file.close()

# Compare the true and estimated likelihood
plt.boxplot(estimatedNollks)
plt.boxplot(trueNollks)

# Numerical summary of the estimated parameters
import pandas as pd

estimatedParametersDF = pd.DataFrame(estimatedParametersMatrix)
estimatedParametersDF.describe()
# estimatedParameterMeans = estimatedParametersMatrix.mean(0)

from mixedlogistic_separate.simulations.plotFunctions import *

# box plots
boxplots_c3dxm3dxr3(estimatedParametersMatrix)

# true parameters
print params

"""Simulation results
By running a systemic simulation to investigate the performance of estimation when the two layers
share the same covariates. The estimated intercepts and covariates' coefficients are plotted in
"boxplots_10252015.png". The followings are some observations:

1. The coefficients for the hidden and observed layers can be identified as the estimated parameters
   are close to the true parameters. However, the estimated intercepts are not reasonably close to
   their true counterparts.

   The followings are the five-number summary for the parameter estimates. Note that the order of the
   first and the second groups are switched.

                   a1    1-alpha1    1-alpha2    1-alpha3          a2    2-alpha1   \
    mean     0.889310   -2.729676    0.049493    4.038618   -1.267663    7.393897
    std      0.548777    5.338802    0.248463    2.681459    0.980870    5.023638
    min     -1.259284   -7.490985   -0.705265   -4.501932   -6.983170    3.074454
    25%      0.711019   -5.591319   -0.102487    4.178759   -1.501093    4.905947
    50%      0.931176   -4.821970    0.042150    4.793024   -1.075984    5.263822
    75%      1.065763   -3.935994    0.168464    5.568121   -0.783694    6.558943
    max      5.058222   10.386280    0.843112    8.103726    1.596515   27.263356

             2-alpha2    2-alpha3          b1     1-beta1     1-beta2     1-beta3  \
    mean    -0.129904   -6.494220    0.304853    8.577649    0.341513   -0.407107
    std      0.427033    2.913178    0.880005    5.158702    1.734192    1.630004
    min     -1.744827  -19.974986   -1.907984   -5.387925   -0.258633   -2.422720
    25%     -0.289002   -6.644883   -0.178162    9.216807   -0.054669   -1.194358
    50%     -0.076030   -5.395502    0.116365   10.091271    0.004710   -0.961699
    75%      0.122347   -4.931488    0.458965   10.701216    0.137651   -0.728092
    max      1.043464   -3.512034    3.064245   18.883248   16.947575    5.052961

                   b2     2-beta1     2-beta2     2-beta3          b3     3-beta1  \
    mean    -0.120597   -0.974582    0.004551   10.136448   -0.657144    1.593618
    std      0.691087    0.397296    0.184066    1.524048    1.585518    3.340542
    min     -4.928126   -1.862274   -1.145703    3.542339   -4.664093   -4.615839
    25%     -0.374882   -1.255519   -0.069115    9.505770   -1.520834   -0.925914
    50%     -0.056931   -0.970377    0.014098    9.967587   -0.422063    1.801403
    75%      0.279966   -0.749489    0.113271   10.699456    0.424808    3.447399
    max      0.905956   -0.091049    0.352210   19.683032    4.745016   10.458425

              3-beta2     3-beta3
    mean     9.125260   -2.454030
    std      5.032939    2.974939
    min     -0.064097  -10.476249
    25%      5.250994   -4.120538
    50%     10.083587   -2.906967
    75%     13.116235   -0.460170
    max     16.973811    3.577631

   comparing with the true parameters:

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

"""