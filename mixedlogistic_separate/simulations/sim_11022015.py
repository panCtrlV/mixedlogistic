from mixedlogistic_separate.fisher.fisherInformation import *


"""
Given the sample size, and true parameters, the true Fisher information
is estimated by the average of the empirical Fisher information matrices
obtained from simulated data.
"""

# dxm = 3
# dxr = 3
# c = 3
# # have intercepts
# parameterDimension = (dxm + 1) * (c-1) + (dxr + 1) * c
#
# i = 0  # initialize counter
# nSim = 1000
# seeds = np.arange(nSim)  # fix seeds to make a reproducible example
#
# fisherInfo = np.zeros((parameterDimension, parameterDimension))  # number of parameters * number of parameters
# while i < nSim:
#     i += 1
#     print i
#     fisherInfoi = empiricalFisherInformationFromSimulatedData2(5000, seeds[i-1], dxm, dxr, c)
#     fisherInfo += fisherInfoi / nSim
#
# # Heatmap of the Fisher information
# plt.imshow(fisherInfo)
# plt.colorbar()
#
# # Pretty print the whole Fisher information matrix
# for i in range(parameterDimension):
#     print '\n'
#     for j in range(parameterDimension):
#         print('%.02f\t' % fisherInfo[i, j]),
#
# # Inverse of the Fisher information is the variance of the parameter estimates
# variance = np.linalg.inv(fisherInfo)
#
# plt.imshow(variance)
# plt.colorbar()
#
# # Pretty print the variance-covariance matrix
# for i in range(parameterDimension):
#     print '\n'
#     for j in range(parameterDimension):
#         print('%.02f\t' % variance[i, j]),
# print '\n'
#
# # only print the diagonal elements
# for i in range(parameterDimension):
#     print '%.04f' % np.diag(variance)[i]

dxm = 3
dxr = 3
c = 3

pHidden = dxm + 1
pObserved = dxr + 1
nHiddenParameters = pHidden * (c-1)
nObservedParameters = pObserved * c
nParameters = nHiddenParameters + nObservedParameters

fisherInformation = estimatedFisherInformationFromSimulation(25, nSim=1000, nSample=5000, dxm=dxm, dxr=dxr, c=c)

# Heatmap of the estimated Fisher information
plt.imshow(fisherInformation)
plt.colorbar()

# Pretty print the whole Fisher information matrix
for i in range(nParameters):
    print '\n'
    for j in range(nParameters):
        print('%.02f\t' % fisherInformation[i, j]),
print '\n'

# Inversion of Fisher information is the variance-covariance of the parameter estimates
varianceCovarianceMatrix = np.linalg.inv(fisherInformation)

# Heatmap of the variance-covariance matrix
plt.imshow(varianceCovarianceMatrix)
plt.colorbar()

# Print the diagonal of the variance-covariance matrix
variance = np.diag(varianceCovarianceMatrix)
for i in range(nParameters):
    print '%.04f' % variance[i]

"""Simulation observation
With 5000 sample size, we simulate the empirical Fisher information matrix at the true parameters 1000 times.
Their average is the estimate of the true Fisher information.

The following are the diagonal elements of the inverse of the Fisher information (variance of parameter estimates):

                    group   Intercept   alpha1  alpha2  alpha3
    hidden layer    1       0.3051      0.0476  0.2358  4.8176
                    2       0.3652      0.0607  4.8022  0.1808
                    3
    observed layer  1       0.0694      0.0571  0.1276  17.1332
                    2       0.0309      0.0259  3.3232  0.0122
                    3       0.0388      1.1729  0.0091  0.0140
"""

"""
Once again, estimate a mixed logistic model using simulated data.
This is intended to see if the goodness of the estimation is conformable to
what the Fisher information expects.
"""
simulator = MixedLogisticDataSimulator(5000, dxm, dxr, c)  # try n = 500,000 or 5,000,000
simulator.simulate(25)
data = simulator.data
parameters = simulator.parameters
# print data
print parameters

res = trainEM(data, parameters, optMethod='BFGS', maxIter=10000)
print res['params']

"""Simulation observation
The parameters estimated from the simulated data is:

    Hidden Layer:
        Group	Intercept	        alpha1	            alpha2	            alpha3
        1		-0.987170318229		0.510729961671		1.75338532093		33.8284705511
        2		5.33449209244		2.45815250395		33.1065884578		-0.0197595494205
        3		0.0		            0.0		            0.0		            0.0
    Observed Layer:
        Group	Intercept	        beta1	            beta2	            beta3
        1		0.281963768676		0.302413880707		2.39258618278		35.1019126997
        2		0.513527258882		1.87019838187		28.4156780935		-0.0742261395671
        3		-0.362219713339		35.8124721837		0.00242449514338	2.35059149061

with the negative log-likelihood: 216.138213

We can compare with the following true parameters:

    Hidden Layer:
        Group	Intercept	alpha1	alpha2	alpha3
        1		0.0		    0.1		2.0		30.0
        2		5.0		    2.0		30.0	0.1
        3		0.0		    0.0		0.0		0.0
    Observed Layer:
        Group	Intercept	beta1	beta2	beta3
        1		-0.0		0.1		2.0		30.0
        2		-5.0		2.0		30.0	0.1
        3		-10.0		30.0	0.1		2.0

"""
negObservedLogLikelihood(data, parameters)
"""
with the negative log-likelihood:  832.38689477018602
"""
