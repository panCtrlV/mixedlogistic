from mixedlogistic_separate.fisher.fisherInformation import *


"""
This simulation is used to check if the estimated membership is consistent to the true membership.

In order to do so, we calculate the conditional membership probabilities based on the estimated
parameters and the memberships are determined by the one with the highest posterior probability.
Then the estimated memberships are compared with the true memberships to determine the number /
proportion of misclassification.
"""


nSample = 5000

simulator = MixedLogisticDataSimulator(nSample, 3, 3, 3)
simulator.simulate(25)
data = simulator.data
parameters = simulator.parameters
trueMembership = simulator.hiddenMembers

# Randomly initialize parameters
np.random.seed(25)

# Alpha0 = np.random.normal(2, 1, size=parameters.Alpha.shape)
# Beta0 = np.random.normal(2, 1, size=parameters.Beta.shape)
# a0 = np.random.normal(2, 1, size=parameters.a.shape)
# b0 = np.random.normal(2, 1, size=parameters.b.shape)

Alpha0 = np.random.uniform(2, 5, size=parameters.Alpha.shape)
Beta0 = np.random.uniform(2, 5, size=parameters.Beta.shape)
a0 = np.random.uniform(2, 5, size=parameters.a.shape)
b0 = np.random.uniform(2, 5, size=parameters.b.shape)

parameters0 = Paramters(Alpha0, Beta0, a0, b0)

# Estimation
res = trainEM(data, parameters0, maxIter=10000)  # user random initial
# res = trainEM(data, parameters, maxIter=10000)  # use true parameters as initial

## compare true and estimated parameters
estimatedParameters = res['params']
print parameters
print estimatedParameters

## compare hidden memeberships
estimatedMembershipProbs = posteriorHiddenMemberProbs(data, estimatedParameters)
estimatedMembership = np.argmax(estimatedMembershipProbs, 1)

print estimatedMembership[:10]
print trueMembership[:10]

nMisclassified = (estimatedMembership  != trueMembership).sum()
print("%d out of %d samples (%.04f %%) are misclassified." % (nMisclassified, nSample, float(nMisclassified)/nSample*100))

"""Observational results
1. It seems that the classification accuracy of the estimated membership is good.
   We initialize the parameters by Normal(2, 1), and estimate the parameters with different sample size.
   After align the estimated parameters with the true parameters, the following reports the classification
   performance:
    - When sample size is 5000, the misclassification rate is 1.2%.
    - When sample size is 50000, the misclassification rate is  0.95%.

Similar results are obtained if the initial parameters are set by Uniform(2, 5).

2. The parameter estimates are sensitive to the initial values.

   When the initial values are randomly assigned, we often saw the estimate for the largest parameter
   in the observed layer (true value is 30) is around 17 (almost half of the true value).

   When we use the true parameters as the initial, the estimate is much closer to the true value 30.

   The following reports the simulation results, where the random initials are set by Unif(2, 5). The
   sample size is fixed at 5000.

                                    group1:beta3    group2:beta2    group3:beta1
   true                             30              30              30
   estimate (random initial)        17.9277         17.3042         17.4250
   estimate (true as initial)       35.1024         28.4155         35.8129

3. It is observed that the intercepts of the observed layer are not very well estimated.
"""