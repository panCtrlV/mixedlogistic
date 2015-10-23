import random
import numpy as np

from data.DataSimulatorForMixedLogistic import *
from mixedlogistic.set_initial import *
from mixedlogistic.helper import *
from mixedlogistic.mixedLogistic import *
from mixedlogistic.predict import mixedLogistic_pred


# === Generating data ===
# By setting the same seed and data specification, the simulated data set
# is the same as that for 'examples/mlp_simulateExample'.
random.seed(25)

n = 500  # sample size
dxm = 2  # number of variables for mixing probabilities
dxr = 2  # number of variables for component probabilities
c = 2  # number of hidden groups
m = 1  # number of Binomial trials, m=1 reduces the distribution to Bernoulli

# 'data' is an object, which contains x, y, and
# the initial values for alpha and beta.
data = DataSimulatorForMixedLogistic(n, dxm, dxr, c, m)
data.simpleSimulator()

# === Train model ===
# Assume c is known
c = 2

# Initialize parameters
b0, a0 = set_initial(c, data.dxm + 1, data.dxr + 1)
param0 = mapOptimizationMatrices2Vector(b0, a0)

# nParams = (data.dxm + 1) * (c - 1) + (data.dxr + 1) * c
# param0 = np.zeros(nParams)
# param0 = np.ones(nParams)

# res = mixedLogistic_EMtrain(param0=param0, c=c,
#                             y=np.matrix(data.y).T,
#                             xm=data.xm,
#                             xr=data.xr,
#                             m=1)

res = mixedLogistic_EMtrain_separate(param0=param0, c=c,
                                     y=np.matrix(data.y).T,
                                     xm=data.xm,
                                     xr=data.xr,
                                     m=1)

# === Check parameters ===
beta_est, alpha_est = mapOptimizationVector2Matrices(res['param'], 2, 3)
beta_est
alpha_est

data.beta
data.alpha

# === Predict ===
pred = mixedLogistic_pred(res, data.xm, data.xr, c, data.y)
