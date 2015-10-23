import random

from data.heartscale import *
import helper
import mixedLogistic
import preprocess
from mixedlogistic import set_initial
from mixedlogistic.predict import *

"""
Expeiment mixed logistic model on heart scale data from Cj Lin's distributed liblinear
"""

# === Preparing data ===
# read data
data = readData_heartScale(location)

# prepare input
yc = np.matrix(data['labels']).T
xc = data['covariates'].toarray()  # the original data['covariates'] is sparse

# Partition data into training and test data
trainIndex = random.sample(range(xc.shape[0]), 200)
testIndex = [x for x in range(270) if x not in trainIndex]
# training data
x = xc[trainIndex, :]
y = yc[trainIndex, :]
# test data
xt = xc[testIndex, :]
yt = yc[testIndex, :]

nSample, nCovariates = x.shape

# Partition covariates for mixing and conditional binomial parts
d = 5
lxm = d+1
lxr = 13 - d + 1

_xm = x[:,:d]
xm = preprocess.addIntercept(_xm)

_xr = x[:, d:]
xr = preprocess.addIntercept(_xr)

m = np.matrix(np.repeat(1, nSample)).T

# === Hyper-parameters ===
# Number of hidden groups
c = 2

# === Training ===
# Set initial values for parameters
b0, a0 = set_initial.set_initial(c, lxm, lxr)
param0 = helper.mapOptimizationMatrices2Vector(b0, a0)
# Train
res = mixedLogistic.mixedLogistic_EMtrain(param0, c, y, xm, xr, m=1)

# === Testing ===
# b, a = helper.mapOptimizationVector2Matrices(res['param'], c, lxm)
# Prepare test data
_xmt = xt[:,:d]
xmt = preprocess.addIntercept(_xmt)

_xrt = xt[:, d:]
xrt = preprocess.addIntercept(_xrt)

performance = mixedLogistic_test(res, xmt, xrt, c, yt)
performance['precision']
performance['recall']
performance['f1']

# TODO: Use AIC, BIC or Cross-validation to tune the number of components.
