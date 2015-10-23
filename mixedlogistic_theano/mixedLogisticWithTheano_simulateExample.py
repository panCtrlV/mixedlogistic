"""
Application of mixed logistic regression on simulated data.
In this example, a non-hierarchical hidden layer is used. In
other words, the membership probabilities are assumed to be
fixed constants. They are the parameters to be estimated.
"""


import theano, random
import theano.tensor as T
import numpy as np

from data.DataSimulatorForMixedLogistic import DataSimulatorForMixedLogistic
from MixedLogisticRegression import MixedLogisticRegression


# === Generating data ===
n = 6000  # sample size
dxm = 3  # number of variables for mixing probabilities
dxr = 5  # number of variables for component probabilities
c = 3  # number of hidden groups
m = 1  # number of Binomial trials, m=1 reduces the distribution to Bernoulli

# 'data' is an object, which contains x, y, and
# the initial values for alpha and beta.
data = DataSimulatorForMixedLogistic(n, dxm, dxr, c, m)
data.simpleSimulator()

# === Processing data ===
# Combine data.xr and data.xm into a single covariate matrix
# which is converted to a Theano shared variable.
#
# The first column in data.xm and data.xr are ones vectors for
# the intercept. Before combining the two data sets, those ones
# columns are removed.
#
# Besides, data.y is also converted to a Theano shared variable.
# Since y contains 0-1 labels, so they are stored as integer type.
dataX = np.hstack((data.xm[:, 1:], data.xr[:, 1:]))
dataY = data.y
# dataX = theano.shared(_x, borrow=True)
# dataY = theano.shared(data.y, borrow=True)

# Split the data into training, validation, and test subsets.
# In the 6000 samples, 1000 are used for test, 1000 are used for
# validation, and the remaining 4000 samples are used for training.
random.seed(25)
trainIds = random.sample(range(n), 4000)
remainingIds = list(set(range(n)) - set(trainIds))
validateIds = random.sample(remainingIds, 1000)
testIds = list(set(remainingIds) - set(validateIds))

# Convert data to Theano shared variables
trainX = theano.shared(dataX[trainIds, :])
trainY = T.cast(theano.shared(dataY[trainIds]), 'int32')
validateX = theano.shared(dataX[validateIds, :])
validateY = T.cast(theano.shared(dataY[validateIds]), 'int32')
testX = theano.shared(dataX[testIds, :])
testY = T.cast(theano.shared(dataY[testIds]), 'int32')

# === Build the model ===
print '--- Building the model ---'

xm = T.dmatrix('xm')
xr = T.dmatrix('xr')
y = T.ivector('y')

c = 3

classifier = MixedLogisticRegression(input_for_decision_layer=xr, c=c, n_in_for_decision_layer=dxr,
                                     n_out=2, input_for_hidden_layer=xm, n_in_for_hidden_layer=dxm)

# Cost function is used to check if EM converges.
# When it becomes stable, it's time to terminate EM.
cost = classifier.negative_log_likelihood(y)

# Training a mixed logistic regression model uses EM.
pass
