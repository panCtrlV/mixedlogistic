"""
Application of Mixed Logistic on the Simulated data.

The data in this example is the same as that in 'examples/mlp_simulateExample'.

In this example, the data is simulated from a **hierarchical** mixed logistic model.
This means that the mixing probabilities are explicitly modeled by a logistic regression,
instead of being constants. This requires the mixing probabilities depend on some
covariates 'xm'.
"""

import random
import numpy as np

from data.DataSimulatorForMixedLogistic import *
from mixedlogistic.set_initial import *
from mixedlogistic.helper import *
from mixedlogistic.mixedLogistic import mixedLogistic_EMtrain
from mixedlogistic.predict import mixedLogistic_pred

# === Generating data ===
# By setting the same seed and data specification, the simulated data set
# is the same as that for 'examples/mlp_simulateExample'.
random.seed(25)

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
# In 'examples/mlp_simulateExample', the data set
# is partitioned into train, validation, and test subsets.
# This is suitable for training a MLP.
# In the current application, we use the data to train a
# mixed logistic model. Validation is not necessary.

random.seed(25)
trainIds = random.sample(range(n), 4000)
remainingIds = list(set(range(n)) - set(trainIds))
validateIds = random.sample(remainingIds, 1000)
testIds = list(set(remainingIds) - set(validateIds))

trainAndValidateIds = trainIds + validateIds

# subset for training data with cross-validation
trainAndValidateXm = data.xm[trainAndValidateIds, :]
trainAndValidateXr = data.xr[trainAndValidateIds, :]
trainAndValidateY = data.y[trainAndValidateIds]
# subset for testing
testXm = data.xm[trainAndValidateIds, :]
testXr = data.xr[trainAndValidateIds, :]
testY = data.y[trainAndValidateIds]

# === Training the model ===
# By using `nfolds` corss-validation, each value for 'c' is used
# to train the model on (nfolds - 1)/nfolds of the data subset
# for train and validate and then a classification error will be
# calculated for the remaining 1/nfolds of the data subset.
# By repeating it for `nfolds` times, the average validation error
# will be calculated.
# The average validation errors will be compared across all considered
# values of `c`. Then the `c` with the smallest average validation error
# will be chosen as the tuned value.
c_candidates = np.arange(10) + 1  # candidates for number of hidden groups
                                  # It is tuned by cross-validation
nfolds = 10  # 'nfolds' cross-validation
batchSize = trainAndValidateXm.shape[0] / nfolds

# params = []
print('--- Start %d-fold cross-validation ---' % (nfolds))
avgValidateScores = []
for c in c_candidates:
    print(" \tCurrent number of hidden groups 'c' = %d " % (c))
    validateScores = []
    for batchIndex in xrange(nfolds):
        print('\t\t Batch %d ...' % (batchIndex))
        # Initial parameters
        random.seed(25)
        b0, a0 = set_initial(c, data.dxm + 1, data.dxr + 1)
        param0 = mapOptimizationMatrices2Vector(b0, a0)

        validateIds = np.arange(batchIndex * batchSize, (batchIndex + 1) * batchSize)
        trainIds = list(set(np.arange(trainAndValidateXm.shape[0])) - set(validateIds))

        res = mixedLogistic_EMtrain(param0=param0, c=c,
                                    y=np.matrix(trainAndValidateY[trainIds]).T,
                                    xm=trainAndValidateXm[trainIds, :],
                                    xr=trainAndValidateXr[trainIds, :],
                                    m=1)

        pred = mixedLogistic_pred(res=res,
                                  xmt=trainAndValidateXm[validateIds, :],
                                  xrt=trainAndValidateXr[validateIds, :],
                                  c=c,
                                  yt=np.matrix(trainAndValidateY[validateIds]).T)

        validateScores.append(pred['f1'])
        print('\t\twith validation score (F1) = %f' % (pred['f1']))

    avgValidateScores.append(np.mean(validateScores))
    print('\tAverage validation score = %f for c = %d' % (np.mean(validateScores), c))

c_optim = c_candidates[np.array(avgValidateScores).argmax()]
print('Optimal c = %d' % (c_optim))
