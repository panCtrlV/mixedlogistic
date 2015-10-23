"""
This example illustrates how to apply MLP on the simulated data set.
The data set is simulated from a mixed logistic model.

A tutorial on MLP can be found at:
http://deeplearning.net/tutorial/mlp.html
"""
__docformat__ = 'restructedtext en'

import theano, random
import theano.tensor as T

from data.DataSimulatorForMixedLogistic import *
from mlp.mlp import *
from mlp.logistic_sgd import *


# === Generating data ===
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
# Combine data.xm and data.xm into a single covariate matrix
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

trainX = dataX[trainIds, :]
trainY = dataY[trainIds]
validateX = dataX[validateIds, :]
validateY = dataY[validateIds]
testX = dataX[testIds, :]
testY = dataY[testIds]

# Convert data to Theano shared variables
trainX = theano.shared(trainX)
trainY = T.cast(theano.shared(trainY), 'int32')
validateX = theano.shared(validateX)
validateY = T.cast(theano.shared(validateY), 'int32')
testX = theano.shared(testX)
testY = T.cast(theano.shared(testY), 'int32')

# === Setting training parameters ===
batchSize = 200
nTrainBatches = trainX.eval().shape[0] / batchSize
nValidateBatches = validateX.eval().shape[0] / batchSize
nTestBatches = testX.eval().shape[0] / batchSize
nEpochs = 1000

# === Building the model ===
# The construction of the model is symbolic by using Theano.
print '--- building the model ---'

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')  # [a minibatch of] covariate matrix
y = T.ivector('y')  # [a minibatch of] labels as a 1D vector

# Random number generator for initializing W and b for MLP
rng = np.random.RandomState(1234)

# Construct the MLP object
nHidden = 5
l1Reg = 0.0
l2Reg = 0.0

classifier = MLP(rng=rng, input=x, n_in=dxm + dxr, n_hidden=nHidden, n_out=2)
cost = classifier.negative_log_likelihood(y) + l1Reg * classifier.L1 + l2Reg * classifier.L2_sqr

testModel = theano.function(inputs=[index],
                             outputs=classifier.errors(y),
                             givens={
                                 x : testX[index * batchSize : (index + 1) * batchSize],
                                 y : testY[index * batchSize : (index + 1) * batchSize]
                             })

validateModel = theano.function(inputs=[index],
                                outputs=classifier.errors(y),
                                givens={
                                    x : validateX[index * batchSize : (index + 1) * batchSize],
                                    y : validateY[index * batchSize : (index + 1) * batchSize]
                                })

# Gradient of the cost wrt the parameters
# which is used in gradient decent optimization
gradParams = [T.grad(cost, param) for param in classifier.params]
# Specify parameter update rules for optimization
updates = [(param, param - gradParam) for param, gradParam in zip(classifier.params, gradParams)]

trainModel = theano.function(inputs=[index],
                             outputs=cost,
                             updates=updates,
                             givens={
                                 x : trainX[index * batchSize : (index + 1) * batchSize],
                                 y : trainY[index * batchSize : (index + 1) * batchSize]
                             })

# === Train the model ===
print '--- Training ---'

patience = 10000  # at least this many samples will be used
patienceIncrease = 2
improvementThreshold = 0.995
validationFreq = min(nTrainBatches, patience / 2)

bestValidationLoss = np.inf
bestIter = 0
testScore = 0.

startTime = timeit.default_timer()

epoch = 0
doneLooping = False

while (epoch < nEpochs) and (not doneLooping):
    epoch = epoch + 1
    for minibatchIndex in xrange(nTrainBatches):
        minibatchAvgCost = trainModel(minibatchIndex)
        iter = (epoch - 1) * batchSize + minibatchIndex  # 0-index, this is the total number of iterations throughout training

        if (iter + 1) % validationFreq == 0:  # it's time to validate
            validationLosses = [validateModel(validateBatchIndex) for validateBatchIndex in xrange(nValidateBatches)]
            currentValidationLoss = np.mean(validationLosses)

            print(
                'epoch %i, minibatch %i/%i, validation error %f %%' %
                (
                    epoch,
                    minibatchIndex + 1,
                    nTrainBatches,
                    currentValidationLoss * 100.
                )
            )

            if currentValidationLoss < bestValidationLoss:
                if currentValidationLoss < bestValidationLoss * improvementThreshold:
                    patience = max(patience, iter * patienceIncrease)

                bestValidationLoss = currentValidationLoss
                bestIter = iter

                testLosses = [testModel(testBatchIndex) for testBatchIndex in xrange(nTestBatches)]
                testScore = np.mean(testLosses)

                print(('     epoch %i, minibatch %i/%i, test error of '
                       'best model %f %%') %
                      (epoch, minibatchIndex + 1, nTrainBatches,
                       testScore * 100.))

        if patience <= iter:
            doneLooping = True
            break

endTime = timeit.default_timer()
print(('Optimization complete. Best validation score of %f %% '
       'obtained at iteration %i, with test performance %f %%') %
      (bestValidationLoss * 100., bestIter + 1, testScore * 100.))

print >> sys.stderr, ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((endTime - startTime) / 60.))

