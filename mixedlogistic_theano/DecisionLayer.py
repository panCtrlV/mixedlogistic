__author__ = 'panc'

"""
Given the number of hidden groups c, there are c different
logistic regression models in the decision layer. Each model
is equipped with its own parameters. They share the same input
data.
"""


import theano
import theano.tensor as T

from LogisticRegression import LogisticRegression


class DecisionLayer(object):
    def __init__(self, c, input, n_in, n_out):
        self.input = input
        self.c = c

        self.logisticComponents = [LogisticRegression(input, n_in, n_out) for i in range(c)]

        self.params = [logisticComponent.params for logisticComponent in self.logisticComponents]

        # Conditional on a sample belonging to a hidden group,
        # likelihood can be calculated for that sample.

        # Each logistic component has a p_y_given_x theano matrix,
        # the likelihood for each sample is the element in each row
        # indexed by the label.

    def conditionalLikelihood(self, y):
        conditionalLikelihoods = [logisticComponent.p_y_given_x[T.arange(y.shape[0]), y]
                                  for logisticComponent in self.logisticComponents]
        return T.concatenate(conditionalLikelihoods).reshape((self.c, y.shape[0])).T
        # It seems theano reshape fills row first. In order to make each column correspond
        # to one component, we need to transpose the matrix.


# === Test code ===
if __name__ == '__main__':
    from data.mnist_pkl import *
    data = readData_mnist_pkl()
    trainX = data['trainx']
    trainY = data['trainy']
    trainY = T.cast(trainY, dtype='int32')

    x = T.dmatrix('x')
    y = T.ivector('y')
    decisionLayer = DecisionLayer(c=3, input=x, n_in=28*28, n_out=10)
    conditionalLikelihood = decisionLayer.conditionalLikelihood(y)
    f1 = theano.function(inputs=[],
                         outputs=conditionalLikelihood,
                         givens={
                             x : trainX,
                             y : trainY
                         })
    res1 = f1()
    print res1
    print type(res1)
    print res1.shape

    conditionalLikelihood0 = decisionLayer.logisticComponents[0].p_y_given_x[T.arange(y.shape[0]), y]
    f2 = theano.function(inputs=[],
                         outputs=conditionalLikelihood0,
                         givens={
                             x : trainX,
                             y : trainY
                         })
    res2 = f2()
    print res2
    print type(res2)
    print res2.shape
