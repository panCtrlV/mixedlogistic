__author__ = 'panc'

from LogisticRegression import LogisticRegression
from HiddenLayer import HiddenLayer
from DecisionLayer import DecisionLayer


class MixedLogisticRegression(object):
    def __init__(self, input_for_decision_layer, c, n_in_for_decision_layer, n_out,
                 p_membership=None, input_for_hidden_layer=None, n_in_for_hidden_layer=None):
        self.hiddenLayer = HiddenLayer(c, p_membership,
                                       input_for_hidden_layer, n_in_for_hidden_layer)

        self.decisionLayer = DecisionLayer(c, input_for_decision_layer,
                                           n_in_for_decision_layer, n_out)

        self.params = self.hiddenLayer.params + self.decisionLayer.params

    def conditionalMembershipProbability(self, y):
        conditionalLikelihood = self.decisionLayer.conditionalLikelihood(y)
        p_membership = self.hiddenLayer.p_membership
        # if self.hiddenLayer.isHierarchical:
        #     p_membership = self.hiddenLayer.p_membership
        # else:
        #     p_membership = self.hiddenLayer.params

        jointProbabilities = conditionalLikelihood * p_membership
        return (jointProbabilities.T / T.sum(jointProbabilities, axis=1)).T

    def negative_expected_log_likelihood(self, y):
        """
        Minimization target for M-step

        :param y:
        :return:
        """
        lp_membership = T.log(self.hiddenLayer.p_membership)
        lp_decision = T.log(self.decisionLayer.conditionalLikelihood(y))
        p_conditionalMembership = self.conditionalMembershipProbability(y)
        return -T.sum(p_conditionalMembership * (lp_membership + lp_decision))

    def negative_log_likelihood(self, y):
        p_membership = self.hiddenLayer.p_membership
        p_decision = self.decisionLayer.conditionalLikelihood(y)
        return -T.sum(T.log(T.sum(p_decision * p_membership, axis=1)))
        # return p_membership
        # return p_decision


# === Test code ===
if __name__ == '__main__':
    from data.mnist_pkl import *

    data = readData_mnist_pkl()
    trainX = data['trainx']
    trainY = data['trainy']
    trainY = T.cast(trainY, dtype='int32')

    input1 = T.dmatrix()  # input for hidden layer
    input2 = T.dmatrix()  # input for the decision layer
    y = T.ivector('y')

    mixedLogisticRegression = MixedLogisticRegression(input_for_decision_layer=input2, c=3,
                                                      n_in_for_decision_layer=27 * 28, n_out=10,
                                                      input_for_hidden_layer=input1,
                                                      n_in_for_hidden_layer=28)

    conditionalProbabilities = mixedLogisticRegression.conditionalMembershipProbability(y)

    f1 = theano.function(inputs=[],
                         outputs=conditionalProbabilities,
                         givens={
                             input1: trainX[:, : 28],
                             input2: trainX[:, 28:],
                             y: trainY
                         })

    res = f1()
    print res
    print res.shape

    nllk1 = mixedLogisticRegression.negative_log_likelihood(y)
    f_nllk1 = theano.function(inputs=[],
                              outputs=nllk1,
                              givens={
                                  input1: trainX[:, : 28],
                                  input2: trainX[:, 28:],
                                  y: trainY
                              },
                              on_unused_input='ignore')

    res_nllk1 = f_nllk1()
    print 'negative log-likelihood =', res_nllk1

    p = T.dvector()
    p_membership = theano.shared(np.array([0.3, 0.4, 0.3]))
    mixedLogisticRegression2 = MixedLogisticRegression(input_for_decision_layer=input2, c=3,
                                                       n_in_for_decision_layer=28 * 28, n_out=10,
                                                       p_membership=p)
    conditionalProbabilities2 = mixedLogisticRegression2.conditionalMembershipProbability(y)
    f2 = theano.function(inputs=[],
                         outputs=conditionalProbabilities2,
                         givens={
                             input2: trainX,
                             y: trainY,
                             p: p_membership
                         })
    res2 = f2()
    print res2
    print res2.shape

    nllk2 = mixedLogisticRegression2.negative_log_likelihood(y)
    f_nllk2 = theano.function(inputs=[],
                              outputs=nllk2,
                              givens={
                                  input2: trainX,
                                  y: trainY,
                                  p: p_membership
                              },
                              on_unused_input='ignore')
    res_nllk2 = f_nllk2()
    print 'negative log-likelihood =', res_nllk2
