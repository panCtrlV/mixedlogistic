import theano.tensor as T

from MixedLogisticRegression import MixedLogisticRegression
from HiddenLayer import HiddenLayer
from DecisionLayer import DecisionLayer


class MixedLogisticRegressionEM(MixedLogisticRegression):
    """
    Class of EM algorithm
    """
    def __init__(self, input_for_decision_layer, c, n_in_for_decision_layer, n_out,
                 p_membership=None, input_for_hidden_layer=None, n_in_for_hidden_layer=None):
        MixedLogisticRegression.__init__(self, input_for_decision_layer, c, n_in_for_decision_layer, n_out,
                                         p_membership, input_for_hidden_layer, n_in_for_hidden_layer)

        self.hiddenLayer2 = HiddenLayer(c, p_membership, input_for_hidden_layer, n_in_for_hidden_layer)
        self.decisionLayer2 = DecisionLayer(c, input_for_decision_layer, n_in_for_decision_layer, n_out)

        self.params = self.hiddenLayer2.params + self.decisionLayer2.params  # parameter for updating

    def negative_expected_log_likelihood(self, y):
        """

        :param y:
        :return:
        """
        lp_membership = T.log(self.hiddenLayer2.p_membership)
        lp_decision = T.log(self.decisionLayer2.conditionalLikelihood(y))
        p_conditionalMembership = self.conditionalMembershipProbability(y)  # inherited from MixedLogisticRegression
        return -T.sum(p_conditionalMembership * (lp_membership + lp_decision))

    def m_step(self):
        # Pylearn2 may be helpful
        # Reference: http://deeplearning.net/software/pylearn2/index.html#
        pass