import theano
import theano.tensor as T

from LogisticRegression import LogisticRegression


class HiddenLayer(object):
    def __init__(self, c, p_membership=None, input=None, n_in=None):
        """

        :param c:
        :param p_membership: symbolic variable
        :param input:
        :param n_in:
        :return:
        """
        if (p_membership is None) and (input is None):
            raise ValueError(" The membership probability 'p_membership' ' \
                             'and data 'input' cannot be both None. One ' \
                             'of them has to be specified. ")

        if p_membership is not None:  # do not model mixing probabilities
            self.isHierarchical = False
            self.p_membership = p_membership
            self.params = p_membership  # list or 1d numpy.ndarray
        else:
            self.isHierarchical = True
            hiddenLayerModel = LogisticRegression(input, n_in, c)
            self.params = hiddenLayerModel.params  # W and b
            self.p_membership = hiddenLayerModel.p_y_given_x
            self.input = input
