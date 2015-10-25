import numpy as np


class Paramters(object):
    """
    Encapsulate the parameters needed for a mixed logistic regression model.
    In the hidden layer, the last group is assumed to be the base group whose
    parameters are forced to be 0s.
    In the observed layer, the response is assumed to be binomial.
    """
    def __init__(self, Alpha, Beta, a=None, b=None):
        """
        Construct a new Parameters object.

        :type Alpha: 2d numpy array
        :param Alpha: parameter matrix for hidden layer covariates. Since the last group is
                      assumed to be the base group, its parameters are forced to be 0s and
                      excluded from the parameter matrix. Each column is corresponding to one
                      group.

        :type Beta: 2d numpy array
        :param Beta: parameter matrix for observed layer covariates. Since the response is
                     binomial (resp. binary), there is only one binomial (resp. bernoulli)
                     probability to be modeled by a sigmoid function (equivalently logit as
                     the link function) for one hidden group. Each column is corresponding to
                     the parameters of a sigmoid.

        :type a: None or 1d numpy array
        :param a: hidden layer intercepts. The default value is None, when the intercept is
                  not used in modelling a logit as linear regression. When it is not None,
                  intercept is activated in the regression. Since the last group is assumed to
                  be the base group and its intercept is forced to be 0, the size of the array
                  should be one fewer than the number of hidden groups.

        :type b: None or 1d numpy array
        :param b: observed layer intercepts. The default value is None, when the intercept is
                  not used in a logistic regression. When it is not None, each logistic regression
                  in a hidden group requires one intercept, so the size of the array is the
                  same as the number of hidden groups.

        :return: nothing
        """
        self.Alpha = Alpha
        self.Beta = Beta
        self.a = a  # hidden layer intercepts
        self.b = b  # observed layer intercepts
        self.hasHiddenIntercepts = a is not None
        self.hasObservedIntercepts = b is not None
        self.dxm = Alpha.shape[0]
        self.dxr, self.c = Beta.shape  # number of hidden groups

    def getParameters(self, layer, groupNumber):
        """
        By specifying the layer and hidden group number, it returns the
        logistic regression coefficients for the specified layer and group.

        :type layer: int
        :param layer: layer number. 1 - hidden, 2 - observed.

        :type groupNumber: int
        :param groupNumber: hidden group number. Group index starts from 1.
                            The maximum group number equals the total number of

        :return: parameter list for the specified layer and group.
        """
        if groupNumber <= 0 or groupNumber > self.c:
            raise ValueError("** 'groupNumber' is out of bounds. **")
        else:
            if layer == 1:  # hidden layer
                if self.a is None:
                    print "** Hidden layer has no intercept. **"
                    if groupNumber == self.c:
                        return np.zeros(self.dxm)
                    else:
                        return self.Alpha[:, groupNumber - 1]
                else:
                    if groupNumber == self.c:
                        return np.zeros(self.dxm + 1)
                    else:
                        return np.hstack([self.a[groupNumber - 1], self.Alpha[:, groupNumber - 1]])

            elif layer == 2:  # observed layer
                if self.b is None:
                    print "** Observed layer has no intercept. **"
                    return self.Beta[:, groupNumber - 1]
                else:
                    return np.hstack([self.b[groupNumber - 1], self.Beta[:, groupNumber - 1]])
            else:
                raise ValueError("** Layer is out of bound. **")

    # This method is not intended to be inherited
    def flattenHiddenLayerParameters(self):
        """
        Squeeze / flatten hidden layer parameters such that the intercept (if `a` is not None)
        comes before the coefficients (`Alpha`) within each group. Then the grouped parameters
        are concatenated. This is shown below:

            a, Alpha => [a_1, Alpha_{11}, Alpha_{12}, ...,
                         a_2, Alpha_{21}, Alpha_{22}, ..., ,
                         ...]

        :return: flattened hidden layer parameters
        """
        if self.hasHiddenIntercepts:
            return np.vstack([self.a, self.Alpha]).flatten(order='F')
            # return np.hstack([self.a, self.Alpha.flatten(order='F')])
        else:
            return self.Alpha.flatten(order='F')

    # This method is not intended to be inherited
    def flattenObservedLayerParametersj(self, j):
        """
        Squeeze / flatten observed layer parameters for the j_th hidden group such that
        the intercept comes before the coefficients for the covariates. This shown below:

            b, Beta, j_th group => [b_j, Beta_{1j}, Beta_{2j}, ...]

        :param j:
        :return: flattened observed layer parameters for j_th hidden group
        """
        if self.hasObservedIntercepts:
            return np.hstack([self.b[j], self.Beta[:, j]])
        else:
            return self.Beta[:, j]

    # This method is not intended to be inherited
    def flatten(self):
        """
        Create a flatten list of the parameters in the model. The concatenation order is given
        as follows:

            a1, Alpha1, a2, Alpha2, ..., b1, Beta1, b2, Beta2, ...

        :return: flattened model parameters
        """
        vHiddenLayerParams = self.flattenHiddenLayerParameters()
        vObservedLayerParams = np.array([self.flattenObservedLayerParametersj(j) for j in range(self.c)]).flatten()
        return np.hstack([vHiddenLayerParams, vObservedLayerParams])

    def normedDifference(self, that):
        """
        Calculate the L2 norm of the difference between the current parameter list and another
        given parameter list. This is mainly used to check parameter convergence
        in the EM algorithm.

        :type that: Paramters
        :param that: another set of parameters to compare to.

        :return: L2 norm of the difference between two sets of parameters.
        """
        # if not isinstance(that, Paramters)
        #     raise("The argument is not of type Parameters.")
        return np.linalg.norm(self.flatten() - that.flatten())

    # This method is not intended to be inherited
    def getOrderedParameters(self, trueParameters):
        """
        Re-order the groups of parameters to be conformable to the given true parameters. After
        re-ordering, the L2 norm between the two sets of parameters belonging to the same group
        is minimized.

        This is mainly used in simulation studies, since the parameters of a mixed logistic
        regression are identifiable up to a permutation of groups.

        By re-ordering the groups, the base group parameters (i.e. 0s) are augmented.

        :type trueParameters: Paramters
        :param trueParameters: the true parameter set whose group order the current parameter
                               set is trying to be conformable to.

        :type return: OrderedParameters
        :return: an OrderedParameters object encapsulating the re-ordered parameters.
        """
        return OrderedParameters(self.Alpha, self.Beta, self.a, self.b, trueParameters)

    def getAlphaFromFlatInput(self, flatAlpha):
        pass

    def getBetajFromFlatInput(self, flatBetaj):
        pass

    def __str__(self):
        line1 = "Hidden Layer:"
        line2 = "\tGroup\tIntercept\t" + '\t'.join(['alpha'+str(i) for i in np.arange(self.dxm) + 1])
        # Construct print strings for hidden layer parameters
        # by collecting as a list of string lines where each
        # line corresponds to one hidden group
        hidden_parameter_lines = []
        for j in range(self.c - 1):
            if self.hasHiddenIntercepts:
                parameter_line = \
                    ( '\t{' + '}\t\t{'.join(str(i) for i in range(2 + self.dxm)) + '}' ).format( *((j+1, self.a[j],) + tuple(self.Alpha[:, j])) )
            else:
                parameter_line = \
                    ( '\t{' + '}\t\t{'.join(str(i) for i in range(2 + self.dxm)) + '}' ).format( *( (j+1, None,) + tuple(self.Alpha[:, j]) ) )
            hidden_parameter_lines.append(parameter_line)
        # Base group parameters are forced to 0s
        base_hidden_parameter_line = ('\t{0}' + '\t\t{1}'*(1 + self.dxm)).format(self.c, 0.)
        hidden_parameter_lines.append(base_hidden_parameter_line)
        # Insert line breaks between lines
        line3 = '\n'.join(hidden_parameter_lines)

        line4 = "Observed Layer:"
        line5 = "\tGroup\tIntercept\t" + '\t'.join(['beta'+str(i) for i in np.arange(self.dxr) + 1])
        # Construct print strings for observed layer parameters
        # by collecting as a list of string lines where each
        # line corresponds to one hidden group
        observed_parameter_lines = []
        for j in range(self.c):
            if self.hasObservedIntercepts:
                parameter_line = \
                    ( '\t{' + '}\t\t{'.join(str(i) for i in range(2 + self.dxr)) + '}' ).format( *((j+1, self.b[j],) + tuple(self.Beta[:, j])) )
            else:
                parameter_line = \
                    ( '\t{' + '}\t\t{'.join(str(i) for i in range(2 + self.dxr)) + '}' ).format( *( (j+1, None,) + tuple(self.Beta[:, j]) ) )
            observed_parameter_lines.append(parameter_line)
        # Insert line breaks between lines
        line6 = '\n'.join(observed_parameter_lines)

        return '\n'.join([line1, line2, line3, line4, line5, line6])


class OrderedParameters(Paramters):
    def __init__(self, Alpha, Beta, a, b, trueParameters):
        Paramters.__init__(self, Alpha, Beta, a, b)
        self._reorderGroups(trueParameters)

    def _reorderGroups(self, trueParameters):
        """
        Mixture model is identifiable up to the permutation of groups.
        In order to ensure unique ordering, the estimated groups are
        ordered such that the L2 distance between a give group and the
        the corresponding true group is minimized.

        :type trueParameters: Paramters
        :param trueParameters:
        """
        if self.c != trueParameters.c:
            raise ValueError("** The number of groups for the two parameters are not conformable. **")
        if self.dxm != trueParameters.dxm:
            raise ValueError("** The hidden layer dimensions for the two parameters are not conformable. **")
        if self.dxr != trueParameters.dxr:
            raise ValueError("** The observed layer dimensions for the two parameters are not conformable. **")

        # Re-organize parameters as a (dimension * number_of_groups) matrix
        thisParamsMtx = np.vstack(
                            np.hstack(
                                [self.getParameters(1, i), self.getParameters(2, i)]
                            ) for i in (np.arange(self.c) + 1)
                        )
        trueParamsMtx = np.vstack(
                            np.hstack(
                                [trueParameters.getParameters(1, i), trueParameters.getParameters(2, i)]
                            ) for i in (np.arange(trueParameters.c) + 1)
                        )

        # For each estimated group, calculate L2 norm against all true groups
        l2_dist = []
        for i in range(self.c):
            l2_dist_i = []
            for j in range(self.c):
                l2_dist_i.append(np.linalg.norm(thisParamsMtx[i, :] - trueParamsMtx[j, :]))
            l2_dist.append(l2_dist_i)

        # Find groups each estimates should belong
        l2_dist_array = np.array(l2_dist)
        groupIndices = l2_dist_array.argmin(axis=0)

        # Re-order parameters among groups
        aWithBaseGroup = np.append(self.a, 0.)
        if self.hasHiddenIntercepts:
            self.a = aWithBaseGroup[groupIndices]
        if self.hasObservedIntercepts:
            self.b = self.b[groupIndices]
        AlphaWithBaseGroup = np.append(self.Alpha, np.zeros((self.dxm, 1)), axis=1)
        self.Alpha = AlphaWithBaseGroup[:, groupIndices]
        self.Beta = self.Beta[:, groupIndices]
