import numpy as np
from scipy.special import expit
from scipy.misc import logsumexp


def sigmoid(x):
    return expit(x)


def sigmoidForData(X, coef, intercept=None, returnFull=False):
    if coef.ndim == 1:  # coef is 1d array
        if intercept is None:
            linearTerms = X.dot(coef)
        else:
            if isinstance(intercept, float):
                linearTerms = X.dot(coef) + intercept  # intercept should be a float
            else:
                raise("'coef' is 1-dimensional, which requires a scalar intercept."
                      "Non-scalar intercept is not yet implemented.")

        sigmoids = sigmoid(linearTerms)
        if returnFull:
            return np.vstack((sigmoids, 1-sigmoids)).T
        else:
            return sigmoids
    else:  # coef is 2d array
        if intercept is None:
            linearTerms = X.dot(coef)
        else:
            if len(intercept) == coef.shape[1]:
                linearTerms = X.dot(coef) + intercept
            else:
                ValueError("The shapes of 'coef' and 'intercept' are not conformable. "
                           "'coef' has shape %s and 'intercept' has shape %s." % (coef.shape, intercept.shape))
        return sigmoid(linearTerms)


def softmax(x, addDefaultBase=True, axis=1, returnLog=False):
    # x is a 1d or 2d array
    if x.ndim == 1:
        if addDefaultBase:
            x = np.append(x, 0)
        logSumExp = logsumexp(x)
        logSoftMax = x - logSumExp
        if returnLog:
            return logSoftMax
        else:
            return np.exp(logSoftMax)
    else:
        if addDefaultBase:
            x = np.append(x, np.zeros((x.shape[0], 1)), axis=1)
        logSumExp = logsumexp(x, axis, keepdims=True)
        logSoftMax = x - logSumExp
        if returnLog:
            return logSoftMax
        else:
            return np.exp(logSoftMax)


def softmaxForData(X, Coef, intercepts=None, addDefaultBase=True, returnLog=False):
    if intercepts is None:
        linearTerms = X.dot(Coef)
    else:
        linearTerms = X.dot(Coef) + intercepts

    return softmax(linearTerms, addDefaultBase, returnLog=returnLog)
