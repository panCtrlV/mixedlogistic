import numpy as np

import cal_condprob
from helper import *


"""
After a mixed logistic model is trained,
make prediction on the test data and calculate performance measures.
"""


def mixedLogistic_pred(res, xmt, xrt, c, yt):
    lxm = xmt.shape[1]
    b, a = mapOptimizationVector2Matrices(res['param'], c, lxm)

    # calculate mixing probabilities
    bWithZero = appendZeros(b)
    pi = cal_condprob.cal_softmaxForData(xmt, bWithZero)
    # calculate component probabilities
    p = cal_condprob.cal_sigmoidForData(xrt, a)
    # weight average is the classification rule
    predictedLabels = (np.multiply(pi, p).sum(axis=1) > .5).astype(int)

    # precision and recall as performance measures
    predictedTrue = (predictedLabels == 1)
    groundTrue = (yt == 1)
    truePositive = np.hstack([predictedLabels == 1, yt == 1]).all(axis=1)

    precision = truePositive.sum() * 1.0 / groundTrue.sum()
    recall = truePositive.sum() * 1.0 / predictedTrue.sum()
    f1 = 2. * precision * recall / (precision + recall)

    return dict(pred=predictedLabels, precision=precision, recall=recall, f1=f1)
