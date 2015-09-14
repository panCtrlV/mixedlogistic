__author__ = 'panc'

"""
Generating data from mixed logistic model
"""
from preprocess_data import *
from cal_condprob import *


def addBaselineCoefficients(coefMtx):
    """
    Append a 0 column to the input, which would be the coefficients for the baseline group.

    :param coefMtx:
        numpy matrix. If the covariate dimension is p, and number of components is c, then
        the matrix is of size p * (c-1).
    :return:
        numpy matrix, of size p * c, where the last column contains all 0s.
    """
    m, n = coefMtx.shape
    fullMtx = np.zeros((m, n+1))
    fullMtx[:,:-1] = coefMtx
    return fullMtx

# sample size
n = 100
# xm dimension
dxm = 3
# number of components
c = 3
# xr dimension
dxr = 3
# Binomial number of trials (same for all observations)
m = 50

# generate xm from independent normal
xm = []
sd = [1., 2., 3.]
for i in range(len(sd)):
    xm.append(np.random.normal(0., sd[i], n))
xm = np.matrix(np.vstack(xm)).T
xm = addIntercept(xm)

# beta coefficients for the mixing probability
beta = []
mu = [-10, 10]
for i in range(len(mu)):
    beta.append(np.random.normal(mu[i], 6., dxm+1))
beta = np.matrix(np.vstack(beta)).T
beta = addBaselineCoefficients(beta)

# Calculate the membership probabilities
pxb = cal_softmaxForData(xm, beta)
groupid = pxb.argmax(axis=1)  # 0-based group membership

# generate xr from independent normal
xr = []
sd2 = [0.01, 0.2, 3]
for i in range(len(sd2)):
    xr.append(np.random.normal(0., sd2[i], n))
xr = np.matrix(np.vstack(xr)).T
xr = addIntercept(xr)

# alpha coefficient for the components
alpha = []
mu2 = [1., -5, 5, 10]  # dxr = 3
scale = [0.1, 1., 10]
mu2 = np.outer(mu2, scale)  # each column contains the coefficients for one group, including the intercept
for j in range(mu2.shape[1]):
    alpha.append( np.random.normal(mu2[:,j], 0.5) )
alpha = np.matrix(np.vstack(alpha)).T

# conditional component probabilities
x = np.array(cal_sigmoid(xr * alpha))
pxa = x[np.arange(n), groupid]