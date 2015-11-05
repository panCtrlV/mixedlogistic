from mixedlogistic_separate.mixedLogistic_rawScript import *
from mixedlogistic_separate.DataSimulators import *


"""
Evaluating empirical fisher information

All indices in the function definition start from 0, e.g. j
"""


def pj(x, coefj, interceptj):
    """
    x is 2d, coef is 1d, intercept is scalar

    :param x:
    :param coefj:
    :param interceptj: can be None
    :return:
    """

    return sigmoidForData(x, coefj, interceptj)  # 1d


def Aj(pj, pc, y):
    # pj is 1d, pc is 1d, y is 1d
    # they have the same size
    # one example has one value
    return (pc - pj) * (-1)**y  # y=0 => pc-pj; y=1 => pj-pc; 1d

# y = np.array([0,1,0,0,1,1,1])
# (-1)**y
# pc = np.arange(7)
# pj = np.arange(7) + 10
# (pc - pj) * (-1)**y

# simulator = MixedLogisticDataSimulator(100, 3, 3, 3)
# data = simulator.data
# parameters = simulator.parameters



def pie(x, Coef, intercepts, j):
    # x is 2d, Coef is 2d, intercept is 1d
    # ncol of X == nrow of Coef
    # ncol of Coef == len of intercept
    probs = softmaxForData(x, Coef, intercepts)  # 2d, nrow_of_x * (ncol_of_Coef + 1)
    if j is not None:
        return probs[:, j-1]
    else:
        return probs


# element-wise sigmoid gradients
def pj_grad(x, coefj, interceptj):
    """
    Element-wise sigmoid gradient, i.e. one for each sample.

    :param x:
    :param coefj:
    :param interceptj:

    :type return: 2d array, of size n * number_of_observed_layer_coef (including intercept)
    :return: element-wise gradient of sigmoid wrt coefficient. One row is a gradient for one sample.
    """
    probj = pj(x, coefj, interceptj)
    w = probj * (1 - probj)
    # gradj = x.T.dot(w)
    gradj = x * w[:, np.newaxis]
    if interceptj is not None:
        # gradj = np.append(w.sum(), gradj)
        gradj = np.hstack([w[:, np.newaxis], gradj])
    return gradj

# x = np.random.random(15).reshape(5,3)
# coef = np.array([1,2,3])
# intercept = 1.
# pj_grad(x, coef, interceptj=None)


# element-eise softmax gradients
def piej_gradk(x, Coef, intercepts, j, k, returnElementwise=True):
    """
    x: 2d, Coef: 2d, intercepts: 1d
    ncol of X == nrow of Coef
    ncol of Coef == len of intercept

    **Sample-wise** derivative of \pi_j wrt \alpha_k,
    i.e. first-order derivative of softmax function, with linear exponents.
    if j==k => \pi_j * (1 - \pi) * x
    if j!=k => -\pi_j * \pi_k * x

    :type return: 2d array, with size n * number_of_hidden_layer_coef (including intercept)
    :return: sample-wise evaluation of \pi_j wrt \alpha_k
    """
    piej = pie(x, Coef, intercepts, j)
    piek = piej if j==k else pie(x, Coef, intercepts, k)
    w = piej * ((j==k) - piek)
    if returnElementwise:
        gradk = x * w[:, np.newaxis]  # each row is the gradient for a sample
        if intercepts is not None:
            gradk = np.hstack([w[:, np.newaxis], gradk])
        return gradk
    else:
        gradk = x.T.dot(w)
        if intercepts is not None:
            gradk = np.append(w.sum(), gradk)
        return gradk

# np.random.seed(25)
# x = np.random.random(15).reshape(5,3)
# Coef = np.array([[1,2,3], [3,2,1]]).T
# intercepts = np.array([-1, -2])
# piej_gradk(x, Coef, intercepts, 1, 1, True).sum(0)


def loglikelihood():
    pass


def B(data, params):
    """
    A list of likelihoods, one for each sample.

    Coded copied from `Loglikelihood.observedLogLikelihood` except 'return' clause.
    """
    n = data.n
    m = data.m

    memberProbs = softmaxForData(data.Xm, params.Alpha, params.a, True)
    choosingProbs = sigmoidForData(data.Xr, params.Beta, params.b)
    if isinstance(m, int):
        if m == 1:
            observedLikelihoods = np.abs((1. - data.y).reshape(n, 1) - choosingProbs)
        else:
            observedLikelihoods = binom.pmf(data.y.reshape(n, 1),
                                            np.repeat(m, n).reshape(n, 1),
                                            choosingProbs)
    else:
        raise NotImplementedError("Function is not yet implemented for non-integer 'm'.")

    jointProbs = np.multiply(memberProbs, observedLikelihoods)
    return jointProbs.sum(axis=1)  # 1d array, one value for each sample

# simulator = MixedLogisticDataSimulator(10, 3, 2, 3)
# simulator.simulate(25)
# data = simulator.data
# parameters = simulator.parameters
# B(data, parameters)


###########################
# First order derivatives #
###########################
def grad_alphaj(data, parameters, j, returnElementwise=False):
    """
    First-order derivative of log-likelihood wrt \alpha_j.

    Formula (one sample):

        \frac{\partial \ell}{\partial \alpha_j}
            &= \sum_{k=1}^{c-1} \frac{\partial \ell}{\partial \pi_k} * \frac{\partial \pi_k}{\partial \alpha_j} \\
            &= \frac{A_k}{B} * \frac{\partial \pi_k}{\partial \alpha_j}

    :type data: Data
    :param data:

    :type parameters: Parameters
    :param parameters:
    :param j:
    :return:
    """
    y = data.y
    z = data.Xm
    x = data.Xr
    Alpha = parameters.Alpha
    a = parameters.a  # can be None
    Beta = parameters.Beta
    b = parameters.b  # can be None
    c = parameters.c
    n = data.n  # sample size
    pHiddenLayer = parameters.dxm if parameters.a is None else parameters.dxm + 1

    bc = None if b is None else b[c-1]
    probc = pj(x, Beta[:, c-1], bc)  # 1d array, n-vector
    B_ = B(data, parameters)  # 1d array, n-vector

    grad_alphaj_ = np.zeros((n, pHiddenLayer))  # initialize result
    for k in np.arange(c-1) + 1:
        bk_ = None if b is None else b[k-1]
        probk = pj(x, Beta[:, k-1], bk_)  # 1d array, n-vector
        Ak_ = Aj(probk, probc, y)  # 1d array, n-vector
        piek_gradj_ = piej_gradk(z, Alpha, a, k, j)  # 2d array, n * number_of_hidden_layer_coef (including intercept)
                                                     # `a` can be None
        grad_alphaj_ += piek_gradj_ * Ak_[:, np.newaxis]
    grad_alphaj_ /= B_[:, np.newaxis]

    if returnElementwise:
        return grad_alphaj_  # each row is the gradient for one sample
    else:
        return grad_alphaj_.sum(0)

# simulator = MixedLogisticDataSimulator(10, 3, 2, 3)
# simulator.simulate(25)
# data = simulator.data
# parameters = simulator.parameters
# grad_alphaj(data, parameters, 1)
# # or return element-wise
# grad_alphaj(data, parameters, 1, True)


def grad_betaj(data, parameters, j, returnElementwise=False):
    y = data.y
    z = data.Xm
    x = data.Xr
    a = parameters.a
    Alpha = parameters.Alpha
    b = parameters.b
    Beta = parameters.Beta

    piej_ = pie(z, Alpha, a, j)
    B_ = B(data, parameters)

    bj_ = None if b is None else b[j-1]
    pj_gradj_ = pj_grad(x, Beta[:, j-1], bj_)
    grad_betaj_ = (2*y-1)[:, np.newaxis] * piej_[:, np.newaxis] / B_[:, np.newaxis] * pj_gradj_

    if returnElementwise:
        return grad_betaj_
    else:
        return grad_betaj_.sum(0)

# simulator = MixedLogisticDataSimulator(10, 3, 2, 3)
# simulator.simulate(25)
# data = simulator.data
# parameters = simulator.parameters
# grad_betaj(data, parameters, 3)
# # or return element-wise
# grad_alphaj(data, parameters, 3, True)


############################
# Second order derivatives #
############################
def piej_hesskl(x, Coef, intercepts, j, k, l, returnElementwise=False):
    """
    Hessian of \pi_j wrt \alpha_k, \alpha_l

    Formula:
        \frac{\partial^2 \pi_j}{\partial \alpha_k \partial \alpha_l}
            = [\pi_j * (\delta_{jl} - \pi_l) * (\delta_{jk} - \pi_k) - \pi_j * \pi_k * (\delta_{kl} - \pi_l)] * z*z^T

    :param x:
    :param Coef:
    :param intercepts:
    :param j:
    :param k:
    :param l:

    :type return: 3d array, of size n * number_of_hidden_coef * number_of_hidden_coef (including intercept)
    :return: elemnent/sample-wise second-order derivative for softmax. Each element in the first dimension
             is the hessian matrix for a sample.
    """
    piej_ = pie(x, Coef, intercepts, j)  # 1d array
    piek_ = piej_ if j==k else pie(x, Coef, intercepts, k)  # 1d array
    if l==j:
        piel_ = piej_  # 1d array
    elif l==k:
        piel_ = piek_  # 1d array
    else:
        piel_ = pie(x, Coef, intercepts, l)  # 1d array

    w = piej_ * ((j==k) - piek_) * ((j==l) - piel_) - piej_ * piek_ * ((k==l) - piel_)  # 1d array
    # construct outer product for each sample covariate vector
    xxlist = []
    for i in xrange(x.shape[0]):
        xi = x[i, :]  # 1d array
        if intercepts is not None:
            xi = np.append(1., xi)
        xxi = np.outer(xi, xi)[np.newaxis, :, :]
        xxlist.append(xxi)
    xx = np.vstack(xxlist)  # 3d array, of size n * number_of_hidden_coef * number_of_hidden_coef (including intercept)
                            # Each element in the outer product of a sample covaraite vector.

    piej_hesskl_ = w[:, np.newaxis, np.newaxis] * xx  # sample/element-wise hessian

    if returnElementwise:
        return piej_hesskl_  # 3d array
    else:
        return piej_hesskl_.sum(0)


# x = np.random.random(15).reshape(5,3)
# Coef = np.array([[1,2,3], [3,2,1]]).T
# intercepts = np.array([-1, -2])
# for j in [1,2,3]:
#     for k in [1,2,3]:
#         for l in [1,2,3]:
#             print j, ' ', k, ' ', l
#             print piej_hesskl(x, Coef, intercepts, j, k, l), '\n'

# piej_hesskl(x, Coef, intercepts, 1, 2, 3, True)


def hess_alphajk(data, parameters, j, k, returnElementwise=False):
    n = data.n
    y = data.y
    z = data.Xm
    x = data.Xr
    c = parameters.c
    a = parameters.a
    Alpha = parameters.Alpha
    b = parameters.b
    Beta = parameters.Beta
    pHiddenLayer = parameters.dxm if parameters.a is None else parameters.dxm + 1

    bc = None if b is None else b[c-1]
    probc = pj(x, Beta[:, c-1], bc)  # 1d array, n-vector
    B_ = B(data, parameters)  # 1d array, n-vector

    sum1 = np.zeros((n, pHiddenLayer, pHiddenLayer))
    for l in np.arange(c-1) + 1:
        bl_ = None if b is None else b[l-1]
        probl = pj(x, Beta[:, l-1], bl_)  # 1d array, n-vector
        Al_ = Aj(probl, probc, y)  # 1d array, n-vector
        piel_hessjk_ = piej_hesskl(z, Alpha, a, l, j, k, True)  # 3d array
        wl = Al_ / B_
        piel_hessjk_ *= wl[:, np.newaxis, np.newaxis]
        sum1 += piel_hessjk_

    sum2 = np.zeros((n, pHiddenLayer, pHiddenLayer))
    for l1 in np.arange(c-1) + 1:
        bl1_ = None if b is None else b[l1-1]
        probl1 = pj(x, Beta[:, l1-1], bl1_)
        Al1_ = Aj(probl1, probc, y)  # 1d array
        piel1_gradj_ = piej_gradk(z, Alpha, a, l1, j, True)  # 2d array
        for l2 in np.arange(c-1) + 1:
            bl2_ = None if b is None else b[l2-1]
            probl2 = pj(x, Beta[:, l2-1], bl2_)
            Al2_ = Aj(probl2, probc, y)  # 1d array
            piel2_gradk_ = piej_gradk(z, Alpha, a, l2, k, True)  # 2d array
            wl1l2 = Al1_ * Al2_ / B_**2  # 1d array
            # prepare outer products of gradients
            gg2dlist = [np.outer(piel1_gradj_[i, :], piel2_gradk_[i, :]) for i in xrange(n)]
            gg3dlist = [gg2d[np.newaxis, :, :] for gg2d in gg2dlist]
            gg3d = np.vstack(gg3dlist)  # 3d array
            # incremental addition
            sum2 += wl1l2[:, np.newaxis, np.newaxis] * gg3d

    sum1_minus_sum2_ = sum1 - sum2
    if returnElementwise:
        return sum1_minus_sum2_  # 3d array
    else:
        return sum1_minus_sum2_.sum(0)  # 2d array


# simulator = MixedLogisticDataSimulator(10, 3, 2, 3)
# simulator.simulate(25)
# data = simulator.data
# parameters = simulator.parameters
# hess_alphajk(data, parameters, 1, 2, True)


def pj_hesskl(x, Coef, intercepts, j, k, l, returnElementwise=False):
    """
    Second derivative of sigmoid.

    :param x:
    :param Coef:
    :param intercepts:
    :param j:
    :param k:
    :param l:
    :return:
    """
    pass


def hess_betajk(data, parameters, j, k, returnElementwise=False):
    n = data.n
    y = data.y
    z = data.Xm
    x = data.Xr
    c = parameters.c
    a = parameters.a
    Alpha = parameters.Alpha
    b = parameters.b
    Beta = parameters.Beta

    # construct outer product for sample covariate vector
    xxlist = []
    for i in xrange(n):
        xi = x[i, :]  # 1d array
        if b is not None:
            xi = np.append(1., xi)
        xxi = np.outer(xi, xi)[np.newaxis, :, :]
        xxlist.append(xxi)
    xx = np.vstack(xxlist)  # 3d array

    B_ = B(data, parameters)  # 1d array

    if j == k:
        piej_ = pie(z, Alpha, a, j)  # 1d array

        bj_ = None if b is None else b[j-1]
        pj_ = pj(x, Beta[:, j-1], bj_)  # 1d array

        # part 1: with second derivative
        w_pj_hessjj = pj_ * (1 - pj_)**2 - pj_**2 * (1 - pj_)
        pj_hessjj = w_pj_hessjj[:, np.newaxis, np.newaxis] * xx
        w1 = (2. * y - 1.) * piej_ / B_
        part1_ = w1[:, np.newaxis, np.newaxis] * pj_hessjj

        # part 2: with product of two first derivatives
        w2 = (piej_ * pj_ * (1 - pj_) / B_)**2
        part2_ = w2[:, np.newaxis, np.newaxis] * xx

        hess_betajj_ = part1_ + part2_
        if returnElementwise:
            return hess_betajj_
        else:
            return hess_betajj_.sum(0)
    else:
        piej_ = pie(z, Alpha, a, j)  # 1d array
        piek_ = pie(z, Alpha, a, k)  # 1d array

        bj_ = None if b is None else b[j-1]
        pj_ = pj(x, Beta[:, j-1], bj_)  # 1d array

        bk_ = None if b is None else b[k-1]
        pk_ = pj(x, Beta[:, k-1], bk_)  # 1d array

        w = -piej_ * piek_ * pj_ * (1. - pj_) * pk_ * (1. - pk_) / B_**2

        hess_betajk_ = w[:, np.newaxis, np.newaxis] * xx
        if returnElementwise:
            return hess_betajk_
        else:
            return hess_betajk_.sum(0)


# simulator = MixedLogisticDataSimulator(100, 3, 3, 3)
# simulator.simulate(25)
# data = simulator.data
# parameters = simulator.parameters

# for j in [1,2,3]:
#     for k in [1,2,3]:
#         print 'j=', j, ', ', 'k=', k, ':'
#         print hess_betajk(data, parameters, j, k), '\n'

# hess_betajk(data, parameters, 1, 1)
# hess_betajk(data, parameters, 2, 2)  # diagonal positive
# hess_betajk(data, parameters, 3, 3)  # diagonal positive

def hess_alphaj_betak(data, parameters, j, k, returnElementwise=False):
    n = data.n
    y = data.y
    z = data.Xm
    x = data.Xr
    a = parameters.a
    Alpha = parameters.Alpha
    b = parameters.b
    Beta = parameters.Beta
    c = parameters.c
    pHiddenLayer = parameters.dxm if a is None else parameters.dxm + 1
    pObservedLayer = parameters.dxr if b is None else parameters.dxr + 1

    B_ = B(data, parameters)

    bc_ = None if b is None else b[c-1]
    pc_ = pj(x, Beta[:, c-1], bc_)

    if k == c:
        acc = np.zeros((n, pHiddenLayer, pObservedLayer))
        for l in np.arange(c-1) + 1:
            bl_ = None if b is None else b[l-1]
            pl_ = pj(x, Beta[:, l-1], bl_)
            Al_ = Aj(pl_, pc_, y)  # 1d array

            piec_ = pie(z, Alpha, a, c)  # 1d array

            w = piec_ * Al_ / B_**2 + 1. / B_  # 1d array

            piel_gradj_ = piej_gradk(z, Alpha, a, l, j, True)  # 2d array
            pc_grad_ = pj_grad(x, Beta[:, c-1], bc_)  # 2d array
            outerProductList_2d = [np.outer(piel_gradj_[i, :], pc_grad_[i, :]) for i in xrange(n)]
            outerProductList_3d = [elem[np.newaxis, :, :] for elem in outerProductList_2d]
            outerProduct = np.vstack(outerProductList_3d)  # 3d array

            acc += w[:, np.newaxis, np.newaxis] * outerProduct

        hess_ = (1. - 2.*y)[:, np.newaxis, np.newaxis] * acc
    else:
        piek_gradj_ = piej_gradk(z, Alpha, a, k, j, True)  # 2d array
        bk_ = None if b is None else b[k-1]
        pk_grad_ = pj_grad(x, Beta[:, k-1], bk_)  # 2d array
        outerProductList_2d = [np.outer(piek_gradj_[i, :], pk_grad_[i, :]) for i in xrange(n)]
        outerProductList_3d = [elem[np.newaxis, :, :] for elem in outerProductList_2d]
        outerProduct = np.vstack(outerProductList_3d)  # 3d array
        part1 = (1. / B_)[:, np.newaxis, np.newaxis] * outerProduct

        acc = np.zeros((n, pHiddenLayer, pObservedLayer))
        for l in np.arange(c-1) + 1:
            bl_ = None if b is None else b[l-1]
            pl_ = pj(x, Beta[:, l-1], bl_)
            Al_ = Aj(pl_, pc_, y)  # 1d array

            piek_ = pie(z, Alpha, a, k)  # 1d array

            w = piek_ * Al_ / B_**2  # 1d array

            piel_gradj_ = piej_gradk(z, Alpha, a, l, j, True)  # 2d array
            outerProductList_2d = [np.outer(piel_gradj_[i, :], pk_grad_[i, :]) for i in xrange(n)]
            outerProductList_3d = [elem[np.newaxis, :, :] for elem in outerProductList_2d]
            outerProduct = np.vstack(outerProductList_3d)  # 3d array
            acc += w[:, np.newaxis, np.newaxis] * outerProduct
        part2 = -acc

        hess_ = (2. * y - 1.)[:, np.newaxis, np.newaxis] * (part1 + part2)

    if returnElementwise:
        return hess_
    else:
        return hess_.sum(0)


# simulator = MixedLogisticDataSimulator(10, 3, 2, 3)
# simulator.simulate(25)
# data = simulator.data
# parameters = simulator.parameters
#
# for j in [1,2,3]:
#     for k in [1,2,3]:
#         print 'j=', j, ', ', 'k=', k, ':'
#         print hess_alphaj_betak(data, parameters, j, k), '\n'


######################################
# Evaluate Fisher Information Matrix #
######################################
import matplotlib.pyplot as plt

def empiricalFisherInformationFromSimulatedData(n):
    simulator = MixedLogisticDataSimulator(n, 3, 3, 3)
    simulator.simulate(25)
    data = simulator.data
    parameters = simulator.parameters

    c = parameters.c
    pHidden = parameters.pHiddenLayer  # hidden layer dimension
    pObserved = parameters.pObservedLayer  # observed layer dimension
    parameterDimension = parameters.numberOfParameters
    fisherInfomationMatrix = np.zeros((parameterDimension, parameterDimension))  # 2d array, initialize Fisher information

    # blocks for alpha
    for j in np.arange(c-1) + 1:
        for k in np.arange(c-1) + 1:
            fisherInfomationMatrix[(j-1)*pHidden : j*pHidden, (k-1)*pHidden : k*pHidden] = \
                hess_alphajk(data, parameters, j, k, False)

    # blocks for beta
    offset = (c-1) * pHidden
    for j in np.arange(c) + 1:
        for k in np.arange(c) + 1:
            fisherInfomationMatrix[offset+(j-1)*pObserved : offset+j*pObserved, offset+(k-1)*pObserved : offset+k*pObserved] = \
                hess_betajk(data, parameters, j, k, False)

    # cross blacks for alpha-beta
    for j in np.arange(c-1) + 1:
        for k in np.arange(c) + 1:
            fisherInfomationMatrix[(j-1)*pHidden : j*pHidden, offset+(k-1)*pObserved : offset+k*pObserved] = \
                hess_alphaj_betak(data, parameters, j, k, False)

    for j in np.arange(c) + 1:
        for k in np.arange(c-1) + 1:
            fisherInfomationMatrix[offset+(j-1)*pObserved : offset+j*pObserved, (k-1)*pHidden : k*pHidden] = \
                hess_alphaj_betak(data, parameters, j, k).T

    return -fisherInfomationMatrix


# fisherInfo = empiricalFisherInformationFromSimulatedData(5000)
# # fisherInfo
# np.linalg.det(fisherInfo)  # determinant of Fisher Information
#
#
# plt.imshow(fisherInfo, cmap='spectral')  # camp = 'hot', 'spectral'
# plt.colorbar()  # Color scale reference
#
# for i in range(20):
#     print '%.05f' % np.diag(fisherInfo)[i]

"""
The above Fisher information matrix has negative diagonal elements.
So I am not confident the derivation of the second order derivatives
are correct. Since a Fisher information matrix is also the variance-covariance
matrix of the parameters' scores, the diagonal elements should be
non-negative.

We change the approach of evaluating Fisher information matrix
by estimating the variance of the scores, i.e.

    I(\theta) = Var(scores) = E[scores * scores^T] - E[scores] * E[scores]^T

where E[scores * scores^T] and E[scores] are estimated by simulation.
"""

# def empiricalFisherInformationFromSimulatedData2(n, seed, dxm, dxr, c):
#     simulator = MixedLogisticDataSimulator(n, dxm, dxr, c)
#     simulator.simulate(seed)
#     data = simulator.data
#     parameters = simulator.parameters
#
#     c = parameters.c
#     pHidden = parameters.pHiddenLayer  # hidden layer dimension
#     pObserved = parameters.pObservedLayer  # observed layer dimension
#     parameterDimension = parameters.numberOfParameters
#
#     scores = np.zeros(parameterDimension)  # 1d array
#
#     for j in np.arange(c-1) + 1:
#         scores[(j-1)*pHidden : j*pHidden] = grad_alphaj(data, parameters, j, False)
#
#     offset = (c-1) * pHidden
#     for j in np.arange(c) + 1:
#         scores[offset+(j-1)*pObserved : offset+j*pObserved] = grad_betaj(data, parameters, j, False)
#
#     fisherInfo = np.outer(scores, scores)
#     print "** Done **"
#     return fisherInfo
#
# # fisherInfo = empiricalFisherInformationFromSimulatedData2(5000, 25)


def estimatedFisherInformationFromSimulation(startSeed, nSim, nSample, dxm, dxr, c):
    seeds = np.arange(nSim) + startSeed
    # assume intercepts are activated, otherwise
    # pHidden = dxm
    # pObserved = dxr
    pHidden = dxm + 1
    pObserved = dxr + 1
    nHiddenParameters = pHidden * (c-1)
    nObservedParameters = pObserved * c
    nParameters = nHiddenParameters + nObservedParameters

    firstMomnet = np.zeros(nParameters)
    secondMoment = np.zeros((nParameters, nParameters))

    # start simulation
    i = 0
    while i < nSim:
        simulator = MixedLogisticDataSimulator(nSample, dxm, dxr, c)
        simulator.simulate(seeds[i])
        data = simulator.data
        parameters = simulator.parameters

        # Evaluate gradient (i.e. score) for the current simulation
        scores = np.zeros(nParameters)  # 1d array
        ## grad_alpha
        for j in np.arange(c-1) + 1:
            scores[(j-1)*pHidden : j*pHidden] = grad_alphaj(data, parameters, j, False)
        ## grad_beta
        for j in np.arange(c) + 1:
            scores[nHiddenParameters+(j-1)*pObserved : nHiddenParameters+j*pObserved] = grad_betaj(data, parameters, j, False)

        firstMomnet += scores / nSim
        secondMoment += np.outer(scores, scores) / nSim

        i += 1
        if i % 10 == 0: print i

    fisherInformation = secondMoment - np.outer(firstMomnet, firstMomnet)
    return fisherInformation


# res = estimatedFisherInformationFromSimulation(25, 100, 5000, 3, 3, 3)
#
# plt.imshow(res)
# plt.colorbar()