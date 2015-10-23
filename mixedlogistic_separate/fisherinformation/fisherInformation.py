import numpy as np
from mixedlogistic_separate.mixedLogistic_rawScript import sigmoid

"""
c = 3
dxm = 0  # constant hidden membership probabilities
X ~ Unif(-2, 2)
"""
dxr = 1

# Hidden membership probabilities
pi = np.array([1./2, 1./3, 1./6])

# Beta = np.array([[1., 5, 10], [5, 10, 1], [10, 5, 1]]).T
Beta = np.array([[-5., 0., 5.]])


def simulateX(dxr):
    return np.random.uniform(-2., 2., (dxr,))

def simulateEmpiricalFisher():
    # Hidden memberships
    u = np.random.random()
    member = (u > pi.cumsum()).sum()  # group member, start from 0

    x = simulateX(dxr)
    choosingProbs = sigmoid(x.dot(Beta))  # for all hidden groups
    myChoosingProb = choosingProbs[member]  # for my current group

    u2 = np.random.random()
    y = int(u2 < myChoosingProb)

    if y == 0:
        B = (pi * (1 - choosingProbs)).sum()
    else:
        B = (pi * choosingProbs).sum()

    # Calculate second derivative
    x = x.reshape(1, dxr)
    b = pi * choosingProbs * (1 - choosingProbs)
    hess = []
    for j in range(3):
        pj = choosingProbs[j]
        s = (2 * y - 1) * (1 - 2 * pj) * b[j] / B - (b[j] / B)**2
        hess.append(s * x.T.dot(x))

    empiricalFisher = [-m for m in hess]  # for one sample
    return np.array([float(ef) for ef in empiricalFisher])

# det = [np.linalg.det(f) for f in empiricalFisher]  # determinants

res = []
for i in range(100000):
    res.append(simulateEmpiricalFisher())

# Average empirical Fisher information as an estimate
# of the population Fisher information
np.vstack(res).mean(0)


# Plot simulated empirical Fisher information
import matplotlib.pyplot as plt

fig = plt.figure()
# plt.boxplot(np.vstack(res)[:,0])
plt.hist(np.vstack(res)[:,0], 100)
plt.show()
