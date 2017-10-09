import cvxopt

from cvxopt.base import matrix
from cvxopt.solvers import qp

import matplotlib.pyplot as plt
import numpy as np
import pylab
import random, math

# Config
random.seed(999)
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['maxiters'] = 100
ALPHA_THRESH=1e-5

# Parameters
NUM = 10
START_A = (-1.5, 0.5)
START_B = (0.0, -0.5)

def genData(num):
    data = []
    # classA
    data += [(random.normalvariate(-1.5, 0.5), random.normalvariate(0.5, 0.5), 1.0) for _ in range(num//2)]
    # classB
    data += [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5) , -1.0) for _ in range(num//2)]
    random.shuffle(data)
    return data

def kernelLinear(x, y):
    return np.dot(x, y) + 1

def build_P(data, kernel):
    N = len(data)
    P = np.ndarray(shape=(N, N))
    x = [p[:-1] for p in data]
    t = [p[-1] for p in data]
    for i in range(N):
        for j in range(N):
            P[i][j] = t[i]*t[j]*kernel(x[i], x[j])
    return P

def build_q(data):
    return -1*np.ones(len(data))

def build_G(data):
    return -1*np.eye(len(data))

def build_h(data, c=0):
    return -c*np.ones(len(data))

def indicator(xnew, alpha, data, kernel):
    N = len(alpha)
    t = [p[-1] for p in data]
    x = [p[:-1] for p in data]
    ind = sum(alpha[i]*t[i]*kernel(xnew, x[i]) for i in alpha.keys())
    return ind

def build_alpha(r, threshhold):
    raw_alpha = list(r['x'])
    alpha = {}
    for i in range(len(raw_alpha)):
        if raw_alpha[i] > ALPHA_THRESH:
            alpha[i] = raw_alpha[i]
    return alpha

def plotBoundary(alpha, data, kernel):
    xrange = np.arange(-4, 4, 0.05)
    yrange = np.arange(-4, 4, 0.05)
    inds = [[indicator((x, y), alpha, data, kernel) for x in xrange] for y in yrange]
    plt.contour(xrange, yrange, inds, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

def plotPoints(data, alpha):
    dataA = list(filter(lambda p: p[-1] == +1.0, data))
    dataB = list(filter(lambda p: p[-1] == -1.0, data))
    suppv = [s for i,s in enumerate(data) if i in alpha.keys()]
    plt.title('QP Fail')
    plt.plot([p[0] for p in dataA], [p[1] for p in dataA], 'bo')
    plt.plot([p[0] for p in dataB], [p[1] for p in dataB], 'ro')
    plt.plot([p[0] for p in suppv], [p[1] for p in suppv], 'wx')

num = 10
data = genData(num)
P, q, G, h = build_P(data, kernelLinear), build_q(data), build_G(data), build_h(data)
r = qp(P=matrix(P), q=matrix(q), G=matrix(G), h=matrix(h))
alpha = build_alpha(r, threshhold=ALPHA_THRESH)

plotPoints(data, alpha)
plotBoundary(alpha, data, kernelLinear)
plt.show()
