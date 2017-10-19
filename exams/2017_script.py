import numpy as np

x = np.array([[ 1.,  0.,  0.,  1.,  0.,  0.],
       [ 1.,  1.,  0.,  1.,  1.,  0.],
       [ 1.,  1.,  1.,  0.,  0.,  1.],
       [ 1.,  1.,  0.,  0.,  0.,  0.],
       [ 1.,  1.,  1.,  0.,  1.,  1.],
       [ 1.,  0.,  1.,  0.,  0.,  1.],
       [ 1.,  0.,  0.,  1.,  1.,  0.],
       [ 1.,  0.,  1.,  1.,  1.,  1.]])

it = 0
w = np.zeros(6)
while it < 100:
    for i,xi in enumerate(x):
        test = np.sign(w.T.dot(xi))
        if i <= 3 and test <= 0:
            w += xi
        if i >= 4 and test > 0:
            w -= xi
        print(w)
    it += 1
