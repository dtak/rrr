import sys; sys.path.append('rrr')
import decoy_mnist
from multilayer_perceptron import *
import numpy as np
import pdb
import pickle

Xr, X, y, E, Xtr, Xt, yt, Et = decoy_mnist.generate_dataset()
val = len(X) // 15
print(val)

verbits = set(np.logspace(0, 5, 10).astype(int))
powers = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
params = []
accurs = []
l2s = [10**p for p in powers]
for l2 in l2s:
  print('############')
  print('l2 =', l2)
  mlp = MultilayerPerceptron(l2_grads=l2)
  mlp.fit(X[val:], y[val:], E[val:], verbose=lambda i: i in verbits)
  accurs.append([
    mlp.score(X[val:], y[val:]),
    mlp.score(X[:val], y[:val]),
    mlp.score(Xt, yt)])
  print('accuracy =', accurs[-1])
  params.append(mlp.params)

filename = 'data/decoy_mnist_crossval.pkl'
pickle.dump((l2s, accurs, params), open(filename, 'wb'))
