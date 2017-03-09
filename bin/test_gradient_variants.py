import sys; sys.path.append('rrr')
import decoy_mnist
from multilayer_perceptron import *
import numpy as np
import pdb
import pickle

def dumlp(mlp, fname):
  fname = 'data/decoy_mnist_{}.pkl'.format(fname)
  pickle.dump(mlp.params, open(fname, 'wb'))

Xr, X, y, E, Xtr, Xt, yt, Et = decoy_mnist.generate_dataset()

mlp_linmax_1000 = MultilayerPerceptron(l2_grads=1000)
mlp_linmax_1000.fit(X, y, E, scale='linear', y='max')
dumlp(mlp_linmax_1000, 'mlp_linmax_1000')

mlp_linmax_100000 = MultilayerPerceptron(l2_grads=100000)
mlp_linmax_100000.fit(X, y, E, scale='linear', y='max')
dumlp(mlp_linmax_100000, 'mlp_linmax_100000')

mlp_logmax_1000 = MultilayerPerceptron(l2_grads=1000)
mlp_logmax_1000.fit(X, y, E, scale='log', y='max')
dumlp(mlp_logmax_1000, 'mlp_logmax_1000')

mlp_logmax_100000 = MultilayerPerceptron(l2_grads=100000)
mlp_logmax_100000.fit(X, y, E, scale='log', y='max')
dumlp(mlp_logmax_100000, 'mlp_logmax_100000')

mlp_linsum_1000 = MultilayerPerceptron(l2_grads=1000)
mlp_linsum_1000.fit(X, y, E, scale='linear', y='sum')
dumlp(mlp_linsum_1000, 'mlp_linsum_1000')

mlp_linsum_100000 = MultilayerPerceptron(l2_grads=100000)
mlp_linsum_100000.fit(X, y, E, scale='linear', y='sum')
dumlp(mlp_linsum_100000, 'mlp_linsum_100000')

mlp_logsum_1000 = MultilayerPerceptron(l2_grads=1000)
mlp_logsum_1000.fit(X, y, E, scale='log', y='sum')
dumlp(mlp_logsum_1000, 'mlp_logsum_1000')

mlp_logsum_100000 = MultilayerPerceptron(l2_grads=100000)
mlp_logsum_100000.fit(X, y, E, scale='log', y='sum')
dumlp(mlp_logsum_100000, 'mlp_logsum_100000')
