import sys; sys.path.append('rrr')
from toy_colors import *
from multilayer_perceptron import *
import numpy as np
import pdb
import pickle

X, Xt, y, yt = generate_dataset()
A1 = np.array([ignore_rule2 for _ in range(len(y))])
A2 = np.array([ignore_rule1 for _ in range(len(y))])

n_vals = np.array(sorted(list(set(np.logspace(0, 4, 50).astype(int)))))
n_mlps = []
for n in n_vals:
  print(n)
  A = np.array([ignore_rule1 for _ in range(n)] + [np.zeros_like(ignore_rule1).astype(bool) for _ in range(len(y)-n)])
  mlp = MultilayerPerceptron(l2_grads=1000)
  mlp.fit(X, y, A)
  n_mlps.append(mlp.params)

pickle.dump(n_vals, open('data/color_n_vals.pkl', 'wb'))
pickle.dump(n_mlps, open('data/color_n_mlps.pkl', 'wb'))

l2_vals = np.logspace(-3, 3, 50)
l2_mlps = []
for l2 in l2_vals:
  print(l2)
  mlp = MultilayerPerceptron(l2_grads=l2)
  mlp.fit(X, y, A2)
  l2_mlps.append(mlp.params)

pickle.dump(l2_vals, open('data/color_l2_vals.pkl', 'wb'))
pickle.dump(l2_mlps, open('data/color_l2_mlps.pkl', 'wb'))
