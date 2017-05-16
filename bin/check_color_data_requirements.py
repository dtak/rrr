import sys; sys.path.append('rrr')
from toy_colors import *
from multilayer_perceptron import *
import numpy as np
import pdb
import pickle

X, Xt, y, yt = generate_dataset()
A_anti_rule1 = np.array([ignore_rule1 for _ in range(len(y))])
A_anti_rule2 = np.array([ignore_rule2 for _ in range(len(y))])
A_pro_rule1 = ~A_anti_rule1
A_pro_rule2 = ~A_anti_rule2
verbits = set(np.logspace(0, 5, 10).astype(int))

data_counts = np.logspace(1, 4, 32).astype(int)
pickle.dump(data_counts, open('data/color_data_counts.pkl', 'wb'))

normals_by_count = []
anti_r1s_by_count = []
anti_r2s_by_count = []
pro_r1s_by_count = []
pro_r2s_by_count = []

for l in data_counts:
  num_epochs = int((10000 / float(l)) * 32)
  kwargs = { 'num_epochs': num_epochs, 'batch_size': min(256, l) }
  print(l, num_epochs)

  def params_for(A):
    if A is None:
      A = np.zeros_like(X).astype(bool)
    mlp = MultilayerPerceptron(l2_grads=10000)
    mlp.fit(X[:l], y[:l], A[:l], normalize=True, verbose=lambda i: i in verbits, **kwargs)
    print(mlp.score(Xt,yt))
    return mlp.params

  normals_by_count.append(params_for(None))
  pro_r1s_by_count.append(params_for(A_pro_rule1))
  pro_r2s_by_count.append(params_for(A_pro_rule2))
  anti_r1s_by_count.append(params_for(A_anti_rule1))
  anti_r2s_by_count.append(params_for(A_anti_rule2))

pickle.dump(normals_by_count, open('data/color_normals_by_count.pkl', 'wb'))
pickle.dump(pro_r1s_by_count, open('data/color_pro_r1s_by_count.pkl', 'wb'))
pickle.dump(pro_r2s_by_count, open('data/color_pro_r2s_by_count.pkl', 'wb'))
pickle.dump(anti_r1s_by_count, open('data/color_anti_r1s_by_count.pkl', 'wb'))
pickle.dump(anti_r2s_by_count, open('data/color_anti_r2s_by_count.pkl', 'wb'))
