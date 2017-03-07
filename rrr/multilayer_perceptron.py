import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad, elementwise_grad
from autograd.util import flatten
from autograd.optimizers import adam
from explainable_model import ExplainableModel

# Adapted from https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
# with modifications made such that we have a first-class MLP object
# and such that our loss function includes an explanation penalty.

def one_hot(y):
  if len(y.shape) != 1:
    return y
  values = np.array(sorted(list(set(y))))
  return np.array([values == v for v in y], dtype=np.uint8)

def relu(inputs):
  return np.maximum(inputs, 0.)

def l2_norm(params):
  flattened, _ = flatten(params)
  return np.dot(flattened, flattened)

def feed_forward(params, inputs, nonlinearity=relu):
  for W, b in params:
    outputs = np.dot(inputs, W) + b
    inputs = nonlinearity(outputs)
  return outputs - logsumexp(outputs, axis=1, keepdims=True) # outputs log probabilities

def l2_irrelevant_input_gradients(params, inputs, irrelevancies):
  predict_fn = lambda inputs: feed_forward(params, inputs) # swap arg order
  predict_grads = elementwise_grad(predict_fn)(inputs)
  return l2_norm(predict_grads[irrelevancies])

def init_random_params(scale, layer_sizes, rs=npr):
  return [(scale * rs.randn(m, n),   # weight matrix
           scale * rs.randn(n))      # bias vector
          for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

class MultilayerPerceptron(ExplainableModel):
  @classmethod
  def from_params(klass, params):
    mlp = klass()
    mlp.params = params
    return mlp

  def __init__(self, layer_sizes=(50,30), l2_params=0.0001, l2_grads=0.0001):
    self.l2_params = l2_params
    self.l2_grads = l2_grads
    self.layer_sizes = list(layer_sizes)

  def predict_proba(self, inputs):
    return np.exp(feed_forward(self.params, inputs))

  def predict(self, inputs):
    return np.argmax(feed_forward(self.params, inputs), axis=1)

  def score(self, inputs, targets):
    return np.mean(self.predict(inputs) == targets)

  def input_gradients(self, inputs, y=None):
    if y is None:
      predict_fn = lambda inputs: np.max(self.predict_proba(inputs), axis=1)
    else:
      predict_fn = lambda inputs: self.predict_proba(inputs)[:, y]
    return elementwise_grad(predict_fn)(inputs.astype(np.float32))

  def fit(self, inputs, targets, irrelevancies=None, num_epochs=64,
      batch_size=256, step_size=0.001, random_state=npr,
      nonlinearity=relu):

    X = inputs.astype(np.float32)
    y = one_hot(targets)
    layer_sizes = [X.shape[1]] + self.layer_sizes + [y.shape[1]]
    params = init_random_params(0.1, layer_sizes, rs=random_state)

    if irrelevancies is None:
      irrelevancies = np.zeros_like(X).astype(bool)

    num_batches = int(np.ceil(X.shape[0] / batch_size))

    def batch_indices(iteration):
      idx = iteration % num_batches
      return slice(idx * batch_size, (idx+1) * batch_size)

    def objective(params, iteration):
      idx = batch_indices(iteration)
      return -(
        np.sum(feed_forward(params, X[idx], nonlinearity) * y[idx]) # cross-entropy
        - self.l2_params * l2_norm(params) # L2 regularization on parameters directly
        - self.l2_grads * l2_irrelevant_input_gradients( # "Explanation regularization"
          params, X[idx], irrelevancies[idx]))

    self.params = adam(grad(objective), params, step_size=step_size, num_iters=num_epochs*num_batches)

if __name__ == '__main__':
  import toy_colors
  print('generating dataset...')
  X, Xt, y, yt = toy_colors.generate_dataset()
  print('fitting MLP')
  mlp = MultilayerPerceptron()
  mlp.fit(X, y)
  print('train:', mlp.score(X, y))
  print('test:', mlp.score(Xt, yt))
