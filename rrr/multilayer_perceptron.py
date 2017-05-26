import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad, elementwise_grad
from autograd.util import flatten
from autograd.optimizers import adam
from local_linear_explanation import LocalLinearExplanation

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

def input_gradients(params, y=None, scale='log'):
  # log probabilities or probabilities
  if scale is 'log':
    p = lambda x: feed_forward(params, x)
  else:
    p = lambda x: np.exp(feed_forward(params, x))

  # max, sum, or individual y
  if y is None: y = 'sum' if scale is 'log' else 'max'
  if y is 'sum':
    p2 = p
  elif y is 'max':
    p2 = lambda x: np.max(p(x), axis=1)
  else:
    p2 = lambda x: p(x)[:, y]

  return elementwise_grad(p2)

def l2_irrelevant_input_gradients(params, X, A, **kwargs):
  return l2_norm(input_gradients(params, **kwargs)(X)[A])

def init_random_params(scale, layer_sizes, rs=npr):
  return [(scale * rs.randn(m, n),   # weight matrix
           scale * rs.randn(n))      # bias vector
          for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

class MultilayerPerceptron():
  @classmethod
  def from_params(klass, params):
    mlp = klass()
    mlp.params = params
    return mlp

  def __init__(self, layers=(50,30), l2_params=0.0001, l2_grads=0.0001):
    self.l2_params = l2_params
    self.l2_grads = l2_grads
    self.layers = list(layers)

  def predict_proba(self, inputs):
    return np.exp(feed_forward(self.params, inputs))

  def predict(self, inputs):
    return np.argmax(feed_forward(self.params, inputs), axis=1)

  def score(self, inputs, targets):
    return np.mean(self.predict(inputs) == targets)

  def input_gradients(self, X, **kwargs):
    if 'scale' not in kwargs:
      kwargs['scale'] = None # default to non-log probs
    return input_gradients(self.params, **kwargs)(X.astype(np.float32))

  def grad_explain(self, X, **kwargs):
    yhats = self.predict(X)
    coefs = self.input_gradients(X, **kwargs)
    return [LocalLinearExplanation(X[i], yhats[i], coefs[i]) for i in range(len(X))]

  def largest_gradient_mask(self, X, cutoff=0.67, **kwargs):
    grads = self.input_gradients(X, **kwargs)
    return np.array([np.abs(g) > cutoff*np.abs(g).max() for g in grads])

  def fit(self, inputs, targets, A=None, num_epochs=64, batch_size=256,
      step_size=0.001, rs=npr, nonlinearity=relu, verbose=False, normalize=False,
      always_include=None,
      **input_grad_kwargs):
    X = inputs.astype(np.float32)
    y = one_hot(targets)
    if A is None: A = np.zeros_like(X).astype(bool)
    params = init_random_params(0.1, [X.shape[1]] + self.layers + [y.shape[1]], rs=rs)

    if type(verbose) == int:
      v = verbose
      verbose = lambda x: x % v == 0

    batch_size = min(batch_size, X.shape[0])
    num_batches = int(np.ceil(X.shape[0] / batch_size))

    def batch_indices(iteration):
      idx = iteration % num_batches
      return slice(idx * batch_size, (idx+1) * batch_size)

    def objective(params, iteration):
      idx = batch_indices(iteration)
      Ai = A[idx]
      Xi = X[idx]
      yi = y[idx]

      if always_include is not None:
        Ai = np.vstack((A[always_include], Ai))
        Xi = np.vstack((X[always_include], Xi))
        yi = np.vstack((y[always_include], yi))

      if normalize:
        sumA = max(1., float(Ai.sum()))
        lenX = max(1., float(len(Xi)))
      else:
        sumA = 1.
        lenX = 1.

      crossentropy = -np.sum(feed_forward(params, Xi, nonlinearity) * yi) / lenX
      rightreasons = self.l2_grads * l2_norm(input_gradients(params, **input_grad_kwargs)(Xi)[Ai]) / sumA
      smallparams = self.l2_params * l2_norm(params)

      if verbose and verbose(iteration):
        print('Iteration={}, crossentropy={}, rightreasons={}, smallparams={}, sumA={}, lenX={}'.format(
          iteration, crossentropy.value, rightreasons.value, smallparams.value, sumA, lenX))

      return crossentropy + rightreasons + smallparams

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
