from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()
import os
import gzip
import struct
import array
import numpy as np
from urllib.request import urlretrieve

def download_mnist(datadir):
  if not os.path.exists(datadir):
    os.makedirs(datadir)

  base_url = 'http://yann.lecun.com/exdb/mnist/'

  def parse_labels(filename):
    with gzip.open(filename, 'rb') as fh:
      magic, num_data = struct.unpack(">II", fh.read(8))
      return np.array(array.array("B", fh.read()), dtype=np.uint8)

  def parse_images(filename):
    with gzip.open(filename, 'rb') as fh:
      magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
      return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

  for filename in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
    if not os.path.exists(os.path.join(datadir, filename)):
      urlretrieve(base_url + filename, os.path.join(datadir, filename))

  train_images = parse_images(os.path.join(datadir, 'train-images-idx3-ubyte.gz'))
  train_labels = parse_labels(os.path.join(datadir, 'train-labels-idx1-ubyte.gz'))
  test_images  = parse_images(os.path.join(datadir, 't10k-images-idx3-ubyte.gz'))
  test_labels  = parse_labels(os.path.join(datadir, 't10k-labels-idx1-ubyte.gz'))

  return train_images, train_labels, test_images, test_labels

def Bern(p):
  return np.random.rand() < p

def augment(image, digit, randomize=False, mult=25, all_digits=range(10)):
  if randomize:
    return augment(image, np.random.choice(all_digits))

  img = image.copy()
  expl = np.zeros_like(img)

  fwd = [0,1,2,3]
  rev = [-1,-2,-3,-4]
  dir1 = fwd if Bern(0.5) else rev
  dir2 = fwd if Bern(0.5) else rev
  for i in dir1:
    for j in dir2:
      img[i][j] = 255 - mult*digit
      expl[i][j] = 1

  return img.ravel(), expl.astype(bool).ravel()

def _generate_dataset(datadir):
  X_raw, y, Xt_raw, yt = download_mnist(datadir)
  all_digits = list(set(y))

  X = []
  E = []
  Xt = []
  Et = []

  for image, digit in zip(X_raw, y):
    x, e = augment(image, digit, all_digits=all_digits)
    X.append(x)
    E.append(e)

  for image, digit in zip(Xt_raw, yt):
    x, e = augment(image, digit, all_digits=all_digits, randomize=True)
    Xt.append(x)
    Et.append(e)

  X = np.array(X)
  E = np.array(E)
  Xt = np.array(Xt)
  Et = np.array(Et)
  Xr = np.array([x.ravel() for x in X_raw])
  Xtr = np.array([x.ravel() for x in Xt_raw])

  return Xr, X, y, E, Xtr, Xt, yt, Et

def generate_dataset(cachefile='data/decoy-mnist.npz'):
  if cachefile and os.path.exists(cachefile):
    cache = np.load(cachefile)
    data = tuple([cache[f] for f in sorted(cache.files)])
  else:
    data = _generate_dataset(os.path.dirname(cachefile))
    if cachefile:
      np.savez(cachefile, *data)
  return data

if __name__ == '__main__':
  import pdb
  Xr, X, y, E, Xtr, Xt, yt, Et = generate_dataset()
  pdb.set_trace()
  pass
