import matplotlib.pyplot as plt
import numpy as np

def listwrap(l):
  try:
    return list(l)
  except TypeError:
    return [l]

class LocalLinearExplanation():
  def __init__(self, x, y, coefs):
    assert(coefs.shape == x.shape)
    self.x = x
    self.y = y
    self.coefs = coefs

  def imshow(self, imgshape, **kwargs):
    if len(imgshape) == 2:
      self.imshow_bw(imgshape, **kwargs)
    else:
      self.imshow_rgb(imgshape, **kwargs)

  def imshow_rgb(self, imgshape, cutoff=0.67, xoff=0, yoff=0, color='white'):
    # display the image
    l, w, d = imgshape
    plt.imshow(self.x.reshape(imgshape), interpolation='none',
        extent=[w*xoff, w*(xoff+1), l*yoff, l*(yoff+1)])

    # display the coefficients, if they're big enough
    coefs = self.coefs.reshape(imgshape)
    maxcoef = np.abs(self.coefs).max()
    for y in range(l):
      for x in range(w):
        for i, coef in enumerate(listwrap(coefs[y][x])):
          if abs(coef) > maxcoef * cutoff:
            xx = w*xoff + x + 0.5
            yy = l*yoff + y + 0.5
            ymid = l*(yoff+0.5)
            yy = yy + -2*(yy-ymid)
            plt.scatter(xx, yy, c=color)

  def imshow_bw(self, imgshape, cutoff=0.0, xoff=0, yoff=0, size=75, lw=2, pwr=2):
    # display the image
    l, w = imgshape
    plt.imshow(self.x.reshape(imgshape), interpolation='none', cmap='gray',
        extent=[w*xoff, w*(xoff+1), l*yoff, l*(yoff+1)])

    # display the coefficients, if they're big enough
    coefs = self.coefs.reshape(imgshape)
    maxcoef = np.abs(self.coefs).max()
    markers = [None,'+','_']
    colors = [None, 'g', 'r']
    cutoff = 0.1
    for y in range(l):
      for x in range(w):
        for i, coef in enumerate(listwrap(coefs[y][x])):
          if abs(coef) > maxcoef * cutoff:
            xx = w*xoff + x + 0.5
            yy = l*yoff + y + 0.5
            ymid = l*(yoff+0.5)
            yy = yy + -2*(yy-ymid)
            plt.scatter(xx, yy, s=size, lw=lw,
                alpha=(abs(coef)/maxcoef)**pwr, # make opacity relative to magnitude
                c=colors[int(np.sign(coef))], # make color
                marker=markers[int(np.sign(coef))]) # and marker depend on sign

def explanation_grid(explanations, imgshape, length, pad=0.1, **kwargs):
  assert(len(explanations) >= length*length)

  if len(imgshape) == 2:
    l,l2 = imgshape
    assert(l == l2)
  else:
    l,l2,d = imgshape
    assert(l == l2)
    assert(d == 3)

  plt.axis('off')
  plt.xlim(0, l*(length*(1+pad)))
  plt.ylim(0, l*(length*(1+pad)))
  n = 0
  for i in range(length):
    for j in range(length):
      explanations[n].imshow(imgshape, xoff=i*(1+pad), yoff=(length-j-1)*(1+pad), **kwargs)
      n += 1

def image_grid(images, imgshape, length, **kwargs):
  fauxpls = [LocalLinearExplanation(images[i], 0, np.zeros_like(images[i])) for i in range(len(images))]
  explanation_grid(fauxpls, imgshape, length, **kwargs)
