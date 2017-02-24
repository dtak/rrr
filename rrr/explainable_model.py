from local_linear_explanation import LocalLinearExplanation
import lime
import lime.lime_tabular
import lime.lime_text
import numpy as np

class ExplainableModel():
  def predict(self, X):
    raise NotImplementedError('implement me')

  def predict_proba(self, X):
    raise NotImplementedError('implement me')

  def input_gradients(self, X):
    raise NotImplementedError('implement me if using grad_explain')

  def largest_gradient_mask(self, X, cutoff=0.67):
    grads = self.input_gradients(X)
    return np.array([np.abs(g) > cutoff*np.abs(g).max() for g in grads])

  # create wrapper objects around the raw coefficients outputted by the gradients,
  # for easy plotting / manipulation
  def grad_explain(self, X):
    y = self.predict(X)
    coefs = self.input_gradients(X)
    return [LocalLinearExplanation(X[i], y[i], coefs[i]) for i in range(len(X))]

  # coerce LIME explanations into the same form for easier side-by-side comparisons
  def lime_explain(self, X, explainer=None, explainer_type=None, predict_proba=None, num_features=None):
    y = self.predict(X)

    if explainer is None:
      if explainer_type is None:
        explainer_type = lime.lime_tabular.LimeTabularExplainer
      xnames = list(range(len(X[0])))
      ynames = np.unique(y)
      explainer = explainer_type(X, feature_names=xnames, class_names=ynames)

    if predict_proba is None:
      predict_proba = self.predict_proba

    if num_features is None:
      num_features = int(np.sqrt(len(X[0])))

    coefs = np.zeros(X.shape)
    for i in range(len(X)):
      limexp = explainer.explain_instance(
          X[i], predict_proba, labels=[y[i]], num_features=num_features)
      for feature, coef in limexp.local_exp[y[i]]:
        coefs[i][feature] = coef
    return [LocalLinearExplanation(X[i], y[i], coefs[i]) for i in range(len(X))]

  def lime_text_explain(self, X, vectorizer, **kwargs):
    y = self.predict(X)
    ex = lime.lime_text.LimeTextExplainer(class_names=np.unique(y))
    pp = lambda t: self.predict_proba(vectorizer.transform(t).toarray())
    return self.lime_explain(X, explainer=ex, predict_proba=pp, **kwargs)
