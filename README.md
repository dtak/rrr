# Right for the Right Reasons

This repository contains Python modules, scripts, and notebooks used to generate the figures in [Right for the Right Reasons](https://www.ijcai.org/proceedings/2017/371).

## Main Idea

Sometimes machine learning models generalize poorly even though they seem accurate on validation datasets, often because our data contains confounding factors we haven't considered. To detect these problems, we want to generate explanations of machine learning models that help us understand whether they are making decisions for the right reasons. Tools like [LIME](https://github.com/marcotcr/lime) can explain any model by presenting local linear approximations of the model's decision boundary, but that approach is computationally intensive and can't fix the problems it reveals. Input gradients actually _are_ local linear approximations of the decision boundary of differentiable models, and furthermore, we can constrain their values in our loss function, which lets us prevent them from learning implicit rules that we know to be wrong. Altogether, we end up with a computationally efficient method of both explaining models and constraining them to be right for the right reasons (or at least slightly better ones). See the [paper](https://www.ijcai.org/proceedings/2017/371) or the experiments below for more details.

## Repository Structure

- `experiments/` contains iPython notebooks used to generate figures. In particular,
    - [2D Intuition](./experiments/2D%20Intuition.ipynb) provides some visualizations illustrating how to think about input gradients.
    - [Toy Colors](./experiments/Toy%20Colors.ipynb) demonstrates explaining and gradually constraining explanations on a toy image dataset, where we have perfect foreknowledge about what the right explanation should be.
    - [Iris-Cancer](./experiments/Iris-Cancer.ipynb) applies the same techniques to a very tiny mashup of Iris and Wisconsin Breast Cancer, where Iris serves as a misleading decoy rule.
    - [20 Newsgroups](./experiments/20%20Newsgroups.ipynb) tests input gradients' ability to explain text datasets, with comparisons to LIME text.
    - [Decoy MNIST](./experiments/Decoy%20MNIST.ipynb) is a larger example where we introduce a non-generalizable rule into the MNIST training data which hurts test accuracy if we don't provide explanations.
    - [Loss Functions](./experiments/Loss%20Functions.ipynb) does some very basic exploration of different explanation regularization formulations for our loss function in the context of Decoy MNIST. There are a lot of potential variants of this method and we only really scratch the surface.
- `bin/` contains scripts used to train many iterations of models when it was inconvenient to do so from within a notebook
- `data/` is where we cached raw data and saved model parameters (some of which is `.gitignore`d, but will be regenerated automatically when you clone)
- `rrr/` contains the code for representing explanations, explainable models, and datasets. [multilayer_perceptron.py](./rrr/multilayer_perceptron.py) defines the model we used repeatedly, which has an sklearn-style `fit(X, y)` interface (accepting an optional annotation mask `A`) and is implemented using [autograd](https://github.com/HIPS/autograd).

If you have `numpy`, `scikit-learn`, `autograd`, and `lime` installed, you should be able to run all of the notebooks and scripts after cloning this repository.

## Citation

```
@inproceedings{ijcai2017-371,
  author    = {Andrew Slavin Ross, Michael C. Hughes, Finale Doshi-Velez},
  title     = {Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations},
  booktitle = {Proceedings of the Twenty-Sixth International Joint Conference on
               Artificial Intelligence, {IJCAI-17}},
  pages     = {2662--2670},
  year      = {2017},
  doi       = {10.24963/ijcai.2017/371},
  url       = {https://doi.org/10.24963/ijcai.2017/371},
}
```
