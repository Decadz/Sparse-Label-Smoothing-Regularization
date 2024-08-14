<h1 align="center">
Sparse Label Smoothing Regularization
</h1>

This repository contains PyTorch code for the Sparse Label Smoothing Regularization (SparseLSR) loss function proposed in the paper "[Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning](https://arxiv.org/abs/2209.08907)" by Christian Raymond, Qi Chen, Bing Xue, and Mengjie Zhang. The SparseLSR loss function is a significantly faster and more memory-efficient way to compute Label Smoothing Regularization (LSR).

<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/sparse-lsr.png" width="750"/>
</p>

## Contents

A PyTorch implementation of the proposed Sparse Label Smoothing Regularization (SparseLSR) loss function. This repository contains the following useful scripts:

- ```loss_functions.py``` - PyTorch code containing an implementation of SparseLSR and conventional LSR.
- ```visualizations.py``` - Script for visualizing the different classification loss functions.
- ```train.py``` - Code for testing the different loss functions and visualizing the penultimate layer representations.

## Installation

1. Clone this repository to your local machine:
```bash
git clone https://github.com/Decadz/Sparse-Label-Smoothing-Regularization.git
cd Sparse-Label-Smoothing-Regularization
```

2. Install the necessary libraries and dependencies:
```bash
pip install requirements.txt
```

## Sparse Label Smoothing Regularization

The key idea behind sparse label smoothing regularization is to utilize the redistributed loss trick, which takes the expected non-target loss and redistributes it into the target loss, obviating the need to calculate the loss on the non-target outputs. The redistributed loss trick can retain near identical behavior due to the output softmax function redistributing the gradients back into the non-target outputs during backpropagation. The sparse label smoothing regularization loss is defined as follows:

<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-1.png"/>
</p>
<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-2.png"/>
</p>

where the expectation of the model's non-target output $`\mathbb{E}[\log(f_{\theta}(x)_{j})]`$ is approximated via a first-order Taylor-expansion, *i.e.*, a linear approximation, which lets us rewrite the expectation in terms of $`f_{\theta}(x)_j`$.

<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-3.png"/>
</p>
<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-4.png"/>
</p>
<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-5.png"/>
</p>
<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-6.png"/>
</p>

By definition of the softmax activation function the summation of the model's output predictions is $`\sum_{i = 1}^{\mathcal{C}}f_{\theta}(x)_i = 1`$; therefore, the expected value of the non-target output predictions $`\mathbb{E}[f_{\theta}(x)_j]`$ where $`y_j = 0`$ can be given as $`1-f_{\theta}(x)_i`$ where $`y_i = 1`$ normalized over the number of non-target outputs $\mathcal{C}-1$. Substituting this result back into our expression gives the following:

<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-7.png"/>
</p>
<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-8.png"/>
</p>

where the first conditional summation can be removed to make explicit that $`\mathcal{L}_{SparseLSR}`$ is only non-zero for the target output, *i.e.*, where $`y_i = 1`$, and the second conditional summation can be removed to obviate recomputation of the non-target segment of the loss which is currently defined as the summation of a constant. The final definition of Sparse Label Smoothing Regularization Loss ($`\mathcal{L}_{SparseLSR}`$) is:

<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-9.png"/>
</p>

### Numerical Stability

The sparse label smoothing regularization loss is prone to numerical stability issues, analogous to the cross-entropy loss, when computing logarithms and exponentials (exponentials are taken in the softmax when converting logits into probabilities) causing under and overflow. In particular, the following expressions are prone to causing numerical stability issues:

<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-10.png"/>
</p>

In order to attain numerical stability when computing $`log(f_{\theta}(x)_i)`$ the well known *log-sum-exp trick* is employed to stably convert the pre-activation logit $`z_i`$ into a log probability which we further denote as $`\widetilde{f_{\theta}}(x)_i`$:

<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-11.png"/>
</p>
<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-12.png"/>
</p>
<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-13.png"/>
</p>
<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-14.png"/>
</p>

Regarding the remaining numerically unstable term, this can also be computed stably via the log-sum-exp trick; however, it would require performing the log-sum-exp operation an additional time, which would negate the time and space complexity savings over the non-sparse implementation of label smoothing regularization. Therefore, we propose to instead simply take the exponential of the target log probability to recover the raw probability and then add a small constant $`\epsilon=1e-7`$ to avoid the undefined $`\log(0)`$ case. The numerically stable sparse label smoothing loss is defined as follows:

<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/equation-15.png"/>
</p>

### PyTorch Code

```python
class SparseLSRLoss(torch.nn.Module):

    def __init__(self, smoothing=0.0, reduction="mean"):
        super(SparseLSRLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, y_pred, y_target):

        # Retrieving the total number of classes.
        num_classes = torch.tensor(y_pred.size(1))

        # Computing the log probabilities using numerically stable log-sum-exp.
        log_prob = torch.nn.functional.log_softmax(y_pred, dim=1)

        # Extracting the target indexes from the log probabilities.
        log_prob = torch.gather(log_prob, 1, y_target.unsqueeze(1))

        # Calculating the fast label smoothing regularization loss.
        loss = - (1 - self.smoothing + (self.smoothing / num_classes)) * log_prob + \
               ((self.smoothing * (num_classes - 1)) / num_classes) * \
               torch.log((torch.clamp(1 - torch.exp(log_prob), min=1e-7))/(num_classes - 1))

        # Applying the reduction and returning.
        return loss.mean() if self.reduction == "mean" else loss
```

### Visualizing Penultimate Layer Representation

The ```train.py``` script allows you to recreate the penultimate layer representation visualizations from the paper's appendix. In this script, AlexNet is trained on the CIFAR-10 dataset using the cross-entropy loss, label smoothing regularization, and sparse label smoothing regularization. After training, the penultimate layer representations on the testing set are visualized using [t-distributed Stochastic Neighbor Embedding](https://jmlr.org/papers/v9/vandermaaten08a.html) (t-SNE).

<p align="center">
  <img src="https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/blob/main/images/penultimate-layer-representations.png"/>
</p>

### Code Reproducibility

The code has not been comprehensively checked and re-run since refactoring. If you're having any issues, find a problem/bug or cannot reproduce similar results as the paper please [open an issue](https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/issues) or email me.

## References

If you use our library or find our research of value please consider citing our papers with the following Bibtex entry:

```
@article{raymond2023learning,
  title={Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning},
  author={Raymond, Christian and Chen, Qi and Xue, Bing},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
@article{raymond2024thesis,
  title={Meta-Learning Loss Functions for Deep Neural Networks},
  author={Raymond, Christian},
  journal={arXiv preprint arXiv:2406.09713},
  year={2024}
}
```
