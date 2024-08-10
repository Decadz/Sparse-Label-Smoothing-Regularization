<h1 align="center">
Sparse Label Smoothing Regularization
</h1>

This repository contains code for the Sparse Label Smoothing Regularization (SparseLSR) loss function proposed in the paper "[Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning](https://arxiv.org/abs/2209.08907)" by Christian Raymond, Qi Chen, Bing Xue, and Mengjie Zhang.

![banner-image](https://github.com/user-attachments/assets/aaf17557-0aa5-4232-a9e2-b7a1da81399e)

## Contents


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

### Code Reproducibility: 

The code has not been comprehensively checked and re-run since refactoring. If you're having any issues, find
a problem/bug or cannot reproduce similar results as the paper please [open an issue](https://github.com/Decadz/Sparse-Label-Smoothing-Regularization/issues)
or email me.

## Reference

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
