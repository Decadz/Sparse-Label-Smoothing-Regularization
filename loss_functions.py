import torch


class SparseLSRLoss(torch.nn.Module):

    def __init__(self, smoothing=0.0, reduction="mean"):

        """
        An implementation of Sparse Label Smoothing Regularization (SparseLSR)
        Loss implemented in PyTorch. This loss function was presented in
        "Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning"
        (TPAMI-2023). Paper Link: https://arxiv.org/abs/2209.08907

        :param smoothing: Smoothing coefficient value.
        :param reduction: Loss function reduction.
        """

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


class LSRLoss(torch.nn.Module):

    def __init__(self, smoothing=0.0, reduction="mean"):

        """
        A prototypical implementation of Label Smoothing Regularization (LSR) loss.
        This is a standard implementation which is non-sparse, and computes the loss
        over all outputs for each instance in a batch.

        :param smoothing: Smoothing coefficient value.
        :param reduction: Loss function reduction.
        """

        super(LSRLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, y_pred, y_target):

        # Retrieving the total number of classes.
        num_classes = torch.tensor(y_pred.size(1))

        # Computing the log probabilities using numerically stable log-sum-exp.
        log_prob = torch.nn.functional.log_softmax(y_pred, dim=1)

        # Converting the target to one-hot encoded form.
        y_target = torch.nn.functional.one_hot(y_target, num_classes=num_classes)

        # Computing the fast label smoothing regularization loss.
        loss = (y_target * (1 - self.smoothing) + (self.smoothing/num_classes)) * log_prob

        # Summing the loss across the outputs.
        loss = - loss.sum(dim=1)

        # Applying the reduction and returning.
        return loss.mean() if self.reduction == "mean" else loss
