import torch

def fbeta_score(y_true, y_pred, beta, eps=1e-9):
    beta2 = beta ** 2

    y_pred = y_pred.float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum()
    precision = true_positive.div(y_pred.sum().add(eps))
    recall = true_positive.div(y_true.sum().add(eps))

    return torch.mean((precision * recall).div(precision.mul(beta2) + recall + eps).mul(1 + beta2))

