import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        """

        :param preds: [1, N]
        :param labels: [1, N]
        :return:
        """
        self.alpha = torch.tensor(self.alpha, device=preds.device)
        # preds = F.softmax(preds, dim=1)
        preds = preds.sigmoid()
        preds = preds.view(-1)
        labels = labels.view(-1)

        if self.alpha:
            self.alpha = self.alpha.type_as(preds.data)
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        else:
            alpha_t = torch.tensor(1)

        pt = preds * labels + (1 - preds) * (1 - labels) + 1e-10
        diff = torch.pow((1 - pt), self.gamma)

        loss = -1 * alpha_t * diff * pt.log()

        # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
