import torch
import torch.nn as nn
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    """Compute contrastive loss"""

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def sim(self, im, s):
        """Cosine similarity between all the image and sentence pairs"""
        return im.mm(s.t())

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, nmax=1, margin=0.2):
        super(HardNegativeContrastiveLoss, self).__init__()
        self.margin = margin
        self.nmax = nmax

    def forward(self, imgs, caps):
        scores = torch.mm(imgs, caps.t())
        diag = scores.diag()

        # Reducing the score on diagonal so there are not selected as hard negative
        scores = scores - 2 * torch.diag(scores.diag())

        sorted_cap, _ = torch.sort(scores, 0, descending=True)
        sorted_img, _ = torch.sort(scores, 1, descending=True)

        # Selecting the nmax hardest negative examples
        max_c = sorted_cap[: self.nmax, :]
        max_i = sorted_img[:, : self.nmax]

        # Margin based loss with hard negative instead of random negative
        neg_cap = torch.sum(
            torch.clamp(
                max_c + (self.margin - diag).view(1, -1).expand_as(max_c), min=0
            )
        )
        neg_img = torch.sum(
            torch.clamp(
                max_i + (self.margin - diag).view(-1, 1).expand_as(max_i), min=0
            )
        )

        loss = neg_cap + neg_img

        return loss
