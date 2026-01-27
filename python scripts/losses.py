import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLossEuclidean(nn.Module):
    """
    Standard Contrastive Loss (Hadsell, 2006).
    Label attese: 0 = Simili, 1 = Dissimili.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLossEuclidean, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        # Hadsell formulation
        loss_contrastive = torch.mean(
            (1-label) * 0.5 * torch.pow(euclidean_distance, 2) +
            (label)   * 0.5 * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

class ContrastiveLossCosine(nn.Module):
    """
    Cosine Embedding Loss.
    Label attese: 1 = Simili, -1 = Dissimili.
    """
    def __init__(self, margin=0.2):
        super(ContrastiveLossCosine, self).__init__()
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, output1, output2, label):
        return self.loss_fn(output1, output2, label)

class TripletLoss(nn.Module):
    """
    Triplet Margin Loss Standard.
    Input: Anchor, Positive, Negative.
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)