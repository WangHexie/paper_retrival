import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
# from .BatchHardTripletLoss import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction
from sentence_transformers.SentenceTransformer import SentenceTransformer
import torch.nn.functional as F


class InfoNCE(nn.Module):

    def __init__(self, model: SentenceTransformer):
        super(InfoNCE, self).__init__()
        self.sentence_embedder = model
        self.T = 0.07
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        """
        number of negative samples
        :param sentence_features:
        :param labels:
        :return:
        """
        reps = [self.sentence_embedder(sentence_feature)['sentence_embedding'] for sentence_feature in
                sentence_features]

        device = reps[0].device

        anchors = reps[0].unsqueeze(1)
        pos = reps[1].unsqueeze(1)
        neg = torch.stack(reps[2:], dim=1)

        return self.infonce(anchors.to(device), pos.to(device), neg.to(device))

    def infonce(self, anchor_embedding, positive_embedding, negative_embedding):
        query = F.normalize(anchor_embedding, p=2, dim=2)
        pos = F.normalize(positive_embedding, p=2, dim=2)
        neg = F.normalize(negative_embedding, p=2, dim=2)

        pos_score = torch.bmm(query, pos.transpose(1, 2))  # B*1*1
        neg_score = torch.bmm(query, neg.transpose(1, 2))  # B*1*K

        # logits:B*(K+1)
        logits = torch.cat([pos_score, neg_score], dim=2).squeeze()
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(query.device)

        info_loss = self.cross_entropy(logits, labels)

        return info_loss
