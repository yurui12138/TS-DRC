
import torch
import numpy as np
from torch.nn import functional as F

class DRCLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(DRCLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.margin = 0.1

        self.pair_distance = torch.nn.PairwiseDistance(p=2)

    def _get_correlated_mask(self):
        diag = np.eye(N=self.batch_size, M=3 * self.batch_size)
        l_1 = np.eye(N=self.batch_size, M=3 * self.batch_size, k=self.batch_size)
        l_2 = np.eye(N=self.batch_size, M=3 * self.batch_size, k=self.batch_size * 2)
        mask = torch.from_numpy((diag + l_1 + l_2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    def forward(self, feature_ts, feature_image1, feature_image2):
        representations = torch.cat([feature_ts, feature_image1, feature_image2], dim=0)
        expanded_tensor1 = feature_ts.unsqueeze(1)
        expanded_tensor2 = representations.unsqueeze(0)

        distance_matrix = torch.cdist(expanded_tensor1, expanded_tensor2, p=2.0).view(feature_ts.shape[0], -1)
        distance_matrix = F.relu(distance_matrix)

        distance_matrix_2 = self.pair_distance(feature_image1, feature_image2)

        l_pos_1 = torch.diag(distance_matrix, self.batch_size)
        l_pos_2 = torch.diag(distance_matrix, self.batch_size*2)
        positives = (l_pos_1 + l_pos_2 + distance_matrix_2).view(self.batch_size, 1)

        positive_value, positive_index = torch.max(torch.stack([l_pos_1, l_pos_2]), dim=0)
        positive_value = positive_value.view(-1, 1)

        negatives = distance_matrix[self.mask_samples_from_same_repr].view(self.batch_size, -1)
        negative_value, negative_index = torch.min(negatives, dim=-1)

        negatives = negative_value.view(-1, 1)

        triplet_loss = positives - negatives + self.margin
        triplet_loss = F.relu(triplet_loss)

        triplet_loss += positive_value
        triplet_loss = torch.sum(triplet_loss)

        return triplet_loss / (self.batch_size)
