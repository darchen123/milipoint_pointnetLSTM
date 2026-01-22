import torch
import torch.nn as nn
# import torch.nn.functional as F
from mmrnet.models.pointnet import PointNet  

class PointnetLSTMActionClassification(nn.Module):
    def __init__(self, info):
        super().__init__()

        self.num_classes = info["num_classes"]
        self.T = info["stacks"]
        self.num_points = info["max_points"]

        # Create PointNet backbone without Classification head
        pointnet_info = dict(info)
        pointnet_info["num_classes"] = None
        self.pointnet = PointNet(pointnet_info)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size = 1024,
            hidden_size = 512,
            num_layers = 2,
            batch_first = True
        )

        # Final classifier
        self.fc = nn.Linear(512, self.num_classes)

    def forward(self, stacked):
        # DEBUG: shape + config checks
        # print("stacked.shape:", stacked.shape)
        # print("T (from info):", self.T)
        # print("num_points (from info):", self.num_points)
        # print("T * num_points:", self.T * self.num_points)
        
        # Currently stacked = [B, T*N, 3]
        # Need to reshape to [B, T, N, 3]
        B, TN, _ = stacked.shape
        assert TN == self.T * self.num_points, (
            f"Input size mismatch: Got {TN} points expected T*N={self.T * self.num_points}"
        )

        # Restore temporal structure
        frames = stacked.view(B, self.T, self.num_points, 3)

        seq_feats = []
        for t in range(self.T) :
            ft = self.pointnet.extract_global_feature(frames[:, t]) # [B, 1024]
            seq_feats.append(ft)

        feats = torch.stack(seq_feats, dim = 1) # [B, T, 1024]

        lstm_output, _ = self.lstm(feats)
        last = lstm_output[:, -1, :]

        return self.fc(last)