import torch
from mmrnet.models.pointnet import PointNet

def debug_pointnet_output():
    info = {
        "num_classes": 10,          # any valid value, not used when set to None below
        "stacks": 5,                # same as in your config
        "num_points": 1024,         # same as in your config
        "num_keypoints": 1024,      # match num_points for this test
        "in_channels": 3,           # xyz
        "task": "action",           # matches mmr_action
        # add any other keys PointNet expects from your config
    }

    pointnet_info = dict(info)
    pointnet_info["num_classes"] = None
    model = PointNet(pointnet_info)

    B = 2
    N = info["num_points"]
    x = torch.randn(B, N, 3)

    with torch.no_grad():
        y = model.extract_global_feature(x)
    print("Global feature shape:", y.shape)

if __name__ == "__main__":
    debug_pointnet_output()
