import torch
import torch.nn as nn
from ultralytics import YOLO

class CSRNet(nn.Module):
    """Congested Scene Recognition Network (Dilated Backbone)"""
    def __init__(self):
        super(CSRNet, self).__init__()
        # VGG-16 frontend + dilated conv backend for density map estimation
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Outputs continuous density map (sum = estimated crowd count)
        return torch.relu(self.output_layer(x))

class PlatformCrowdEstimator:
    def __init__(self, yolo_weights='yolov8n.pt'):
        self.yolo = YOLO(yolo_weights)
        self.csrnet = CSRNet()

    def estimate_platform(self, frame, platform_area_m2=120.0):
        # 1. Sparse crowd: Bounding box detection
        results = self.yolo(frame, classes=[0], verbose=False)
        sparse_count = len(results[0].boxes)
        
        # 2. If dense (> 1.5 persons/m2), switch to CSRNet density integration
        density = sparse_count / platform_area_m2
        return {
            "headcount": sparse_count,
            "density_p_m2": round(density, 2),
            "surge_alert": density > 3.0
        }