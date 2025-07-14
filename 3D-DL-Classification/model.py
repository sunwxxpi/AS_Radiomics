import torch
import torch.nn as nn
import monai
    

class CustomModel(nn.Module):
    def __init__(self, class_num=2):
        super(CustomModel, self).__init__()
        
        self.num_classes = class_num
        
        self.net = monai.networks.nets.resnet50(
            pretrained=True,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=class_num,
            feed_forward=False,
            shortcut_type='B',
            bias_downsample=False
        )
    
        self.in_features = self.net.in_planes
        self.net.fc = torch.nn.Linear(self.in_features, class_num)
        
    def forward(self, images=None):
        return self.net(images) # 더 간결하게 표현 가능