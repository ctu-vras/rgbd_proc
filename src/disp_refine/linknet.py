import torch
import segmentation_models_pytorch as smp


class DispRef(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Linknet(
            encoder_name='mobilenet_v2',
            encoder_weights='imagenet',
            in_channels=2,
            classes=1,
        )
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        return self.activation(self.model(x))
