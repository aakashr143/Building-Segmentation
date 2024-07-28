import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchsummary import summary
import constants
from UANet.UANet import UANet_Res50

from enum import StrEnum


# https://www.kaggle.com/code/wadewayne001/building-segmentation-using-resnet50-and-u-net


def init_weights(m: nn.Module, depth=0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    else:
        if depth != 0:
            m.apply(lambda x: init_weights(x, depth + 1))


class Encoder(StrEnum):
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"


class DeepLabV3Plus(nn.Module):
    def __init__(self, encoder: Encoder = Encoder.RESNET18, edge_detection=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone = encoder
        self.architecture = "DeepLabV3Plus"

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=4 if edge_detection else 3,
            classes=constants.NUM_CLASSES,
        )

        self.out_activation = nn.Softmax2d()

        for params in self.model.named_parameters():
            if "encoder" in params[0]:
                params[1].requires_grad = False

        self.model.decoder.apply(init_weights)

    def forward(self, x):
        out = self.model(x)
        return self.out_activation(out)


class UNet(nn.Module):
    def __init__(self, encoder: Encoder = Encoder.RESNET18, edge_detection=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = encoder
        self.architecture = "UNet"

        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=4 if edge_detection else 3,
            classes=constants.NUM_CLASSES,
        )

        self.out_activation = nn.Softmax2d()

        for params in self.model.named_parameters():
            if "encoder" in params[0]:
                params[1].requires_grad = False

        self.model.decoder.apply(init_weights)

    def forward(self, x):
        out = self.model(x)
        return self.out_activation(out)


class UANet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = "ResNet50"
        self.architecture = "UANet"

        self.model = UANet_Res50(32, constants.NUM_CLASSES)
        self.out_activation = nn.Softmax2d()

        for params in self.model.backbone.named_parameters():
            params[1].requires_grad = False

    def forward(self, x):
        out = self.model(x)
        return self.out_activation(out)


if __name__ == "__main__":
    model = UNet(Encoder.RESNET50)
    # model = DeepLabV3Plus(Encoder.RESNET50)
    # model = UANet()
    #summary(model, torch.randn((1, 3, 512, 512)))
    print(model)