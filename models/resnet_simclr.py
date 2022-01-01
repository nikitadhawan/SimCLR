import torch.nn as nn
import torch
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, include_mlp = True):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet34": models.resnet34(pretrained=False,
                                                        num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features # 512
        if include_mlp:
            # add mlp projection head
            # originally self.backbone.fc = nn.Linear(512 * block.expansion, num_classes)
            # This modifies the fc layer to add another hidden layer inside.
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        else:
            self.backbone.fc = nn.Linear(dim_mlp, dim_mlp, bias=False)
            self.backbone.fc.weight.data.copy_(torch.eye(dim_mlp))
            self.backbone.fc.weight.requires_grad = False # last layer does nothing. only there to be compatible with resnet

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18, resnet34 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


class HeadSimCLR(nn.Module):
    """ Takes a representation as input and passes it through the head to get g(z)"""
    def __init__(self, out_dim):
        super(HeadSimCLR, self).__init__()
        self.expansion = 1 # 4 for resnet50+
        self.backbone = nn.Linear(512 * self.expansion, out_dim)
        dim_mlp = 512
        self.backbone = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone)


    def forward(self, x):
        return self.backbone(x)