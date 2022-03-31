import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class ConvNet(nn.Module):   # specifically for mnist or F-mnist
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.out = nn.Linear(1024, 512)
        self.out2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.out(x))
        x = self.out2(x)
        return x

class ConvNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, loss=None, include_mlp = True, entropy=False):
        super(ConvNetSimCLR, self).__init__()
        self.backbone = self._get_basemodel(base_model, out_dim=out_dim)
        self.loss = loss
        self.entropy = entropy
        dim_mlp = self.backbone.out2.in_features # 512
        if include_mlp:
            # add mlp projection head
            # originally self.backbone.fc = nn.Linear(512 * block.expansion, num_classes)
            # This modifies the out2 layer to add another hidden layer inside.
            if self.loss == "symmetrized":
                self.backbone.out2 = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                 nn.BatchNorm1d(dim_mlp),
                                                 nn.ReLU(inplace=True), self.backbone.fc)
            else:
                self.backbone.out2 = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.out2)
        else:
            #self.backbone.fc = nn.Linear(dim_mlp, dim_mlp, bias=False)
            #self.backbone.fc.weight.data.copy_(torch.eye(dim_mlp))
            #self.backbone.fc.weight.requires_grad = False # last layer does nothing. only there to be compatible with resnet
            self.backbone.out2 = nn.Identity() # no head used

    def _get_basemodel(self, model_name,out_dim):
        try:
            model = ConvNet(num_classes=out_dim)
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18, resnet34 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


def trial():
    device = 'cuda'
    model = ConvNetSimCLR(base_model="convnet", out_dim=128,
                 entropy=False, include_mlp=True).to(device)
    from torchsummary import summary
    summary(model, input_size = (1,28,28))



if __name__ == "__main__":
    trial()