# Test the watermark performance on a random model that was not stolen from the victim model.

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda')
batch_size = 64
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, \
    RegularDataset, WatermarkDataset
from models.resnet_simclr import ResNetSimCLRV2, SimSiam, WatermarkMLP
from utils import load_victim, load_watermark, accuracy
import os

watermark_dataset = WatermarkDataset('/ssd003/home/akaleem/data').get_dataset(
                "cifar10", 2)
watermark_loader = torch.utils.data.DataLoader(
    watermark_dataset, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True)
watermark_mlp = WatermarkMLP(512, 4).to(device) # was 2 before
watermark_mlp = load_watermark(100, "cifar10",
                                       watermark_mlp,
                                       "resnet18", "infonce",
                                       device=device)

model = ResNetSimCLRV2(base_model="resnet34", out_dim=128, loss=None, include_mlp = False).to(device)
checkpoint = torch.load(
            f"/checkpoint/{os.getenv('USER')}/SimCLR/100resnet34mseSTEAL/stolen_checkpoint_9000_mse_cifar10.pth.tar",
            map_location=device)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)  # random stolen model (not from the watermarked victim)

# model = ResNetSimCLRV2(base_model="resnet18",
#                                   out_dim=128, loss=None,
#                                   include_mlp=False).to(device)
# model = load_victim(100, "cifar10", model,
#                            "resnet18", "infonce",
#                            device=device, discard_mlp=True,
#                            watermark="True") # watermarked victim


# model = ResNetSimCLRV2(base_model="resnet34",
#                                   out_dim=128, loss=None,
#                                   include_mlp=False).to(device)
# model = load_victim(200, "cifar10", model,
#                            "resnet34", "infonce",
#                            device=device, discard_mlp=True,
#                            watermark="False")
model.eval()
watermark_mlp.eval()
watermark_accuracy = 0
for counter, (x_batch, _) in enumerate(watermark_loader):
    x_batch = torch.cat(x_batch, dim=0)
    x_batch = x_batch.to(device)
    logits = watermark_mlp(model(x_batch))
    # y_batch = torch.cat([torch.zeros(batch_size),
    #                      torch.ones(batch_size)],
    #                     dim=0).long().to(device)
    y_batch = torch.cat([torch.zeros(batch_size),
                         torch.ones(batch_size), 2*torch.ones(batch_size), 3*torch.ones(batch_size)],
                        dim=0).long().to(device)
    top1 = accuracy(logits, y_batch, topk=(1,))
    watermark_accuracy += top1[0]
watermark_accuracy /= (counter + 1)
print(f"Watermark accuracy is {watermark_accuracy.item()}.")