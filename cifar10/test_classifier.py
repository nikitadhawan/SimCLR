'''Test CIFAR10 with PyTorch.'''
import argparse
import os
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from cifar10.utils import progress_bar

from models.resnet import ResNet18

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.pth')

if 'net' in checkpoint:
    key = 'net'
elif 'state_dict' in checkpoint:
    key = 'state_dict'
else:
    raise Exception("Missing key net of state dict in the checkpoint.")
net.load_state_dict(checkpoint[key])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
net = net.module

state = {
    'state_dict': net.state_dict(),
    'acc': best_acc,
    'epoch': start_epoch,
}
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(state, './checkpoint/ckpt_state_dict.pth')

print('start epoch: ', start_epoch)
print('best acc: ', best_acc)

criterion = nn.CrossEntropyLoss()


def test(epoch, data_loader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(data_loader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total,
                            correct, total))
    acc = 100 * correct / total
    return acc


if __name__ == "__main__":
    train_acc = test(epoch=start_epoch, data_loader=trainloader)
    test_acc = test(epoch=start_epoch, data_loader=testloader)

    print(f"train acc: {train_acc}%, test acc: {test_acc}%")
