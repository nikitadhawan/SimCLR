import os
import shutil

import torch
import yaml
import gdown


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'),
                  'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_victim(epochs, dataset, model, arch, loss, device, discard_mlp=False):
    checkpoint = torch.load(
        f"/checkpoint/{os.getenv('USER')}/SimCLR/{epochs}{arch}{loss}TRAIN/{dataset}_checkpoint_{epochs}_{loss}.pth.tar",
        map_location=device)
    # checkpoint = torch.load(
    #     f'/ssd003/home/akaleem/SimCLR/runs/{dataset}_checkpoint_{epochs}.pth.tar', map_location=device)
    state_dict = checkpoint['state_dict']

    if discard_mlp:
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix and only save if not backbone.fc (i.e. not including projection head)
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)
    return model
