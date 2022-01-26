from torch import nn


def get_model(a_num):
    model = nn.Sequential(nn.Linear(a_num, 100), nn.ReLU(), nn.Linear(100, 1),
                          nn.Tanh())
    return model
