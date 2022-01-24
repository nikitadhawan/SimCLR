import torch


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, transform=None):
        self.x_data = x
        self.y_data = torch.from_numpy(y).long()
        self.transform = transform
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        x_data_index = self.x_data[index]
        if self.transform:
            x_data_index = self.transform(x_data_index)
        return (x_data_index, self.y_data[index])

    def __len__(self):
        return self.len
