import torch


class NumpyDataset(torch.utils.data.Dataset):
    """Class to create a Pytorch Dataset from Numpy data"""

    def __init__(self, X, Y, transform=None, target_transform=None):
        super(NumpyDataset, self).__init__()

        self.X = X
        self.Y = Y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):

        ima,target = self.X[index], self.Y[index]

        if self.transform is not None:
            ima = self.transform(ima)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return ima, target