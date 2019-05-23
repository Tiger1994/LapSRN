import torch.utils.data as data
import torch
import numpy as np
import h5py
import os
import scipy.misc as misc

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("data")
        self.label_x2 = hf.get("label_x2")
        self.label_x4 = hf.get("label_x4")

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.label_x2[index,:,:,:]).float(), torch.from_numpy(self.label_x4[index,:,:,:]).float()

    def __len__(self):
        return self.data.shape[0]


class DatasetFromFolder(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromFolder, self).__init__()
        self.LR_paths = get_image_path(file_path['LR'])
        self.x2_paths = get_image_path(file_path['x2'])
        self.x4_paths = get_image_path(file_path['x4'])

        assert(len(self.LR_paths) == len(self.x2_paths) and
               len(self.x2_paths) == len(self.x4_paths))

    def __getitem__(self, item):
        # lr = read_img(self.LR_paths[item])
        # x2 = read_img(self.x2_paths[item])
        # x4 = read_img(self.x4_paths[item])
        lr = np.load(self.LR_paths[item])
        x2 = np.load(self.x2_paths[item])
        x4 = np.load(self.x4_paths[item])
        lr = lr[np.newaxis, :]
        x2 = x2[np.newaxis, :]
        x4 = x4[np.newaxis, :]

        return torch.from_numpy(lr).float(), torch.from_numpy(x2).float(), torch.from_numpy(x4).float()

    def __len__(self):
        return len(self.LR_paths)


def get_image_path(path):
    assert(os.path.isdir(path))
    files = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            binary_path = os.path.join(dirpath, fname)
            files.append(binary_path)
    return files


def read_img(path):
    img = misc.imread(path)
    return img
