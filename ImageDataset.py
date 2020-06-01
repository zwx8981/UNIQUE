import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I


def get_default_img_loader():
    return functools.partial(image_loader)


class ImageDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 transform=None,
                 test=False,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        print('start loading csv data...')
        self.data = pd.read_csv(csv_file, sep='\t', header=None)
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.test = test
        self.transform = transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        if self.test:
            image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
            I = self.loader(image_name)
            if self.transform is not None:
                I = self.transform(I)

            mos = self.data.iloc[index, 1]
            std = self.data.iloc[index, 2]
            sample = {'I': I, 'mos': mos, 'std': std}
        else:
            image_name1 = os.path.join(self.img_dir, self.data.iloc[index, 0])
            image_name2 = os.path.join(self.img_dir, self.data.iloc[index, 1])

            I1 = self.loader(image_name1)
            I2 = self.loader(image_name2)
            if self.transform is not None:
                I1 = self.transform(I1)
                I2 = self.transform(I2)

            y = torch.FloatTensor(self.data.iloc[index, 2:].tolist())

            #std1 = torch.FloatTensor(self.data.iloc[index, 3].tolist())

            #std2 = torch.FloatTensor(self.data.iloc[index, 4].tolist())

            #y = torch.from_numpy(self.data.iloc[index, 2])
            #std1 = torch.from_numpy(self.data.iloc[index, 3])
            #std2 = torch.from_numpy(self.data.iloc[index, 4])

            #y = torch.FloatTensor(self.data.iloc[index, 2])
            #std1 = torch.FloatTensor(self.data.iloc[index, 3])
            #std2 = torch.FloatTensor(self.data.iloc[index, 4])

            #y = self.data.iloc[index, 2:].tolist()
            #y = y[0]
            sample = {'I1': I1, 'I2': I2, 'y': y[0], 'std1':y[1], 'std2':y[2], 'yb': y[3]}

        return sample

    def __len__(self):
        return len(self.data.index)
