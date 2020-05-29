import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
import ismrmrd
import ismrmrd.xsd
from tqdm import tqdm
import pickle

class DataHandler(data.Dataset):
    """
    This is a torch class to handle:
    1. loading MR images
    2. cropping
    3. batching
    """
    def __init__(self, config):
        self.config = config
        self.crop_size = config['network']['crop_size']
        self.work_with_crop = config['network']['work_with_crop']
        self.data_path = config['data']['data_folder']
        self.data = self.load_mr_images()
        self.sub_rate = self.config['acceleration_rate']
        self.ACS, self.elements_to_leave, self.elements_to_remove, self.subsampled_data = self.create_subsmapled_data()

    def __len__(self):
        return self.data.shape

    def __getitem__(self, item):
        """
        return an item
        :param item:
        :return:
        """
        hr_crop = self.get_random_crop()
        lr_crop = self.subsample_crop(hr_crop)
        return hr_crop, lr_crop

    def load_mr_images(self):
        # iterate over the path list load and save the data
        file_list = sorted([f for f in os.listdir(self.data_path) if f.endswith('pickle')])
        data = []
        for ind, path in enumerate(file_list[:3]):
            if ind == 0:
                with open(f'{os.path.join(self.data_path,path)}', 'rb') as handle:
                    data = np.squeeze(pickle.load(handle))
            else:
                with open(f'{os.path.join(self.data_path,path)}', 'rb') as handle:
                    tmp_data = np.squeeze(pickle.load(handle))

                data = np.concatenate([data, tmp_data], axis=0)

        return data

    def create_subsmapled_data(self):
        fully_sampled_area_size = self.data.shape[2]//8
        sub_rate = self.sub_rate
        sub_data = np.copy(self.data)
        ACS = list(np.arange(-fully_sampled_area_size//2, fully_sampled_area_size//2) + 308)
        # elements_to_leave = []
        elements_to_remove = list(range(0, self.data.shape[2]))
        elements_to_leave = list(range(0, self.data.shape[2], sub_rate))

        for f in elements_to_remove:
            if f in ACS or f in elements_to_leave:
                continue
            else:
                elements_to_remove.remove(f)

        for i in elements_to_remove:
            sub_data[:, :, i, :] = np.zeros(np.shape(sub_data[:, :, 0, :]), dtype=sub_data.dtype)
        return ACS, elements_to_leave, elements_to_remove, sub_data

    def subsample_crop(self, hr_crop):
        """
        take hr_crop and replace each N-th row with zeros (subsampled and padded)
        :param hr_crop: [2*channels, height, width]
        :return: zero-padded subsampled crop
        """
        lr_crop = np.copy(hr_crop)
        for i in range(0, hr_crop.shape[1], self.sub_rate):
            lr_crop[:, i, :] = np.zeros(np.shape(lr_crop[:, i, :]), dtype=lr_crop.dtype)
        return lr_crop

    def get_random_crop(self):
        if self.work_with_crop:
            frame = np.random.randint(0, self.data.shape[0], 1)
            row = np.random.randint(self.ACS[0], self.ACS[-1] - self.crop_size, 1)
            col = np.random.randint(0, self.data.shape[0] - self.crop_size, 1)
            real_part = np.real(np.squeeze(self.subsampled_data[frame, :, row, col]))
            imag_part = np.squeeze(np.imag(self.subsampled_data[frame, :, row, col]))
            crop = np.concatenate([real_part, imag_part], axis=0)
            return crop
        # not working with crops, but with the full k-space ACS
        else:
            frame = np.random.randint(0, self.data.shape[0], 1)
            real_part = np.real(np.squeeze(self.subsampled_data[frame, :, self.ACS[0]:self.ACS[-1], :]))
            imag_part = np.squeeze(np.imag(self.subsampled_data[frame, :, self.ACS[0]:self.ACS[-1], :]))
            crop = np.concatenate([real_part, imag_part], axis=0)
            return crop
