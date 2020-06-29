import os
import numpy as np
from torch.utils import data
import pickle
from utils import bcolors


class RAKIDataHandler(data.Dataset):
    """
    This is a torch class to handle:
    subsampled data always stored as [real,imag, real,imag, ...]
    1. loading MR images
    2. cropping
    3. batching
    """
    def __init__(self, config):
        self.config = config
        self.crop_size = config['network']['crop_size']
        self.work_with_crop = config['network']['work_with_crop']

        # check if data dir exists
        try:
            assert os.path.isdir(config['data']['data_folder'])
            self.data_path = config['data']['data_folder']
        except AssertionError as error:
            print(error)
            print(bcolors.FAIL + 'ERROR:\nChosen Data folder does not exist. Please go to config file and update.')
            print(bcolors.FAIL + 'Or enter a command line path python main.py -data /path/to/data/folder')
            exit(1)

        self.data = self.load_mr_images()
        self.sub_rate = self.config['acceleration_rate']
        self.ACS_size = 40
        self.ACS = self.get_ACS()
        self.subsampled_data = self.create_subsmapled_data()
        print('-datahandler built-')

    def __len__(self):
        """
        function required to override the built in torch methods
        :return: the length of the dataset
        """
        return self.data.shape[0]

    def __getitem__(self, item):
        """
        return a training example
        :param item: not used, required in the signature by torch
        :return: a training example
        """
        # here we get a crop that is concatenated in the channels dimension
        # [real, imag] - so the first half is real second is imaginary - for reconstruction
        hr_crop = self.get_random_crop()

        # gt_crop is prepared for the network we store the all the subsampling in the channel dimension
        # lr_crop is only every Nth row - that we need to feed through the network
        gt_crop, lr_crop = self.subsample_crop(hr_crop)
        return gt_crop, lr_crop

    def load_mr_images(self):
        """
        Load the MR images from a preset pickle file
        :return: np tensor [subject, width, height, channel]
        """
        # iterate over the path list load and save the data
        file_list = sorted([f for f in os.listdir(self.data_path) if f.endswith('pickle')])
        data = []
        for ind, path in enumerate([file_list[0]]):
            if ind == 0:
                with open(f'{os.path.join(self.data_path,path)}', 'rb') as handle:
                    data = np.squeeze(pickle.load(handle))
            else:
                with open(f'{os.path.join(self.data_path,path)}', 'rb') as handle:
                    tmp_data = np.squeeze(pickle.load(handle))

                data = np.concatenate([data, tmp_data], axis=0)

        return (data[:, :, :768, :])/np.std(data[:, :, :768, :])

    def get_ACS(self):
        """
        Crop out the central region of the K-sapce which is fully sampled
        :return: np tensor [subject, width, height, channel]
        """
        ACS = list(np.arange(-self.ACS_size // 2, self.ACS_size // 2) + 308)
        return self.data[:, :, ACS[:], :]

    def create_subsmapled_data(self):
        """
        Subsample the input tensor along the Y-axis by a factor of sub_rate (in the class, read from config)
        :return: subsampled and zero filled tensor of the same shape as the input
        """

        # concate the real and imaginary parts of the tensor along the channel dimensions
        real_img_data = np.concatenate([np.real(self.data), np.imag(self.data)], axis=1)
        sub_data = np.zeros([real_img_data.shape[0], real_img_data.shape[1]*self.sub_rate,
                             real_img_data.shape[2]//self.sub_rate, real_img_data.shape[3]], dtype=real_img_data.dtype)

        # the channels are arranged [frames, [channels r(mod0), channels r(mod1), ...], y,x]
        # if R=2 [frames, [channels even_rows, channels odd, ...], y,x]
        for i in range(real_img_data.shape[-2]):
            sub_data[:, real_img_data.shape[1]*(i%self.sub_rate):real_img_data.shape[1]*(i%self.sub_rate+1), i//self.sub_rate, :] = real_img_data[:, :, i, :]

        return sub_data

    def subsample_crop(self, hr_crop, test=False):
        """
        subsample a given training crop
        :param hr_crop: [2*channels, height, width]
        :return: zero-padded subsampled crop
        """
        gt_crop = np.zeros([hr_crop.shape[0]*self.sub_rate, hr_crop.shape[1]//self.sub_rate, hr_crop.shape[2]], dtype=hr_crop.dtype)

        for i in range(hr_crop.shape[-2]):
            start_ind = hr_crop.shape[0]*(i%self.sub_rate)
            end_ind = start_ind + hr_crop.shape[0]
            gt_crop[start_ind:end_ind, i//self.sub_rate, :] = hr_crop[:, i, :]

        lr_crop = gt_crop[:hr_crop.shape[0], :, :]
        return gt_crop, lr_crop

    def get_random_crop(self):
        """
        Randomly select a crop (currently no set, we work on the full ACS in training)
        :return: tensor of preset size
        """
        if self.work_with_crop:
            return
        # not working with crops, but with the full k-space ACS
        else:
            frame = np.random.randint(0, self.data.shape[0], 1)
            real_part = np.real(np.squeeze(self.ACS[frame, :, :, :]))
            imag_part = np.imag(np.squeeze(self.ACS[frame, :, :, :]))
            crop = np.concatenate([real_part, imag_part], axis=0)
            return crop


class SpatialDataHandler(RAKIDataHandler):
    """
    ---- currently not set ------
    This is a torch class to handle:
    subsampled data always stored as [real,imag, real,imag, ...]
    1. loading MR images
    2. cropping
    3. batching
    """
    def __init__(self, config):
        super().__init__(config)
        self.subsampled_data = self.create_subsmapled_data()
        print('-datahandler built-')

    def __len__(self):
        return self.data.shape[0]

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
        for ind, path in enumerate([file_list[0]]):
            if ind == 0:
                with open(f'{os.path.join(self.data_path,path)}', 'rb') as handle:
                    data = np.squeeze(pickle.load(handle))
            else:
                with open(f'{os.path.join(self.data_path,path)}', 'rb') as handle:
                    tmp_data = np.squeeze(pickle.load(handle))

                data = np.concatenate([data, tmp_data], axis=0)

        return (data[:, :, :768, :])/np.std(data[:, :, :768, :])

    def get_ACS(self):
        ACS = list(np.arange(-self.ACS_size // 2, self.ACS_size // 2) + 308)
        return self.data[:, :, ACS[:], :]

    def create_subsmapled_data(self):
        real_img_data = np.concatenate([np.real(self.data), np.imag(self.data)], axis=1)
        return real_img_data[:, :, ::self.sub_rate, :]

    def subsample_crop(self, hr_crop, test=False):
        """
        :param hr_crop: [2*channels, height, width]
        :return: zero-padded subsampled crop
        """
        lr_crop = hr_crop[:, ::self.sub_rate, :]
        return lr_crop

    def get_random_crop(self):
        if self.work_with_crop:
            return
        # not working with crops, but with the full k-space ACS
        else:
            frame = np.random.randint(0, self.data.shape[0], 1)
            real_part = np.real(np.squeeze(self.ACS[frame, :, :, :]))
            imag_part = np.imag(np.squeeze(self.ACS[frame, :, :, :]))
            crop = np.concatenate([real_part, imag_part], axis=0)
            return crop
