import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim.optimizer
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter


class RAKINetwork:  # The base network
    """
    Network class.
    An object to wrap the network with all the methods.
    build, train, predict, save & load models, track performance
    """

    def __init__(self, config, device):
        self.device = device
        self.config = config
        self.channels_in = config['network']['input_channels']
        self.R = config['acceleration_rate']

        self.net = self.build_network()
        self.optimizer = self.define_opt()
        self.loss_fn = self.define_loss()
        self.writer = SummaryWriter(os.path.join(config['working_dir'], 'logs_dir'))
        self.scheduler = self.define_lr_sched()
        print('-net built-')

    def build_network(self):
        """
        BASE version. Other modes override this function
        :return: pytorch net object
        """
        net = nn.Sequential(
            nn.Conv2d(in_channels=2 * self.channels_in, out_channels=128, kernel_size=[3, 5], padding=[1, 2], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], padding=[0, 0], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 3], padding=[0, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=self.R * 2 * self.channels_in, kernel_size=[3, 3], padding=[1, 1], padding_mode='replicate'),
        ).to(self.device)
        return net

    @staticmethod
    def define_loss():
        return torch.nn.L1Loss(reduction='sum')

    def define_opt(self):
        # TODO: read the lr and momentum from config
        learning_rate = self.config['network']['optimization']['params']['lr']
        return torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=0.01)

    def define_lr_sched(self):
        """
        take relevant parameters for learning rate scheduler
        :return: lr_scheduler object
        """
        gamma = self.config['network']['lr_sched']['params']['gamma']
        milestones = self.config['network']['lr_sched']['params']['milestones']
        step_size = self.config['network']['lr_sched']['params']['step_size']
        epochs = self.config['network']['num_epochs']
        if self.config['network']['lr_sched']['name'] == 'MultiStepLR':
            return lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        elif self.config['network']['lr_sched']['name'] == 'StepLR':
            return lr_scheduler.StepLR(self.optimizer, step_size=int(epochs * step_size), gamma=gamma)
        else:
            print('****************** NO LR_SCHED DEFINED SETTING DEFAULT *****************************')
            return lr_scheduler.StepLR(self.optimizer, step_size=epochs // 10, gamma=1 / 1.5)

    def forward(self, input_tensor):
        return self.net(input_tensor)

    def calc_loss(self, output, hr_gt_torch):
        return self.loss_fn(output, hr_gt_torch).cuda()

    def train(self, data_loader_object):
        print('-starting training-')
        epochs = self.config['network']['num_epochs']
        for e in range(epochs):
            t = time.time()
            self.optimizer.zero_grad()
            if e % self.config['network']['save_every'] == self.config['network']['save_every'] - 1:
                print(f'saved model at epoch {e}')
                self.save_model(epoch=e, overwrite=False)

            # iterations per epochs
            it = 0
            for (hr_gt, lr) in data_loader_object:
                hr_prediction = self.forward(lr.to(self.device))
                # loss = self.calc_loss(hr_prediction.to(self.device), hr_gt.to(self.device))
                loss1 = torch.nn.L1Loss(reduction='sum')
                # loss2 = torch.nn.MSELoss(reduction='sum')
                loss3 = 0.0 * (
                               torch.sum(torch.abs(hr_prediction[:, :, :, :-1] - hr_prediction[:, :, :, 1:])) +
                               torch.sum(torch.abs(hr_prediction[:, :, :-1, :] - hr_prediction[:, :, 1:, :]))
                               )
                loss = loss1(hr_prediction.to(self.device), hr_gt.to(self.device)) + loss3
                loss.backward()
                it += 1
            print(f'epoch:{e}, loss:{loss.item()}. Time: {(time.time() - t):.2f}, lr={self.optimizer.param_groups[0]["lr"]}')
            # TODO: check about updating after ALL iterations in epoch
            self.optimizer.step()
            self.scheduler.step()
            self.writer.add_scalars('loss', {'loss': loss.item()})
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]["lr"]})

        self.writer.close()
        return

    def eval(self, data):
        #                       frames       channels(rea/imag)     y (full size)           x
        hr_tensor = np.zeros([data.shape[0], data.shape[1] // 2, self.R * data.shape[2], data.shape[3]], dtype='complex128')

        # divide by 2 for real/imag
        num_input_channels = data.shape[1] // 2

        # iterate over the frames and produce prediction for each frame
        for ind, t in enumerate(data):
            cur_tensor = t[np.newaxis, :, :, :]
            tmp = self.forward(torch.from_numpy(cur_tensor).to(self.device))
            for i in range(self.R):
                hr_tensor[ind, :, i::self.R, :] = tmp.detach().cpu().numpy()[:, 2 * i * num_input_channels:(2 * i + 1) * num_input_channels, :, :] + 1j * \
                                                  tmp.detach().cpu().numpy()[:, (2 * i + 1) * num_input_channels:(2 * i + 2) * num_input_channels, :, :]
        return hr_tensor

    def save_model(self, epoch=None, scale=None, overwrite=False):
        """
        Saves the model (state-dict, optimizer and lr_sched
        :return:
        """
        if overwrite:
            checkpoint_list = [i for i in os.listdir(os.path.join(self.config['working_dir'])) if i.endswith('.pth.tar')]
            if len(checkpoint_list) != 0:
                os.remove(os.path.join(self.config['working_dir'], checkpoint_list[-1]))

        filename = 'checkpoint{}{}.pth.tar'.format('' if epoch is None else '-e{:05d}'.format(epoch),
                                                   '' if scale is None else '-s{:02d}'.format(scale))
        torch.save({'sd': self.net.state_dict(),
                    'opt': self.optimizer.state_dict()},
                   # 'lr_sched': self.scheduler.state_dict()},
                   os.path.join(self.config['working_dir'], filename))

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['sd'])
        self.optimizer.load_state_dict(checkpoint['opt'])
        # self.scheduler.load_state_dict(checkpoint['lr_sched'])


class SpatialNetwork(RAKINetwork):
    """
        Network class.
        An object to wrap the network with all the methods.
        build, train, predict, save & load models, track performance
        """

    def __init__(self, config, device):
        super().__init__(config, device)
        self.net = self.build_network()

    def build_network(self):  # BASE version. Other modes override this function
        net = nn.Sequential(
            nn.Conv2d(in_channels=2 * self.channels_in, out_channels=64, kernel_size=[3, 3], padding=[1, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], padding=[1,1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], padding=[1, 1], padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=2 * self.channels_in * self.R**2, kernel_size=[3, 3], padding=[1, 1], padding_mode='replicate'),
            nn.PixelShuffle(self.R),
        ).to(self.device)
        return net

    def train(self, data_loader_object):
        print('-starting training-')
        # epochs
        epochs = self.config['network']['num_epochs']
        for e in range(epochs):
            t = time.time()
            self.optimizer.zero_grad()
            if e % self.config['network']['save_every'] == self.config['network']['save_every'] - 1:
                print(f'saved model at epoch {e}')
                self.save_model(epoch=e, overwrite=False)

            # iterations per epochs
            it = 0
            for (hr_gt, lr) in data_loader_object:
                hr_prediction = nn.functional.interpolate(self.forward(lr.to(self.device)), scale_factor=(1, 1/self.R), mode='bilinear')
                # loss = self.calc_loss(hr_prediction.to(self.device), hr_gt.to(self.device))
                loss1 = torch.nn.L1Loss(reduction='mean')
                # loss2 = torch.nn.MSELoss(reduction='sum')
                loss = loss1(hr_prediction.to(self.device), hr_gt.to(self.device))
                loss.backward()
                it += 1
            print(f'epoch:{e}, loss:{loss.item()}. Time: {(time.time() - t):.2f}, lr={self.optimizer.param_groups[0]["lr"]}')
            # TODO: check about updating after ALL iterations in epoch
            self.optimizer.step()
            self.scheduler.step()
            self.writer.add_scalars('loss', {'loss': loss.item()})
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]["lr"]})

        self.writer.close()
        return

    def eval(self, data):
        #                       frames       channels(rea/imag)     y (full size)           x
        hr_tensor = np.zeros([data.shape[0], data.shape[1] // 2, data.shape[2]*self.R, data.shape[3]], dtype='complex128')

        # divide by 2 for real/imag
        num_input_channels = data.shape[1] // 2

        # iterate over the frames and produce prediction for each frame
        for ind, t in enumerate(data):
            cur_tensor = t[np.newaxis, :, :, :]
            tmp = nn.functional.interpolate(self.forward(torch.from_numpy(cur_tensor).to(self.device)), scale_factor=(1, 1/self.R), mode='bilinear')
            hr_tensor[ind, :, :, :] = tmp.detach().cpu().numpy()[:, :num_input_channels, :, :] + 1j * \
                                      tmp.detach().cpu().numpy()[:, num_input_channels:, :, :]
        return hr_tensor
