import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim.optimizer
from torch.optim import lr_scheduler
import math
import time
from torch.utils.tensorboard import SummaryWriter


class Network:  # The base network
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

    def build_network(self):  # BASE version. Other modes override this function
        net = nn.Sequential(
            nn.Conv2d(in_channels=2*self.channels_in, out_channels=32, kernel_size=[5, 2], padding=[3, 1], padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=[1, 1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=self.R-1, kernel_size=[3, 2], padding=[1, 1], padding_mode='reflect'),
        ).to(self.device)
        return net

    def define_loss(self):
        return torch.nn.MSELoss(reduction='sum')

    def define_opt(self):
        # TODO: read the lr and momentum from config
        learning_rate = self.config['network']['optimization']['params']['lr']
        return torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def define_lr_sched(self):
        gamma = self.config['network']['lr_sched']['params']['gamma']
        milestones = self.config['network']['lr_sched']['params']['milestones']
        step_size = self.config['network']['lr_sched']['params']['step_size']
        epochs = self.config['network']['num_epochs']
        if self.config['network']['lr_sched']['name']=='MultiStepLR':
            return lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        elif self.config['network']['lr_sched']['name']=='StepLR':
            return lr_scheduler.StepLR(self.optimizer, step_size=int(epochs*step_size), gamma=gamma)
        else:
            print('****************** NO LR_SCHED DEFINED SETTING DEFAULT *****************************')
            return lr_scheduler.StepLR(self.optimizer, step_size=epochs//10, gamma=1/1.5)

    def forward(self, input_tensor):  # BASE version. Other modes override this function
        return self.net(input_tensor)

    def calc_loss(self, output, hr_gt_torch):
        return self.loss_fn(output, hr_gt_torch).cuda()

    def train(self, data_loader_object):
        # epochs
        epochs = self.config['lr_sched']['params']['epochs']
        for e in range(epochs):
            t = time.time()
            self.optimizer.zero_grad()
            if e % self.config['save_every'] == self.config['save_every'] - 1:
                print(f'saved model at epoch {e}')
                self.save_model(epoch=e, overwrite=False)

            # iterations per epochs
            it = 0
            for (hr_gt, lr) in data_loader_object:
                print(f'iter: {it}')

                hr_prediction = self.forward(lr.to(self.device))

                loss = self.calc_loss(hr_prediction.to(self.device), hr_gt.to(self.device))

                loss.backward()
                it += 1
            print(f'epoch:{e}, loss:{loss.item():.2f}. Time: {(time.time()-t):.2f}, lr={self.optimizer.param_groups[0]["lr"]}')
            # TODO: check about updating after ALL iterations in epoch
            self.optimizer.step()
            self.scheduler.step(epoch=e)
            self.writer.add_scalars('loss', {'loss': loss.item()})
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]["lr"]})


            # TODO: register training loss

            # TODO: print results (all the frames of the upscaled video) every N epochs

        self.writer.close()
        return

    def eval(self, data):

        return


    def eval_forward_crop(self, crop):
        return None


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
                    #'lr_sched': self.scheduler.state_dict()},
                   os.path.join(self.config['working_dir'], filename))

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['sd'])
        self.optimizer.load_state_dict(checkpoint['opt'])
        #self.scheduler.load_state_dict(checkpoint['lr_sched'])

