import matplotlib.pyplot as plt
import torch
from torch.utils import data

import Network
import data_manager
import utils

json_path = r'./config.json'
# get available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Your current Device is: ', torch.cuda.get_device_name(0))
# read json config file
config = utils.startup(json_path=json_path, copy_files=True)
# define the dataset
params = {'batch_size': config['network']["batch_size"],
          'shuffle': True,
          'num_workers': 0}

# ========================================================================= #
# # build the network object
net = Network.RAKINetwork(config, device)

# load the data
dataset = data_manager.RAKIDataHandler(config)

# instantiate a Pytorch dataloader object
data_generator = data.DataLoader(dataset, **params)

# call the training scheme
net.train(data_generator)

# empty GPU ram
torch.cuda.empty_cache()

# run evaluation on the full k-space
utils.visualize_results(dataset, net, config, net_name='RAKI')

# show all the plots
plt.show()

# clean up
torch.cuda.empty_cache()

# ========================================================================= #

# train second spatial network
# ========================================================================= #
# build the network object
# net = Network.SpatialNetwork(config, device)
#
# dataset = data_manager.SpatialDataHandler(config)
# data_generator = data.DataLoader(dataset, **params)
# # train
# net.train(data_generator)
#
# torch.cuda.empty_cache()
# # eval
# utils.visualize_results(dataset, net, config, net_name='Spatial')
# plt.show()
# ========================================================================= #

print(123)
