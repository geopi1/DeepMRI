import matplotlib.pyplot as plt
import torch
from torch.utils import data
import Network
import data_manager
import utils
from argparse import ArgumentParser
import os

# ========================================================================= #
"""
Initialize base parameters
"""
parser = ArgumentParser()
parser.add_argument('-data', '--data_path', type=str, help='path to data folder', default=None, required=False)
args = parser.parse_args()
# path of the config file. original file is provided with the code
json_path = r'./config.json'

# get available device - if GPU will auto detect and use, otherwise will use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Your current Device is: ', torch.cuda.get_device_name(0))

# read json config file
config = utils.startup(json_path=json_path, copy_files=True)

if not os.path.isdir(config['data']['data_folder']) and not(args is None):
    config['data']['data_folder'] = args.data_path


# define the dataset parameters for the torch loader
params = {'batch_size': config['network']["batch_size"],
          'shuffle': True,
          'num_workers': 0}
# ========================================================================= #

# build the network object
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
