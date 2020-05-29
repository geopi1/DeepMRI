import torch
import utils
import Network
import data_manager
from torch.utils import data


json_path = r'./config.json'
# get available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read json config file
config = utils.startup(json_path=json_path, copy_files=True)

# build the network object
net = Network.Network(config, device)

# define the dataset
params = {'batch_size': config['network']["batch_size"],
          'shuffle': True,
          'num_workers': 0}

dataset = data_manager.DataHandler(config)
data_generator = data.DataLoader(dataset, **params)

# train
net.train(data_generator)

# eval
net.eval(data_generator)