import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
import Network
import data_manager
import utils
from skimage.metrics import structural_similarity as ssim
from mpl_toolkits.axes_grid1 import make_axes_locatable


json_path = r'./config.json'
# get available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Your current Device is: ', torch.cuda.get_device_name(0))
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

torch.cuda.empty_cache()

# eval
subsampled_data = dataset.subsampled_data

interpolated_k_space = net.eval(subsampled_data[:, :2*dataset.data.shape[1], :, :])
interpolated_k_space[:, :, 615:, :] = 0
interpolated_k_space[:, :, ::2, :] = dataset.data[:,:,::2,:]



eval_img = np.abs(np.fft.fftshift(np.fft.fft2(interpolated_k_space[7, 0, ::])))
orig_img = np.abs(np.fft.fftshift(np.fft.fft2(dataset.data[7, 0, ::])))

plt.figure(1)
plt.subplot(1,3,1)
plt.imshow(np.log10(np.abs(dataset.data[7, 0, ::])+1e-10), cmap='gray')
plt.title('Fully-Sampled Image')
plt.tight_layout()


subsampled_kspace = dataset.data
subsampled_kspace[:,:,1::2,:] = 0
subsampled_img = np.abs(np.fft.fftshift(np.fft.fft2(subsampled_kspace[7, 0, ::])))


eval_img = (eval_img-np.min(eval_img))/(np.max(eval_img)-np.min(eval_img))
subsampled_img = (subsampled_img-np.min(subsampled_img))/(np.max(subsampled_img)-np.min(subsampled_img))
orig_img = (orig_img-np.min(orig_img))/(np.max(orig_img)-np.min(orig_img))


plt.subplot(1,3,2)
plt.imshow(np.log10(np.abs(interpolated_k_space[7, 0, ::])+1e-10), cmap='gray')
plt.title('Interp. Kspace')
plt.tight_layout()

plt.subplot(1,3,3)
plt.imshow(np.log10(np.abs(subsampled_kspace[7, 0, ::])+1e-10), cmap='gray')
plt.title('Subsampled K-space')
plt.tight_layout()


plt.figure(2)
plt.subplot(2,3,1)
plt.imshow(orig_img, cmap='gray')
plt.title('GT Image')
plt.tight_layout()

plt.subplot(2,3,2)
plt.imshow(subsampled_img, cmap='gray')
plt.title('Subsampled Image')
plt.tight_layout()

plt.subplot(2,3,3)
plt.imshow(eval_img, cmap='gray')
plt.title('Reconstructed Image')
plt.tight_layout()

plt.subplot(2,3,4)
ax = plt.gca()
im = ax.imshow(orig_img-orig_img, cmap='gray')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.tight_layout()
plt.colorbar(im, cax=cax)

plt.subplot(2,3,5)
ax1 = plt.gca()
im1 = ax1.imshow(subsampled_img-orig_img, cmap='gray')
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
plt.tight_layout()
plt.colorbar(im1, cax=cax1)


plt.subplot(2,3,6)
ax2 = plt.gca()
im2 = ax2.imshow(eval_img-orig_img, cmap='gray')
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
plt.tight_layout()
plt.colorbar(im2, cax=cax2)



print('Subsampled PSNR: ', 10 * np.log10(np.max(orig_img) ** 2 / np.mean((subsampled_img - orig_img) ** 2)), f'ssim: {ssim(orig_img,subsampled_img)}')
print('Reconstruction PSNR: ', 10 * np.log10(np.max(orig_img) ** 2 / np.mean((eval_img - orig_img) ** 2)), f'ssim: {ssim(orig_img,eval_img)}')

plt.show()
print(123)
