import json
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.exposure import equalize_hist
from skimage.metrics import structural_similarity as ssim


class bcolors:
    """
    printing helper class
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ---------------------------------------------------------------------------------------------------
# json config functions
# ---------------------------------------------------------------------------------------------------
def read_json_with_line_comments(cjson_path):
    """
    Read the config file that is used as external configuration file
    :param cjson_path:
    :return: config dictionary
    """
    # open the file
    with open(cjson_path, 'r') as R:
        valid = []
        # real and strip comments
        for line in R.readlines():
            if line.lstrip().startswith('#') or line.lstrip().startswith('//'):
                continue
            valid.append(line)
    # load the clean json format
    return json.loads(' '.join(valid))


def startup(json_path, copy_files=True):
    """
    Setup the results folders and complete loading parameters to config dictionary
    :param json_path: the relative (or absolute) path to the configuration file
    :param copy_files: flag whether to copy or not the current running files (.py and config)
    :return: config dictionary
    """

    # read and load json
    print('-startup- reading config json {}'.format(json_path))
    config = read_json_with_line_comments(json_path)

    # copy files to result folder
    if copy_files and ("working_dir" not in config or not os.path.isdir(config['trainer']['working_dir'])):
        # find available working dir
        v = 0
        while True:
            working_dir = os.path.join(config['working_dir_base'], '{}-v{}'.format(config['tag'], v))
            if not os.path.isdir(working_dir):
                break
            v += 1
        os.makedirs(working_dir, exist_ok=False)
        config['working_dir'] = working_dir
        print('-startup- working directory is {}'.format(config['working_dir']))

    if copy_files:
        for filename in os.listdir('.'):
            if filename.endswith('.py'):
                shutil.copy(filename, config['working_dir'])
            shutil.copy(json_path, config['working_dir'])
        with open(os.path.join(config['working_dir'], 'processed_config.json'), 'w') as W:
            W.write(json.dumps(config, indent=2))
    return config


def visualize_results(dataset, net, config, net_name='RAKI'):
    """
    Print the results
    :param dataset: data class to load the GT, and subsampled data
    :param net:
    :param config:
    :param net_name:
    :return:
    """
    if net_name == 'RAKI':
        subsampled_data = dataset.subsampled_data[:, :2 * dataset.data.shape[1], :, :]
    else:
        subsampled_data = dataset.subsampled_data

    # fix the K-space by setting the true GT values of the actual recording
    interpolated_k_space = net.eval(subsampled_data)
    interpolated_k_space[:, :, 615:, :] = 0
    interpolated_k_space[:, :, ::net.R, :] = dataset.data[:, :, ::net.R, :]
    ACS = list(np.arange(-dataset.ACS_size // 2, dataset.ACS_size // 2) + 308)
    interpolated_k_space[:, :, ACS[:], :] = dataset.data[:, :, ACS[:], :]

    # make images
    eval_img = 0
    for channel in interpolated_k_space[7, :, :, :]:
        eval_img += np.abs(np.fft.fftshift(np.fft.fft2(channel))) ** 2

    # bring the image range to [0 1]
    eval_img = (eval_img - np.min(eval_img)) / (np.max(eval_img) - np.min(eval_img))

    orig_img = 0

    # print a random channel
    for channel in dataset.data[7, :, :, :]:
        orig_img += np.abs(np.fft.fftshift(np.fft.fft2(channel))) ** 2
    orig_img = (orig_img - np.min(orig_img)) / (np.max(orig_img) - np.min(orig_img))

    # plot the GT of the K-space
    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.imshow(np.log10(np.abs(dataset.data[7, 0, ::]) + 1e-10), cmap='gray')
    plt.title('Fully-Sampled Image')
    plt.tight_layout()

    # subsampled data
    subsampled_kspace = dataset.data
    for i in range(1, net.R):
        subsampled_kspace[:, :, i::net.R, :] = 0
    subsampled_kspace[:, :, ACS[:], :] = interpolated_k_space[:, :, ACS[:], :]

    subsampled_img = 0
    for channel in subsampled_kspace[7, :, :, :]:
        subsampled_img += np.abs(np.fft.fftshift(np.fft.fft2(channel))) ** 2
    subsampled_img = (subsampled_img - np.min(subsampled_img)) / (np.max(subsampled_img) - np.min(subsampled_img))

    # calculate the metrics
    MSE_subsampled = np.mean((subsampled_img - orig_img) ** 2)
    MSE_eval = np.mean((eval_img - orig_img) ** 2)
    peak_intens = np.max(orig_img) ** 2
    SSIM_subsampled = ssim(orig_img, subsampled_img)
    SSIM_eval = ssim(orig_img, eval_img)

    # print results to screen
    print('Subsampled PSNR: ', 10 * np.log10(peak_intens / MSE_subsampled), f'ssim: {SSIM_subsampled}')
    print('Subsampled NMSE: ', MSE_subsampled / np.mean(orig_img ** 2))
    print('Reconstruction PSNR: ', 10 * np.log10(peak_intens / MSE_eval), f'ssim: {SSIM_eval}')
    print('Reconstruction NMSE: ', MSE_eval / np.mean(orig_img ** 2))

    # print resutls to file
    with open(f'{config["working_dir"]}/res.txt', 'a') as f:
        f.write(f'Subsampled:\n')
        f.write(f'PSNR: {10 * np.log10(peak_intens / MSE_subsampled)}\n')
        f.write(f'SSIM: {SSIM_subsampled}\n')
        f.write(f'NMSE: {MSE_subsampled / np.mean(orig_img ** 2)}\n')
        f.write(f'\nReconstructed:\n')
        f.write(f'PSNR: {10 * np.log10(peak_intens / MSE_eval)}   ssim: {SSIM_eval}\n')
        f.write(f'SSIM: {SSIM_eval}\n')
        f.write(f'NMSE: {MSE_eval / np.mean(orig_img ** 2)}\n')

    # plot interpolation results
    plt.subplot(1, 3, 2)
    plt.imshow(np.log10(np.abs(interpolated_k_space[7, 0, ::]) + 1e-10), cmap='gray')
    plt.title('Interp. Kspace')
    plt.tight_layout()

    # plot subsampled
    plt.subplot(1, 3, 3)
    plt.imshow(np.log10(np.abs(subsampled_kspace[7, 0, ::]) + 1e-10), cmap='gray')
    plt.title('Subsampled K-space')
    plt.tight_layout()

    # save the figure
    plt.savefig(f'{config["working_dir"]}/K_space_results.png', dpi=500)

    # figure with images
    plt.figure(2)
    plt.subplot(2, 3, 1)
    plt.imshow(equalize_hist(orig_img), cmap='gray')
    plt.title('GT Image')
    # plt.tight_layout()

    plt.subplot(2, 3, 2)
    plt.imshow(equalize_hist(subsampled_img), cmap='gray')
    plt.title(f'Subsampled Image PSNR: {10 * np.log10(peak_intens / MSE_subsampled):.2f} SSIM: {SSIM_subsampled:.3f}')
    # plt.tight_layout()

    plt.subplot(2, 3, 3)
    plt.imshow(equalize_hist(eval_img), cmap='gray')
    plt.title(f'Reconstructed Image {10 * np.log10(peak_intens / MSE_eval):.2f} SSIM: {SSIM_eval:.3f}')
    # plt.tight_layout()

    plt.subplot(2, 3, 4)
    ax = plt.gca()
    im = ax.imshow(orig_img - orig_img, cmap='gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.tight_layout()
    plt.colorbar(im, cax=cax)

    plt.subplot(2, 3, 5)
    ax1 = plt.gca()
    im1 = ax1.imshow(subsampled_img - orig_img, cmap='gray')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    # plt.tight_layout()
    plt.colorbar(im1, cax=cax1)

    plt.subplot(2, 3, 6)
    ax2 = plt.gca()
    im2 = ax2.imshow(eval_img - orig_img, cmap='gray')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    # plt.tight_layout()
    plt.colorbar(im2, cax=cax2)

    fig = plt.gcf()
    fig.set_size_inches((15, 8.5), forward=False)
    plt.savefig(f'{config["working_dir"]}/Restored_images.png', dpi=500)
