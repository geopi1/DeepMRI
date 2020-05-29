import json
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import numpy as np


# ---------------------------------------------------------------------------------------------------
# json config functions
# ---------------------------------------------------------------------------------------------------
def read_json_with_line_comments(cjson_path):
    with open(cjson_path, 'r') as R:
        valid = []
        for line in R.readlines():
            if line.lstrip().startswith('#') or line.lstrip().startswith('//'):
                continue
            valid.append(line)
    return json.loads(' '.join(valid))


def startup(json_path, copy_files=True):
    print('-startup- reading config json {}'.format(json_path))
    config = read_json_with_line_comments(json_path)

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


def visualize_tuple(hr_lr_tuple, name=0, name_hr='HR', name_lr='LR', save_to_file=False, save_path='./results/imgs'):
    """
    take a tensor and its low resolution version (lr) and show them side-by-side
    :param hr_lr_tuple: (hr,lr) tuple of np arrays
    :param name: save folder name (selected randomly to allow saving seq.)
    :return: none, plots the frames or tensors
    """

    hr_tensor = hr_lr_tuple[0]
    lr_tensor = hr_lr_tuple[1]
    normalize = True
    if normalize:
        hr_tensor = hr_tensor / np.max(hr_tensor)
        lr_tensor = lr_tensor / np.max(lr_tensor)
    subsample_ratio = hr_tensor.shape[0] // lr_tensor.shape[0]

    for i in range(lr_tensor.shape[0]):
        plt.figure(1)
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(lr_tensor[i, :])
        plt.title(f'{name_lr} frame {i}')
        for j in range(subsample_ratio):
            plt.subplot(1, 2, 2)
            plt.imshow(hr_tensor[subsample_ratio * i + j, :], vmin=0.0)
            plt.title(f'{name_hr} frame {i * subsample_ratio + j}')
            plt.draw()
            plt.pause(0.05)

            if save_to_file:
                rand = np.random.randint(0, 500)
                folder_name = save_path
                os.makedirs(folder_name, exist_ok=True)
                plt.savefig(f'{folder_name}/{subsample_ratio * i + j}.png')


def save_output_result(vid_tensor, path):
    """
    take a video tensor [f,h,w,c] and save it as frames
    :param vid_tensor: video tensor [f,h,w,c] numpy ndarray
    :param path: folder to save the frames
    :return: none
    """
    for i, im in enumerate(vid_tensor):
        # frame = Image.fromarray(np.uint8(np.clip(im, 0, 1)) * 255)
        pltimg.imsave(f'{path}/{i:05d}.png', np.clip(im, 0, 1))
        # frame.save()
        # plt.imshow(im)
        # plt.show()


def visualize_tuple_tensors(hr_tensor, lr_tensor, name=0):
    """
    take a tensor and its low resolution version (lr) and show them side-by-side
    :param hr_lr_tuple: (hr,lr) tuple of np arrays
    :param name: save folder name (selected randomly to allow saving seq.)
    :return: none, plots the frames or tensors
    """
    from matplotlib import pyplot as plt
    subsample_ratio = hr_tensor.shape[1] // lr_tensor.shape[1]
    if False:
        rand = np.random.randint(0, 500)
        folder_name = f'./results/imgs_{name}_{rand}'
        os.makedirs(folder_name, exist_ok=True)

    for i in range(lr_tensor.shape[1]):
        plt.figure(1)
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(lr_tensor[:, i, :, :], (1, 2, 0)))
        # plt.imshow(lr_tensor[i, :])
        plt.title(f'LR frame {i}')
        for j in range(subsample_ratio):
            plt.subplot(1, 2, 2)
            plt.imshow(np.transpose(hr_tensor[:, subsample_ratio * i + j, :, :], (1, 2, 0)))
            # plt.imshow(hr_tensor[subsample_ratio * i + j, :])
            plt.title(f'HR frame {i * subsample_ratio + j}')
            plt.draw()
            plt.pause(1)

            if False:
                plt.savefig(f'{folder_name}/{subsample_ratio * i + j}.png')
