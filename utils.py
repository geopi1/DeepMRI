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
