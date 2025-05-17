import os
import click
import numpy as np
from shutil import copyfile
from omegaconf import OmegaConf, DictConfig
from glob import glob
import hydra

#####
import time
import wandb
import subprocess
#####
#####
# wandb.init(project="mcnerf_training", config={})  # optionally fill config dict
#####


backup_file_patterns = [
    './CMakeLists.txt',
    './*.cpp', './*.h', './*.cu',
    './src/*.cpp', './src/*.h', './src/*.cu',
    './src/*/*.cpp', './src/*/*.h', './src/*/*.cu',
    './src/*/*/*.cpp', './src/*/*/*.h', './src/*/*/*.cu',
]


def make_image_list(data_path, factor):
    image_list = []
    suffix = ['*.jpg', '*.png', '*.JPG', '*.jpeg']
    if 0.999 < factor < 1.001:
        for suf in suffix:
            image_list += glob(os.path.join(data_path, 'images', suf)) +\
                          glob(os.path.join(data_path, 'images_1', suf))
    else:
        f_int = int(np.round(factor))
        for suf in suffix:
            image_list += glob(os.path.join(data_path, 'images_{}'.format(f_int), suf))

    assert len(image_list) > 0, "No image found"
    image_list.sort()

    f = open(os.path.join(data_path, 'image_list.txt'), 'w')
    for image_path in image_list:
        f.write(image_path + '\n')

    f = open(os.path.join(data_path, 'shutter_list.txt'), 'w')
    for image_path in image_list:
        f.write(image_path[:-14] + 'shutters' + image_path[-8:-4] + '.txt' + '\n')


@hydra.main(version_base=None, config_path='../confs', config_name='default')
def main(conf: DictConfig) -> None:
    if 'work_dir' in conf:
        base_dir = conf['work_dir']
    else:
        base_dir = os.getcwd()

    print('Working directory is {}'.format(base_dir))

    data_path = os.path.join(base_dir, 'data', conf['dataset_name'], conf['case_name'])
    base_exp_dir = os.path.join(base_dir, 'exp', conf['case_name'], conf['exp_name'])

    os.makedirs(base_exp_dir, exist_ok=True)

    # backup codes
    file_backup_dir = os.path.join(base_exp_dir, 'record/')
    os.makedirs(file_backup_dir, exist_ok=True)

    for file_pattern in backup_file_patterns:
        file_list = glob(os.path.join(base_dir, file_pattern))
        for file_name in file_list:
            new_file_name = file_name.replace(base_dir, file_backup_dir)
            os.makedirs(os.path.dirname(new_file_name), exist_ok=True)
            copyfile(file_name, new_file_name)

    make_image_list(data_path, conf['dataset']['factor'])

    conf = OmegaConf.to_container(conf, resolve=True)
    conf['dataset']['data_path'] = data_path
    conf['base_dir'] = base_dir
    conf['base_exp_dir'] = base_exp_dir
    print(data_path)
    OmegaConf.save(conf, os.path.join(file_backup_dir, 'runtime_config.yaml'))
    OmegaConf.save(conf, './runtime_config.yaml')
    

    for build_dir in ['build', 'cmake-build-release']:
        if os.path.exists('{}/{}/main'.format(base_dir, build_dir)):
            os.system('{}/{}/main'.format(base_dir, build_dir))
            return
    
    ###########################
    ##############################
    # for build_dir in ['build', 'cmake-build-release']:
    #     exe_path = os.path.join(base_dir, build_dir, 'main')
    #     if os.path.exists(exe_path):
    #         print(f"Running executable: {exe_path}")
    #         os.system(exe_path)  # launch binary normally

    #         log_file = os.path.join(base_exp_dir, "logs", "loss_log.txt")
    #         last_iter_logged = -1

    #         while True:
    #             if os.path.exists(log_file):
    #                 with open(log_file, "r") as f:
    #                     lines = f.readlines()
    #                     if lines:
    #                         last_line = lines[-1]
    #                         tokens = last_line.strip().split()
    #                         if len(tokens) >= 8:
    #                             iter_num = int(tokens[0])
    #                             if iter_num > last_iter_logged:
    #                                 last_iter_logged = iter_num
    #                                 wandb.log({
    #                                     'iter': iter_num,
    #                                     'loss': float(tokens[1]),
    #                                     'color_loss': float(tokens[2]),
    #                                     'tv_loss': float(tokens[3]),
    #                                     'disp_loss': float(tokens[4]),
    #                                     'var_loss': float(tokens[5]),
    #                                     'mse': float(tokens[6]),
    #                                     'psnr': float(tokens[7]),
    #                                 })
    #             time.sleep(1)
    #         return
    #########################################
    ############################################

    assert False, 'Can not find executable file'


if __name__ == '__main__':
    main()
    