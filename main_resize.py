# -*- coding: utf-8 -*-

import os.path
import logging
import cv2
import numpy as np
from collections import OrderedDict
from scipy.io import loadmat
#import hdf5storage
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_deblur
from utils import utils_sisr as sr



def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    model_name = 'usrnet'      # 'usrgan' | 'usrnet' | 'usrgan_tiny' | 'usrnet_tiny'
    testset_name = 'HR'     # test set,  'set5' | 'srbsd68'
    need_degradation = True    # default: True
    sf = 2                    # scale factor, only from {2, 3, 4}
    show_img = False           # default: False
    save_L = True              # save LR image


    task_current = 'sr'       # fixed, 'sr' for super-resolution
    n_channels = 3            # fixed, 3 for color image
    model_pool = 'model_zoo'  # fixed
    testsets = 'dataset/benchmark/face_test'     # fixed
    results = 'dataset/benchmark/face_test/LR_bicubic'       # fixed

    result_name = testset_name + '_' + model_name + '_bicubic'


    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name) # L_path, fixed, for Low-quality images
    H_path = L_path                               # H_path, 'None' | L_path, for High-quality images
    E_path = os.path.join(results, 'x'+str(sf))   # E_path, fixed, for Estimated images
    util.mkdir(E_path)

    if H_path == L_path:
        need_degradation = True
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)

        # degradation process, bicubic downsampling
        if need_degradation:
            img_L = util.modcrop(img_L, sf)
            img_L = util.imresize_np(img_L, 1/sf)

            # img_L = util.uint2single(util.single2uint(img_L))
            # np.random.seed(seed=0)  # for reproducibility
            # img_L += np.random.normal(0, noise_level_img/255., img_L.shape)

        w, h = img_L.shape[:2]

        if save_L:
            util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name+'x'+str(sf)+'.png'))

        img = cv2.resize(img_L, (sf*h, sf*w), interpolation=cv2.INTER_NEAREST)
        img = utils_deblur.wrap_boundary_liu(img, [int(np.ceil(sf*w/8+2)*8), int(np.ceil(sf*h/8+2)*8)])
        img_wrap = sr.downsample_np(img, sf, center=False)
        img_wrap[:w, :h, :] = img_L
        img_L = img_wrap

        util.imshow(util.single2uint(img_L), title='LR image with noise level {}'.format(noise_level_img)) if show_img else None

        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)


if __name__ == '__main__':

    main()
