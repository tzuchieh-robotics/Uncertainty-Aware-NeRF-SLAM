# This file is a part of SNI-SLAM
#
import argparse

from src import config
from src.SNI_SLAM import SNI_SLAM

import sys
sys.path.insert(0, '/home/tzuchieh/SNI-SLAM/seg/facebookresearch_dinov2_main/facebookresearch_dinov2_main')

def main():
    parser = argparse.ArgumentParser(
        description='Arguments for running SNI_SLAM.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU id to use, e.g. 0, 1, 2')
    args = parser.parse_args()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = config.load_config(args.config, 'configs/SNI-SLAM.yaml')
    sni_slam = SNI_SLAM(cfg, args)

    sni_slam.run()

if __name__ == '__main__':
    main()
