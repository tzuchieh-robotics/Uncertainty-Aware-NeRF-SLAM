import torch
import numpy as np
import argparse
import os
from src import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract trajectory from SNI-SLAM ckpt.')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--output', type=str, help='Output folder path')
    args = parser.parse_args()

    # 載入設定
    cfg = config.load_config(args.config, 'configs/SNI-SLAM.yaml')
    scale = cfg['scale']
    
    # 決定路徑
    output_dir = cfg['data']['output'] if args.output is None else args.output
    # 這裡直接套用你提供的硬編碼路徑
    ckptsdir = f"/home/tzuchieh/SNI-SLAM/output2/Replica/room1/test/ckpts"
    
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print(f'Loading checkpoint: {ckpt_path}')
            
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            
            # 提取並還原尺度
            estimate_c2w_list = ckpt['estimate_c2w_list']
            gt_c2w_list = ckpt['gt_c2w_list']
            estimate_c2w_list[:, :3, 3] /= scale
            gt_c2w_list[:, :3, 3] /= scale
            
            est_np = estimate_c2w_list.cpu().numpy()
            gt_np = gt_c2w_list.cpu().numpy()

            # 儲存到同一個檔案 (trajectories.npz)
            save_path = os.path.abspath(os.path.join(output_dir, 'trajectories.npz'))
            np.savez(save_path, est=est_np, gt=gt_np)
            
            print("\n" + "="*50)
            print(f"軌跡已成功合併儲存！")
            print(f"完整路徑: {save_path}")
            print(f"讀取方法: data = np.load('{save_path}'); est = data['est']; gt = data['gt']")
            print("="*50)
        else:
            print("錯誤：找不到任何 .tar 檔案")
    else:
        print(f"錯誤：路徑 {ckptsdir} 不存在")