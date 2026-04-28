# This file is a part of SNI-SLAM

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import time

from src.common import (get_samples, random_select, matrix_to_cam_pose, cam_pose_to_matrix, get_rays, normalize_3d_coordinate)
from src.utils.datasets import get_dataset, SeqSampler
from src.utils.Frame_Visualizer import Frame_Visualizer
from src.tools.cull_mesh import cull_mesh
import wandb

class Mapper(object):
    def __init__(self, cfg, args, sni):
        self.cfg = cfg
        self.args = args

        self.idx = sni.idx
        self.truncation = sni.truncation
        self.bound = sni.bound
        self.logger = sni.logger
        self.mesher = sni.mesher
        self.output = sni.output
        self.verbose = sni.verbose
        self.renderer = sni.renderer
        self.mapping_idx = sni.mapping_idx
        self.mapping_cnt = sni.mapping_cnt
        self.decoders = sni.shared_decoders

        self.planes_xy = sni.shared_planes_xy
        self.planes_xz = sni.shared_planes_xz
        self.planes_yz = sni.shared_planes_yz

        self.c_planes_xy = sni.shared_c_planes_xy
        self.c_planes_xz = sni.shared_c_planes_xz
        self.c_planes_yz = sni.shared_c_planes_yz

        self.use_gt_semantic = cfg['func']['use_gt_semantic']

        self.s_planes_xy = sni.shared_s_planes_xy
        self.s_planes_xz = sni.shared_s_planes_xz
        self.s_planes_yz = sni.shared_s_planes_yz

        self.estimate_c2w_list = sni.estimate_c2w_list
        self.mapping_first_frame = sni.mapping_first_frame

        self.model_manager = sni.model_manager

        self.enable_wandb = cfg['func']['enable_wandb']
        if self.enable_wandb:
            self.wandb_run = sni.wandb_run

        self.scale = cfg['scale']
        self.device = cfg['device']
        self.keyframe_device = cfg['keyframe_device']
        self.feature_device = cfg['feature_device']

        self.eval_rec = cfg['meshing']['eval_rec']
        self.joint_opt = False
        self.joint_opt_cam_lr = cfg['mapping']['joint_opt_cam_lr']
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.every_frame = cfg['mapping']['every_frame']
        self.w_sdf_fs = cfg['mapping']['w_sdf_fs']
        self.w_sdf_center = cfg['mapping']['w_sdf_center']
        self.w_sdf_tail = cfg['mapping']['w_sdf_tail']
        self.w_depth = cfg['mapping']['w_depth']
        self.w_color = cfg['mapping']['w_color']
        self.w_feature = cfg['mapping']['w_feature']
        self.w_semantic = cfg['mapping']['w_semantic']

        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']

        self.c_dim = cfg['model']['c_dim']
        self.n_classes = cfg['model']['cnn']['n_classes']

        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, num_workers=1, pin_memory=True,
                                       prefetch_factor=2, sampler=SeqSampler(self.n_img, self.every_frame))

        self.visualizer = Frame_Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device,
                                           n_classes=cfg['model']['cnn']['n_classes'])

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = sni.H, sni.W, sni.fx, sni.fy, sni.cx, sni.cy

        # loss log
        self.loss_log = {
            'total': [], 'sdf': [], 'color': [],
            'depth': [], 'feature': [], 'semantic': []
        }

    def sdf_losses(self, sdf, z_vals, gt_depth):
        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()
        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss
        return sdf_losses

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, num_keyframes, num_samples=8, num_rays=50):
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, _, _, _, _ = get_samples(
            0, H, 0, W, num_rays, H, W, fx, fy, cx, cy,
            c2w.unsqueeze(0), gt_depth.unsqueeze(0), gt_color.unsqueeze(0), device=device, dim=self.c_dim)

        gt_depth = gt_depth.reshape(-1, 1)
        nonzero_depth = gt_depth[:, 0] > 0
        rays_o = rays_o[nonzero_depth]
        rays_d = rays_d[nonzero_depth]
        gt_depth = gt_depth[nonzero_depth]
        gt_depth = gt_depth.repeat(1, num_samples)
        t_vals = torch.linspace(0., 1., steps=num_samples).to(device)
        near = gt_depth * 0.8
        far = gt_depth + 0.5
        z_vals = near * (1. - t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts = pts.reshape(1, -1, 3)

        keyframes_c2ws = torch.stack([self.estimate_c2w_list[idx] for idx in self.keyframe_list], dim=0)
        w2cs = torch.inverse(keyframes_c2ws[:-2])

        ones = torch.ones_like(pts[..., 0], device=device).reshape(1, -1, 1)
        homo_pts = torch.cat([pts, ones], dim=-1).reshape(1, -1, 4, 1).expand(w2cs.shape[0], -1, -1, -1)
        w2cs_exp = w2cs.unsqueeze(1).expand(-1, homo_pts.shape[1], -1, -1)
        cam_cords_homo = w2cs_exp @ homo_pts
        cam_cords = cam_cords_homo[:, :, :3]
        K = torch.tensor([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]], device=device).reshape(3, 3)
        cam_cords[:, :, 0] *= -1
        uv = K @ cam_cords
        z = uv[:, :, -1:] + 1e-5
        uv = uv[:, :, :2] / z
        edge = 20
        mask = (uv[:, :, 0] < W - edge) * (uv[:, :, 0] > edge) * \
               (uv[:, :, 1] < H - edge) * (uv[:, :, 1] > edge)
        mask = mask & (z[:, :, 0] < 0)
        mask = mask.squeeze(-1)
        percent_inside = mask.sum(dim=1) / uv.shape[1]

        selected_keyframes = torch.nonzero(percent_inside).squeeze(-1)

        if selected_keyframes.shape[0] == 0:
            return []

        kf_uncertainties = torch.tensor([
            self.keyframe_dict[i]['uncertainty']
            for i in selected_keyframes.tolist()
        ], device=device)

        kf_depth_u = torch.tensor([
            self.keyframe_dict[i].get('depth_uncertainty', 0.0)
            for i in selected_keyframes.tolist()
        ], device=device)

        kf_sem_u = torch.tensor([
            self.keyframe_dict[i].get('sem_uncertainty', 0.0)
            for i in selected_keyframes.tolist()
        ], device=device)

        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        geo_u_norm   = normalize(kf_uncertainties)
        depth_u_norm = normalize(kf_depth_u)
        sem_u_norm   = normalize(kf_sem_u)

        w_geo, w_depth, w_sem = 1.0, 1.0, 1.0
        combined_score = percent_inside[selected_keyframes] * (
            w_geo * geo_u_norm + w_depth * depth_u_norm + w_sem * sem_u_norm
        )
        probs = torch.softmax(combined_score, dim=0)

        rnd_inds = torch.multinomial(probs, min(num_keyframes, selected_keyframes.shape[0]))
        selected_keyframes = selected_keyframes[rnd_inds[:num_keyframes]]

        return list(selected_keyframes.cpu().numpy())

    def compute_uncertainty(self, gt_depth, c2w, n_rays=64):
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz,
                      self.c_planes_xy, self.c_planes_xz, self.c_planes_yz,
                      self.s_planes_xy, self.s_planes_xz, self.s_planes_yz)

        with torch.no_grad():
            rays_o, rays_d = get_rays(H, W, fx, fy, cx, cy, c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            idx = torch.randperm(rays_o.shape[0])[:n_rays]
            rays_o = rays_o[idx]
            rays_d = rays_d[idx]
            gt_depth_sample = gt_depth.reshape(-1)[idx]

            depth, color, sdf, z_vals, _, _, _ = self.renderer.render_batch_ray(
                all_planes, self.decoders,
                rays_d, rays_o,
                device, self.truncation,
                gt_depth=gt_depth_sample,
                return_emb=False
            )

            alpha = self.renderer.sdf2alpha(sdf, self.decoders.beta)
            weights = alpha * torch.cumprod(
                torch.cat([torch.ones((alpha.shape[0], 1), device=device),
                           (1. - alpha + 1e-10)], -1), -1)[:, :-1]

            uncertainty = -torch.sum(
                weights * torch.log(weights + 1e-10), dim=-1
            ).mean()

        return uncertainty.item()

    def optimize_mapping(self, iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, cur_sem_feat,
                         cur_sem_label, gt_label, keyframe_dict, keyframe_list, cur_c2w):
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz,
                      self.c_planes_xy, self.c_planes_xz, self.c_planes_yz,
                      self.s_planes_xy, self.s_planes_xz, self.s_planes_yz)
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        cfg = self.cfg
        device = self.device

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                optimize_frame = random_select(len(self.keyframe_dict) - 2, self.mapping_window_size - 1)
            elif self.keyframe_selection_method == 'overlap':
                optimize_frame = self.keyframe_selection_overlap(cur_gt_color, cur_gt_depth, cur_c2w, self.mapping_window_size - 1)

        if len(keyframe_list) > 1:
            optimize_frame = optimize_frame + [len(keyframe_list) - 1] + [len(keyframe_list) - 2]
            optimize_frame = sorted(optimize_frame)
        optimize_frame += [-1]

        pixs_per_image = self.mapping_pixels // len(optimize_frame)

        decoders_para_list = list(self.decoders.parameters())

        planes_para = []
        for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
            for i, plane in enumerate(planes):
                plane = nn.Parameter(plane)
                planes_para.append(plane)
                planes[i] = plane

        c_planes_para = []
        for c_planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]:
            for i, c_plane in enumerate(c_planes):
                c_plane = nn.Parameter(c_plane)
                c_planes_para.append(c_plane)
                c_planes[i] = c_plane

        s_planes_para = []
        for s_planes in [self.s_planes_xy, self.s_planes_xz, self.s_planes_yz]:
            for i, s_plane in enumerate(s_planes):
                s_plane = nn.Parameter(s_plane)
                s_planes_para.append(s_plane)
                s_planes[i] = s_plane

        gt_depths, gt_colors, c2ws, gt_c2ws = [], [], [], []
        for frame in optimize_frame:
            if frame != -1:
                gt_depths.append(keyframe_dict[frame]['depth'].to(device))
                gt_colors.append(keyframe_dict[frame]['color'].to(device))
                c2ws.append(keyframe_dict[frame]['est_c2w'])
                gt_c2ws.append(keyframe_dict[frame]['gt_c2w'])
            else:
                gt_depths.append(cur_gt_depth)
                gt_colors.append(cur_gt_color)
                c2ws.append(cur_c2w)
                gt_c2ws.append(gt_cur_c2w)

        gt_depths = torch.stack(gt_depths, dim=0)
        gt_colors = torch.stack(gt_colors, dim=0)
        c2ws = torch.stack(c2ws, dim=0)

        kf_sem_feats, kf_rgb_feats, kf_gt_label = [], [], []
        for frame in optimize_frame:
            if frame != -1:
                sem_feat = keyframe_dict[frame]['sem_feat'].to(device)
                rgb_feat = self.model_manager.head(sem_feat)
                kf_sem_feats.append(sem_feat.squeeze(0))
                kf_rgb_feats.append(rgb_feat.squeeze(0))
                kf_gt_label.append(keyframe_dict[frame]['gt_sem_label'].to(device))
            else:
                rgb_feat = self.model_manager.head(cur_sem_feat)
                kf_sem_feats.append(cur_sem_feat.squeeze(0))
                kf_rgb_feats.append(rgb_feat.squeeze(0))
                kf_gt_label.append(cur_sem_label)

        kf_sem_feats = torch.stack(kf_sem_feats, dim=0)
        kf_rgb_feats = torch.stack(kf_rgb_feats, dim=0)
        kf_gt_label = torch.stack(kf_gt_label, dim=0)

        if self.joint_opt:
            cam_poses = nn.Parameter(matrix_to_cam_pose(c2ws[1:]))
            model_paras = [{'params': decoders_para_list, 'lr': 0},
                           {'params': planes_para, 'lr': 0},
                           {'params': c_planes_para, 'lr': 0},
                           {'params': [cam_poses], 'lr': 0}]
        else:
            model_paras = [{'params': decoders_para_list, 'lr': 0},
                           {'params': planes_para, 'lr': 0},
                           {'params': c_planes_para, 'lr': 0}]

        model_paras.append({'params': s_planes_para, 'lr': 5e-3})
        model_paras.append({'params': self.model_manager.encoder.parameters(), 'lr': 3e-3})
        model_paras.append({'params': self.model_manager.head.parameters(), 'lr': 3e-3})
        model_paras.append({'params': self.model_manager.feat_fusion.parameters(), 'lr': cfg['mapping']['lr']['fusion_lr']})

        optimizer = torch.optim.Adam(model_paras)
        optimizer.param_groups[0]['lr'] = cfg['mapping']['lr']['decoders_lr'] * lr_factor
        optimizer.param_groups[1]['lr'] = cfg['mapping']['lr']['planes_lr'] * lr_factor
        optimizer.param_groups[2]['lr'] = cfg['mapping']['lr']['c_planes_lr'] * lr_factor

        if self.joint_opt:
            optimizer.param_groups[3]['lr'] = self.joint_opt_cam_lr

        for joint_iter in range(iters):
            if self.verbose:
                start_time = time.time()

            if not (idx == 0 and self.no_vis_on_first_frame):
                self.visualizer.save_imgs(idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, all_planes, self.decoders,
                                          gt_sem=gt_label, est_sem=cur_sem_label)

            if self.joint_opt:
                c2ws_ = torch.cat([c2ws[0:1], cam_pose_to_matrix(cam_poses)], dim=0)
            else:
                c2ws_ = c2ws

            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, batch_sem_feats, batch_rgb_feats, batch_gt_label = get_samples(
                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2ws_, gt_depths, gt_colors,
                sem_feats=kf_sem_feats, rgb_feats=kf_rgb_feats, gt_label=kf_gt_label, device=device, dim=self.c_dim)

            progress = idx / self.n_img
            depth, color, sdf, z_vals, gt_feat, plane_feat, render_semantic = self.renderer.render_batch_ray(
                all_planes, self.decoders,
                batch_rays_d, batch_rays_o, device, self.truncation,
                gt_depth=batch_gt_depth,
                sem_feats=batch_sem_feats, rgb_feats=batch_rgb_feats,
                return_emb=True)

            depth_mask = (batch_gt_depth > 0)

            sdf_loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])
            loss = sdf_loss

            color_loss = self.w_color * torch.square(batch_gt_color - color).mean()
            loss = loss + color_loss

            depth_loss = self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()
            loss = loss + depth_loss

            plane_feature = plane_feat.detach()
            gt_feature = gt_feat.detach()
            feature_loss = self.w_feature * (gt_feature - plane_feature).abs().mean()
            loss = loss + feature_loss

            CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-1)
            semantic_loss = self.w_semantic * CrossEntropyLoss(render_semantic, batch_gt_label)
            loss = loss + semantic_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

            # 記錄 loss
            self.loss_log['total'].append(loss.item())
            self.loss_log['sdf'].append(sdf_loss.item())
            self.loss_log['color'].append(color_loss.item())
            self.loss_log['depth'].append(depth_loss.item())
            self.loss_log['feature'].append(feature_loss.item())
            self.loss_log['semantic'].append(semantic_loss.item())

            if self.verbose:
                end_time = time.time()
                print(f"mapping: {(end_time - start_time)*1000} ms")

            if self.enable_wandb:
                log_dict = {
                    "total_loss": loss.item(),
                    "sdf_loss": sdf_loss.item(),
                    "color_loss": color_loss.item(),
                    "depth_loss": depth_loss.item(),
                    "feature_loss": feature_loss.item(),
                    "semantic_loss": semantic_loss.item(),
                }
                self.wandb_run.log(log_dict)

        if self.joint_opt:
            optimized_c2ws = cam_pose_to_matrix(cam_poses.detach())
            camera_tensor_id = 0
            for frame in optimize_frame[1:]:
                if frame != -1:
                    keyframe_dict[frame]['est_c2w'] = optimized_c2ws[camera_tensor_id]
                    camera_tensor_id += 1
                else:
                    cur_c2w = optimized_c2ws[-1]

        return cur_c2w

    def add_noise(self, gt_sem_label, noise_ratio):
        import random
        h, w = gt_sem_label.shape
        noise_label = gt_sem_label.clone()
        for i in range(h):
            for j in range(w):
                if random.random() < noise_ratio:
                    noise_label[i, j] = random.randint(0, self.n_classes - 1)
        return noise_label

    def _plot_uncertainty(self, uncertainty_log):
        import matplotlib.pyplot as plt
        import numpy as np

        idxs = [x[0] for x in uncertainty_log]
        uncertainties = [x[1] for x in uncertainty_log]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(idxs, uncertainties, color='steelblue', linewidth=1.5)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Uncertainty (Weight Entropy)')
        ax.set_title('Per-Frame Uncertainty')
        ax.grid(True, alpha=0.3)

        mean_u = np.mean(uncertainties)
        ax.axhline(y=mean_u, color='orange', linestyle='--', label=f'Mean: {mean_u:.4f}')
        ax.legend()

        save_path = os.path.join(self.output, 'uncertainty_plot.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'Uncertainty plot saved to {save_path}')

        np.save(os.path.join(self.output, 'uncertainty_log.npy'), np.array(uncertainty_log))

    def _plot_loss(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        keys = ['total', 'sdf', 'color', 'depth', 'feature', 'semantic']
        colors = ['black', 'red', 'blue', 'green', 'orange', 'purple']

        for i, (key, color) in enumerate(zip(keys, colors)):
            axes[i].plot(self.loss_log[key], color=color, linewidth=0.8)
            axes[i].set_title(f'{key} loss')
            axes[i].set_xlabel('iteration')
            axes[i].set_ylabel('loss')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output, 'loss_plot.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'Loss plot saved to {save_path}')

        import numpy as np
        np.save(os.path.join(self.output, 'loss_log.npy'), self.loss_log)

    def _update_keyframe_uncertainties(self):
        with torch.no_grad():
            if len(self.keyframe_dict) == 0:
                return

            all_planes = (self.planes_xy, self.planes_xz, self.planes_yz,
                          self.c_planes_xy, self.c_planes_xz, self.c_planes_yz,
                          self.s_planes_xy, self.s_planes_xz, self.s_planes_yz)
            n_rays = 32
            n_strat = self.renderer.n_stratified
            n_imp = self.renderer.n_importance
            n_samples = n_strat + n_imp
            device = self.device
            n_kf = len(self.keyframe_dict)

            all_depths = torch.stack([f['depth'].to(device) for f in self.keyframe_dict])
            all_c2ws = torch.stack([f['est_c2w'] for f in self.keyframe_dict])

            pi = torch.randint(0, self.H, (n_kf, n_rays), device=device)
            pj = torch.randint(0, self.W, (n_kf, n_rays), device=device)

            dirs = torch.stack([
                (pj - self.cx) / self.fx,
                -(pi - self.cy) / self.fy,
                -torch.ones(n_kf, n_rays, device=device)
            ], dim=-1)

            rays_d = torch.sum(dirs.unsqueeze(-2) * all_c2ws[:, None, :3, :3], dim=-1)
            rays_o = all_c2ws[:, None, :3, 3].expand(-1, n_rays, -1)

            gt_depth_sample = all_depths[
                torch.arange(n_kf, device=device).unsqueeze(1), pi, pj
            ].unsqueeze(-1)

            t_strat = torch.linspace(0., 1., steps=n_strat, device=device)
            t_imp   = torch.linspace(0., 1., steps=n_imp,   device=device)

            z_vals_free    = 1.2 * gt_depth_sample * t_strat
            z_vals_surface = gt_depth_sample - (1.5 * self.truncation) + \
                             (3 * self.truncation * t_imp)

            z_vals, _ = torch.sort(
                torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1
            )

            rays_o_flat  = rays_o.reshape(n_kf * n_rays, 3)
            rays_d_flat  = rays_d.reshape(n_kf * n_rays, 3)
            z_vals_flat  = z_vals.reshape(n_kf * n_rays, n_samples)
            gt_depth_flat = gt_depth_sample.reshape(n_kf * n_rays)

            # render_batch_ray 取得 depth 和 semantic
            rendered_depth, _, sdf, _, _, _, rendered_semantic = \
                self.renderer.render_batch_ray(
                    all_planes, self.decoders,
                    rays_d_flat, rays_o_flat,
                    device, self.truncation,
                    gt_depth=gt_depth_flat,
                    return_emb=False
                )

            # geometry uncertainty
            alpha = self.renderer.sdf2alpha(sdf, self.decoders.beta)
            weights = alpha * torch.cumprod(
                torch.cat([torch.ones((alpha.shape[0], 1), device=device),
                           (1. - alpha + 1e-10)], -1), -1)[:, :-1]

            geo_u = (-torch.sum(weights * torch.log(weights + 1e-10), dim=-1)
                      .reshape(n_kf, n_rays).mean(dim=-1))

            # depth uncertainty
            valid = gt_depth_flat > 0
            depth_error = torch.zeros(n_kf * n_rays, device=device)
            depth_error[valid] = torch.abs(rendered_depth[valid] - gt_depth_flat[valid])
            depth_u = depth_error.reshape(n_kf, n_rays).mean(dim=-1)

            # semantic uncertainty
            sem_probs = torch.softmax(rendered_semantic, dim=-1)
            sem_u = (-torch.sum(sem_probs * torch.log(sem_probs + 1e-10), dim=-1)
                      .reshape(n_kf, n_rays).mean(dim=-1))

            for i, f in enumerate(self.keyframe_dict):
                f['uncertainty']       = geo_u[i].item()
                f['depth_uncertainty'] = depth_u[i].item()
                f['sem_uncertainty']   = sem_u[i].item()

    def run(self):
        cfg = self.cfg

        uncertainty_log = []
        update_freq = 10

        all_planes = (
            self.planes_xy, self.planes_xz, self.planes_yz,
            self.c_planes_xy, self.c_planes_xz, self.c_planes_yz,
            self.s_planes_xy, self.s_planes_xz, self.s_planes_yz)

        idx, gt_color, gt_depth, gt_c2w, gt_semantic = self.frame_reader[0]
        data_iterator = iter(self.frame_loader)

        self.estimate_c2w_list[0] = gt_c2w

        init_phase = True
        prev_idx = -1
        while True:
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img - 1:
                    break
                if idx % self.every_frame == 0 and idx != prev_idx:
                    break
                time.sleep(0.001)

            prev_idx = idx

            _, gt_color, gt_depth, gt_c2w, gt_semantic = next(data_iterator)
            gt_color    = gt_color.squeeze(0).to(self.device, non_blocking=True)
            gt_depth    = gt_depth.squeeze(0).to(self.device, non_blocking=True)
            gt_c2w      = gt_c2w.squeeze(0).to(self.device, non_blocking=True)
            gt_semantic = gt_semantic.squeeze(0).to(self.device)

            cur_c2w = self.estimate_c2w_list[idx]

            uncertainty = self.compute_uncertainty(gt_depth, cur_c2w, n_rays=64)
            uncertainty_log.append((idx.item(), uncertainty))
            print(f"Frame {idx}: uncertainty = {uncertainty:.4f}")

            if not init_phase:
                lr_factor = cfg['mapping']['lr_factor']
                iters = cfg['mapping']['iters']
            else:
                lr_factor = cfg['mapping']['lr_first_factor']
                iters = cfg['mapping']['iters_first']

            self.joint_opt = (len(self.keyframe_list) > 4) and cfg['mapping']['joint_opt']

            with torch.no_grad():
                frame_rgb = gt_color.permute(2, 0, 1).unsqueeze(0).to(self.device)
                self.model_manager.set_mode_feature()
                sem_feat = self.model_manager.cnn(frame_rgb)

                if self.use_gt_semantic:
                    gt_sem_label = gt_semantic
                else:
                    self.model_manager.set_mode_result()
                    gt_sem_label = self.model_manager.cnn(frame_rgb)

            cur_c2w = self.optimize_mapping(iters, lr_factor, idx, gt_color, gt_depth, gt_c2w, sem_feat,
                                            gt_sem_label, gt_semantic,
                                            self.keyframe_dict, self.keyframe_list, cur_c2w)

            if self.joint_opt:
                self.estimate_c2w_list[idx] = cur_c2w

            if idx % self.keyframe_every == 0:
                self.keyframe_list.append(idx)

                kf_uncertainty = self.compute_uncertainty(gt_depth, cur_c2w, n_rays=64)

                frame_dict = {
                    'gt_c2w': gt_c2w,
                    'idx': idx,
                    'color': gt_color.to(self.keyframe_device),
                    'depth': gt_depth.to(self.keyframe_device),
                    'est_c2w': cur_c2w.clone(),
                    'uncertainty': kf_uncertainty,
                    'depth_uncertainty': 0.0,
                    'sem_uncertainty': 0.0,
                }
                frame_dict['sem_feat'] = sem_feat.to(self.feature_device)
                frame_dict['gt_sem_label'] = gt_sem_label.to(self.feature_device)
                self.keyframe_dict.append(frame_dict)

            if idx % update_freq == 0:
                self._update_keyframe_uncertainties()

            init_phase = False
            self.mapping_first_frame[0] = 1

            if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) or idx == self.n_img - 1:
                self.logger.log(idx, self.keyframe_list)

            self.mapping_idx[0] = idx
            self.mapping_cnt[0] += 1

            if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
                mesh_out_semantic = f'{self.output}/mesh/{idx:05d}_mesh_sem.ply'
                mesh_out_color = f'{self.output}/mesh/{idx:05d}_mesh_rgb.ply'
                self.mesher.get_mesh(mesh_out_color, all_planes, self.decoders, self.keyframe_dict,
                                     self.device, mesh_out_semantic=mesh_out_semantic, color=False)

            if idx == self.n_img - 1:
                self._plot_uncertainty(uncertainty_log)
                self._plot_loss()

                mesh_out_semantic = f'{self.output}/mesh/final_mesh_semantic.ply'
                mesh_out_color = f'{self.output}/mesh/final_mesh_color.ply'
                self.mesher.get_mesh(mesh_out_color, all_planes, self.decoders, self.keyframe_dict,
                                     self.device, mesh_out_semantic=mesh_out_semantic, semantic=False)
                cull_mesh(mesh_out_color, self.cfg, self.args, self.device,
                          estimate_c2w_list=self.estimate_c2w_list)
                break