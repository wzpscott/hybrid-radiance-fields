#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, get_tensor_size, get_param_size
from utils.field_utils import GaussianField

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        # We use sigmoid activation instead of exp for scales to avoid numerical instability
        self.scaling_activation = torch.sigmoid
        self.scaling_inverse_activation = inverse_sigmoid

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.rgb_activation = torch.sigmoid

    def __init__(self, dataset):
        self.hash_size = dataset.hash_size
        self.n_levels = dataset.n_levels
        self.n_features_per_level = dataset.n_features_per_level
        self.base_resolution = dataset.base_resolution
        self.max_resolution = dataset.max_resolution

        self._xyz = torch.empty(0)
        self._scale = torch.empty(0)
        self._rgb = torch.empty(0)
        self._opacity = torch.empty(0)

        self.field = None

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.confidence_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self.point_optimizer = None
        self.field_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.setup_functions()

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scale(self):
        return self._scale
    
    @property
    def get_rgb(self):
        return self._rgb
    
    @property
    def get_opacity(self):
        return self._opacity
    
    def get_gaussians(self, camera=None, culling_mask=None):
        if culling_mask is None:
            culling_mask = torch.ones((self.get_xyz.shape[0]), dtype=bool, device="cuda")

        xyz = self.get_xyz[culling_mask].detach().clone()

        if camera is None:
            viewdir = xyz
        else:
            viewdir = (xyz - camera.camera_center.view(1, 3)).view(-1, 3)

        neural_scale, neural_rotation, neural_opacity, neural_rgb = self.field(xyz, viewdir)
        explicit_scale, explicit_opacity, explicit_rgb = self.get_scale[culling_mask], self.get_opacity[culling_mask], self.get_rgb[culling_mask]

        scale = self.scaling_activation(neural_scale) * torch.sigmoid(explicit_scale)
        rotation = self.rotation_activation(neural_rotation)
        opacity = self.opacity_activation(neural_opacity + explicit_opacity)
        rgb = self.rgb_activation(neural_rgb + explicit_rgb)
        
        outputs = {
            'scale': scale,
            'rotation': rotation,
            'opacity': opacity,
            'rgb': rgb,
        }

        return outputs
    
    def get_background_image(self, xyz, camera):
        viewdir = (xyz - camera.camera_center.view(1, 3)).view(-1, 3)
        _, _, _, background_image = self.field(xyz, viewdir)
        return background_image

    def create_from_pcd(self, pcd: BasicPointCloud, normalization, aabb):
        self.spatial_lr_scale = normalization["radius"]
        self.pts_aabb = aabb

        point_cloud = torch.as_tensor(pcd.points, device="cuda", dtype=torch.float)    
        print(f"Number of points at initialization : {point_cloud.shape[0]}, bounding box: {self.pts_aabb.cpu().numpy().round(3)[:3]}, {self.pts_aabb.cpu().numpy().round(3)[3:]}")

        self._xyz = nn.Parameter(point_cloud.requires_grad_(True))

        rgb = inverse_sigmoid(torch.as_tensor(pcd.colors, device="cuda", dtype=torch.float))
        # rgb = torch.zeros((point_cloud.shape[0], 3), device="cuda", dtype=torch.float)
        self._rgb = nn.Parameter(rgb.requires_grad_(True))

        dist2 = torch.clamp(distCUDA2(point_cloud), 0.0000001, 0.999999)
        scale = self.scaling_inverse_activation(torch.sqrt(dist2))[...,None]
        self._scale = nn.Parameter(scale.requires_grad_(True))

        opacity = self.inverse_opacity_activation(0.1 * torch.ones((point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # opacity = torch.zeros((point_cloud.shape[0], 1), device="cuda", dtype=torch.float)
        self._opacity = nn.Parameter(opacity.requires_grad_(True))

        self.field = GaussianField(
            aabb=self.pts_aabb,
            n_levels=self.n_levels,
            n_features_per_level=self.n_features_per_level,
            log2_hashmap_size=self.hash_size,
            base_resolution=self.base_resolution,
            max_resolution=self.max_resolution
        )

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def load_from_npz(self, path, normalization, aabb):
        self.spatial_lr_scale = normalization["radius"]
        self.pts_aabb = aabb

        ckpt = np.load(path)
        self._xyz = nn.Parameter(torch.from_numpy(ckpt["xyz"]).float().cuda().requires_grad_(True))
        self._scale = nn.Parameter(torch.from_numpy(ckpt["scale"]).float().cuda().requires_grad_(True))
        self._rgb = nn.Parameter(torch.from_numpy(ckpt["rgb"]).float().cuda().requires_grad_(True))
        self._opacity = nn.Parameter(torch.from_numpy(ckpt["opacity"]).float().cuda().requires_grad_(True))

        self.field = GaussianField(
            aabb=self.pts_aabb,
            n_levels=self.n_levels,
            n_features_per_level=self.n_features_per_level,
            log2_hashmap_size=self.hash_size,
            base_resolution=self.base_resolution,
            max_resolution=self.max_resolution
        )

        self.field.load_from_npz(ckpt)

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._scale], 'lr': training_args.scale_lr, "name": "scale"},
            {'params': [self._rgb], 'lr': training_args.rgb_lr, "name": "rgb"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"}
        ]

        self.point_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.field_optimizer = torch.optim.Adam(self.field.parameters(), lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )
        self.field_scheduler_args = get_expon_lr_func(
            lr_init=training_args.field_lr_init,
            lr_final=training_args.field_lr_final,
            max_steps=training_args.iterations
        )

    def step(self):
        self.field_optimizer.step()
        self.point_optimizer.step()
        self.field_optimizer.zero_grad(set_to_none=True)
        self.point_optimizer.zero_grad(set_to_none=True)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.point_optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr

        for param_group in self.field_optimizer.param_groups:
            param_group['lr'] = self.field_scheduler_args(iteration)

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.point_optimizer.param_groups:
            stored_state = self.point_optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.point_optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.point_optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._scale = optimizable_tensors["scale"]
        self._rgb = optimizable_tensors["rgb"]
        self._opacity = optimizable_tensors["opacity"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.point_optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.point_optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.point_optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.point_optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_scale, new_rgb, new_opacity):
        d = {"xyz": new_xyz, "scale": new_scale, "rgb": new_rgb, "opacity": new_opacity}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._scale = optimizable_tensors["scale"]
        self._rgb = optimizable_tensors["rgb"]
        self._opacity = optimizable_tensors["opacity"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat((self.max_radii2D, torch.zeros((new_xyz.shape[0]), device="cuda")), dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        outputs = self.get_gaussians()
        scale, rotation = outputs["scale"], outputs["rotation"]

        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scale, dim=1).values > self.percent_dense*scene_extent)

        stds = scale[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(rotation[selected_pts_mask]).repeat(N,1,1)
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scale = self.scaling_inverse_activation(scale[selected_pts_mask].repeat(N,1) / (0.8*N)).mean(dim=1, keepdim=True)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_rgb = self._rgb[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_scale, new_rgb, new_opacity)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        outputs = self.get_gaussians()
        scale = outputs["scale"]

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scale, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_scale = self._scale[selected_pts_mask]
        new_rgb = self._rgb[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]

        self.densification_postfix(new_xyz, new_scale, new_rgb, new_opacity)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        outputs = self.get_gaussians()
        scale, opacity = outputs["scale"], outputs["opacity"]

        prune_mask = (opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = scale.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        torch.cuda.empty_cache()

        if torch.isnan(self.get_xyz).any():
            print(f"{torch.isnan(self.get_xyz).sum()} NaNs in the model")

    def add_densification_stats(self, viewspace_point_tensor, update_filter, radii):
        grad = torch.norm(viewspace_point_tensor.grad[:, 2:], dim=-1)
        self.xyz_gradient_accum[update_filter] += grad[radii > 0].unsqueeze(-1)
        self.denom[update_filter] += 1

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.point_optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.point_optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.point_optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.point_optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def reset_opacity(self):
        opacity = torch.sigmoid(self._opacity)
        opacity_new = self.inverse_opacity_activation(torch.min(opacity, torch.ones_like(opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacity_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def save_as_npz(self, path, iteration):
        save_path = os.path.join(path, "ckpts", f"iteration_{iteration}.npz")
        mkdir_p(os.path.join(path, "ckpts"))

        save_dict = {}
        save_dict["xyz"] = self._xyz.detach().cpu().half().numpy()
        save_dict["scale"] = self._scale.detach().cpu().half().numpy()
        save_dict["rgb"] = self._rgb.detach().cpu().half().numpy()
        save_dict["opacity"] = self._opacity.detach().cpu().half().numpy()
        save_dict["geo_encoding"] = self.field.geo_encoding.params.detach().cpu().half().numpy()
        save_dict["rad_encoding"] = self.field.rad_encoding.params.detach().cpu().half().numpy()

        np.savez(save_path, **save_dict)

    @property
    def get_model_size(self):
        model_size = {
            "xyz": get_tensor_size(self._xyz),
            "rgb": get_tensor_size(self._rgb),
            "scale": get_tensor_size(self._scale),
            "opacity": get_tensor_size(self._opacity),
            "field": get_param_size(self.field)
        }
        model_size["total"] = sum(model_size.values())

        return model_size