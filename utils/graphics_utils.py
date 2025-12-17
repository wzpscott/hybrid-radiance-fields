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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getView2World(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center

    return np.float32(C2W)

@torch.no_grad()
def get_culling_mask(xyz, proj_mat, tol=0.1):
    xyz_proj = geom_transform_points(xyz, proj_mat)
    culling_mask = (xyz_proj[:, 0] > -1 - tol) & (xyz_proj[:, 0] < 1 + tol) & (xyz_proj[:, 1] > -1 - tol) & (xyz_proj[:, 1] < 1 + tol)
    return culling_mask

@torch.no_grad()
def get_intersect_with_sphere(rays_o, rays_d, radius=100):
    # Compute coefficients for the quadratic equation: At^2 + Bt + C = 0
    A = torch.sum(rays_d * rays_d, dim=-1)  # A = d · d
    B = 2 * torch.sum(rays_o * rays_d, dim=-1)  # B = 2(o · d)
    C = torch.sum(rays_o * rays_o, dim=-1) - radius**2  # C = o · o - r^2

    # Discriminant
    discriminant = B**2 - 4 * A * C

    # Since rays_o are inside the sphere, discriminant >= 0
    sqrt_discriminant = torch.sqrt(discriminant)

    # Compute the two solutions for t
    t1 = (-B - sqrt_discriminant) / (2 * A)
    t2 = (-B + sqrt_discriminant) / (2 * A)

    # Choose the positive t (since rays_o are inside the sphere, one t will be positive and one negative)
    t = torch.where(t1 >= 0, t1, t2)

    # Compute intersection points
    intersections = rays_o + t.unsqueeze(-1) * rays_d

    return intersections