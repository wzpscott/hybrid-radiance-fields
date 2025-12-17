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
from diff_gaussian_rasterization_accum import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, render_bkgd = False):
    """
    Render the scene. 
    """
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    culling_mask = rasterizer.markVisible(pc.get_xyz)
    assert culling_mask.sum() > 0, "No Gaussians are visible in the frustum. Check the camera position and FoV."

    screenspace_points = torch.zeros_like(pc.get_xyz[culling_mask], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = pc.get_xyz[culling_mask]
    means2D = screenspace_points

    gaussians = pc.get_gaussians(viewpoint_camera, culling_mask)

    scales = gaussians["scale"]
    rotations = gaussians["rotation"]
    opacity = gaussians["opacity"]
    rgb = gaussians["rgb"]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth_image, alpha_image, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = rgb,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)
    
    rendered_image = rendered_image.clamp(0, 1)

    background_image = 0
    if render_bkgd:
        bkgd_points = viewpoint_camera.get_bkgd_points()
        background_image = pc.get_background_image(bkgd_points, viewpoint_camera)
        background_image = background_image.view(viewpoint_camera.image_height, viewpoint_camera.image_width, 3).permute(2, 0, 1)
        rendered_image = rendered_image + alpha_image * background_image

    visibility_filter = torch.zeros_like(culling_mask, dtype=torch.bool, device="cuda")
    visibility_filter[culling_mask] = radii > 0
    
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "depth" : depth_image,
        "gaussian_attributes": gaussians,
        "background_image": alpha_image * background_image,
        "foreground_image": rendered_image - alpha_image * background_image,
        }
    
    return out
