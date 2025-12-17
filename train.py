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

import os
import time
import numpy as np
from lpipsPyTorch import lpips
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_bkgd = dataset.render_background and (iteration > opt.densify_from_iter)

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_bkgd=render_bkgd)
        image, viewspace_point_tensor, visibility_filter, radii, gaussian_attributes = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["gaussian_attributes"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
            
        Lssim = 1.0 - ssim_value
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
        loss_dict = {'Ll1': Ll1, 'Lssim': Lssim, 'loss': loss}

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, gaussians, iteration, loss_dict, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., render_bkgd))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                gaussians.save_as_npz(dataset.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                if iteration > opt.densify_from_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[radii > 0])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, radii)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > 15_000 else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.normalization["radius"], size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.step()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, gaussians, iteration, loss_dict, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer and iteration % 100 == 0:
        for loss_name, loss in loss_dict.items():
            tb_writer.add_scalar('train/0-' + loss_name, loss, iteration)
        tb_writer.add_scalar('train/1-training_fps', 1000 / elapsed, iteration)
        tb_writer.add_scalar('train/2-num_points', gaussians.get_xyz.shape[0], iteration)
        # tb_writer.add_scalar('train/3-model_size', gaussians.get_model_size["total"], iteration)
        model_size = gaussians.get_model_size
        for key, value in model_size.items():
            tb_writer.add_scalar(f'model_size/{key}', value, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        test_cameras = scene.getTestCameras()
        psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
        for idx, viewpoint in enumerate(test_cameras):
            render_pkg = renderFunc(viewpoint, gaussians, *renderArgs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            if tb_writer:
                if iteration == testing_iterations[0]:
                    tb_writer.add_images(
                        f"{idx:03d}/0-{viewpoint.image_name}_gt",
                        gt_image[None],
                        global_step=iteration
                    )
                tb_writer.add_images(
                    f"{idx:03d}/1-{viewpoint.image_name}_render",
                    image[None],
                    global_step=iteration
                )
                tb_writer.add_images(
                    f"{idx:03d}/2-{viewpoint.image_name}_depth",
                    render_pkg["depth"][None],
                    global_step=iteration
                )
                
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += lpips(image, gt_image, net_type="vgg").mean().double()

        psnr_test = (psnr_test / len(test_cameras)).item()
        ssim_test = (ssim_test / len(test_cameras)).item()
        lpips_test = (lpips_test / len(test_cameras)).item()

        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            for idx, viewpoint in enumerate(test_cameras):
                image = torch.clamp(renderFunc(viewpoint, gaussians, *renderArgs)["render"], 0.0, 1.0)
        torch.cuda.synchronize()
        fps_test = (10 * len(test_cameras) / (time.perf_counter() - t))

        model_size = gaussians.get_model_size

        tqdm.write(f"[ITER {iteration}] PSNR {psnr_test:.4f} SSIM {ssim_test:.4f} LPIPS {lpips_test:.4f} SIZE {model_size['total']:.2f}MB FPS {fps_test:.2f}")

        if tb_writer:
            tb_writer.add_scalar('test/0-psnr', psnr_test, iteration)
            tb_writer.add_scalar('test/1-ssim', ssim_test, iteration)
            tb_writer.add_scalar('test/2-lpips', lpips_test, iteration)
            tb_writer.add_scalar('test/3-model_size', model_size["total"], iteration)
            tb_writer.add_scalar('test/4-fps', fps_test, iteration)
            tb_writer.add_scalar('test/5-num_points', gaussians.get_xyz.shape[0], iteration)
   
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
