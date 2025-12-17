import collections
import itertools
from typing import Callable, List, Union
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn

POS_DIM = 3
DIR_DIM = 3
OPACITY_DIM = 1
SCALE_DIM = 3
ROTATION_DIM = 4
RGB_DIM = 3
CONFIDENCE_DIM = 1

class Gaussians(NamedTuple):
	xyz: torch.Tensor
	opacity: torch.Tensor
	scale: torch.Tensor
	rotation: torch.Tensor
	rgb: torch.Tensor
	confidence: torch.Tensor
	scale_reg: torch.Tensor

def get_per_level_scale(base_resolution, max_resolution, n_levels):
	if max_resolution == base_resolution:
		per_level_scale = 1.0
	else:	
		per_level_scale = np.exp2(
			np.log2(max_resolution / base_resolution) / (int(n_levels) - 1))
	return per_level_scale


class GaussianField(nn.Module):
	def __init__(self, aabb, n_levels, n_features_per_level, log2_hashmap_size, base_resolution, max_resolution, n_hidden_layers=2, n_neurons=64):
		super().__init__()

		self.aabb = aabb
		
		self.geo_encoding = get_mlp_with_gridencoding(
			SCALE_DIM + ROTATION_DIM + OPACITY_DIM,
			n_levels=n_levels,
			n_features_per_level=n_features_per_level,
			log2_hashmap_size=log2_hashmap_size - 1,
			base_resolution=base_resolution,
			max_resolution=max_resolution,
			n_hidden_layers=n_hidden_layers,
			n_neurons=n_neurons,
			encode_dir=False,
		)

		self.rad_encoding = get_mlp_with_gridencoding(
			RGB_DIM,
			n_levels=n_levels,
			n_features_per_level=n_features_per_level,
			log2_hashmap_size=log2_hashmap_size,
			base_resolution=base_resolution,
			max_resolution=max_resolution,
			n_hidden_layers=n_hidden_layers,
			n_neurons=n_neurons,
			encode_dir=True,
		)

	def forward(self, xyz, viewdir):
		xyz = contract_to_unisphere(xyz, self.aabb)
		viewdir = viewdir / torch.linalg.norm(viewdir, dim=1, keepdim=True)
		viewdir = (viewdir + 1) / 2

		geo_feature = self.geo_encoding(xyz).float()
		scale, rotation, opacity = torch.split(geo_feature, [SCALE_DIM, ROTATION_DIM, OPACITY_DIM], dim=1)

		rgb = self.rad_encoding(torch.cat([xyz, viewdir], dim=1)).float()

		return scale, rotation, opacity, rgb
	
	def load_from_npz(self, ckpt):
		geo_encoding_params = ckpt["geo_encoding"]
		rad_encoding_params = ckpt["rad_encoding"]

		self.geo_encoding.params.data = torch.from_numpy(geo_encoding_params).cuda()
		self.rad_encoding.params.data = torch.from_numpy(rad_encoding_params).cuda()

# class GaussianField(nn.Module):
# 	def __init__(self, aabb, n_levels, n_features_per_level, log2_hashmap_size, base_resolution, max_resolution, n_hidden_layers=2, n_neurons=64):
# 		super().__init__()

# 		self.aabb = aabb
		
# 		self.geo_encoding = get_mlp_with_gridencoding(
# 			SCALE_DIM + ROTATION_DIM + OPACITY_DIM,
# 			n_levels=n_levels,
# 			n_features_per_level=n_features_per_level,
# 			log2_hashmap_size=log2_hashmap_size - 1,
# 			base_resolution=base_resolution,
# 			max_resolution=max_resolution,
# 			n_hidden_layers=n_hidden_layers,
# 			n_neurons=n_neurons,
# 			encode_dir=False,
# 		)

# 		self.rad_encoding = get_mlp_with_gridencoding(
# 			RGB_DIM + SCALE_DIM + ROTATION_DIM + OPACITY_DIM,
# 			n_levels=n_levels,
# 			n_features_per_level=n_features_per_level,
# 			log2_hashmap_size=log2_hashmap_size + 1,
# 			base_resolution=base_resolution,
# 			max_resolution=max_resolution,
# 			n_hidden_layers=n_hidden_layers,
# 			n_neurons=n_neurons,
# 			encode_dir=True,
# 		)

# 	def forward(self, xyz, viewdir):
# 		xyz = contract_to_unisphere(xyz, self.aabb)
# 		viewdir = viewdir / torch.linalg.norm(viewdir, dim=1, keepdim=True)
# 		viewdir = (viewdir + 1) / 2

# 		# geo_feature = self.geo_encoding(xyz).float()
# 		# scale, rotation, opacity = torch.split(geo_feature, [SCALE_DIM, ROTATION_DIM, OPACITY_DIM], dim=1)

# 		# rgb = self.rad_encoding(torch.cat([xyz, viewdir], dim=1)).float()

# 		scale, rotation, opacity, rgb = torch.split(self.rad_encoding(torch.cat([xyz, viewdir], dim=1)).float(), [SCALE_DIM, ROTATION_DIM, OPACITY_DIM, RGB_DIM], dim=1)

# 		return scale, rotation, opacity, rgb
	
# 	def load_from_npz(self, ckpt):
# 		geo_encoding_params = ckpt["geo_encoding"]
# 		rad_encoding_params = ckpt["rad_encoding"]

# 		self.geo_encoding.params.data = torch.from_numpy(geo_encoding_params).cuda()
# 		self.rad_encoding.params.data = torch.from_numpy(rad_encoding_params).cuda()


def get_mlp_with_gridencoding(out_dim, n_levels, n_features_per_level, log2_hashmap_size, base_resolution, max_resolution, n_hidden_layers, n_neurons, output_activation="None", encode_dir=False) -> tcnn.Encoding:
	per_level_scale = get_per_level_scale(base_resolution, max_resolution, n_levels)
	
	encoding_config = {
		"otype": "HashGrid",
		"n_levels": n_levels,
		"n_features_per_level": n_features_per_level,
		"log2_hashmap_size": log2_hashmap_size,
		"base_resolution": base_resolution,
		"per_level_scale": per_level_scale,
	}
	network_config = {
		"otype": "FullyFusedMLP",
		"activation": "ReLU",
		"output_activation": output_activation,
		"n_neurons": n_neurons,
		"n_hidden_layers": n_hidden_layers,
	}
	
	if not encode_dir:
		return tcnn.NetworkWithInputEncoding(
			n_input_dims = POS_DIM,
			n_output_dims = out_dim,
			encoding_config = encoding_config,
			network_config = network_config,
		)
	else:
		return tcnn.NetworkWithInputEncoding(
			n_input_dims = POS_DIM + DIR_DIM,
			n_output_dims = out_dim,
			encoding_config = {
				"otype": "Composite",
				"nested": [
					{
						"n_dims_to_encode": POS_DIM,
						**encoding_config
					},
					{
						"n_dims_to_encode": DIR_DIM,
						"otype": "SphericalHarmonics",
						"degree": 3,
					},
				]
			},
			network_config = network_config
		)
	
def get_mlp_with_triplaneencoding(out_dim, n_features_per_level, log2_hashmap_size, resolution, n_hidden_layers, n_neurons, output_activation="None") -> tcnn.Encoding:
	plane_config = {
		"n_dims_to_encode": 2,
		"otype": "HashGrid",
		"log2_hashmap_size": log2_hashmap_size,
		"n_levels": 1,
		"n_features_per_level": n_features_per_level,
		"base_resolution": resolution,
		"per_level_scale": 1,
	}

	return tcnn.NetworkWithInputEncoding(
		n_input_dims=POS_DIM * 2,
		n_output_dims=out_dim,
		encoding_config = {
			"otype": "Composite",
			"nested": [plane_config, plane_config, plane_config],
		},
		network_config = {
			"otype": "FullyFusedMLP",
			"activation": "ReLU",
			"output_activation": output_activation,
			"n_neurons": n_neurons,
			"n_hidden_layers": n_hidden_layers,
		},
	)

def contract_to_unisphere(
	x: torch.Tensor,
	aabb: torch.Tensor,
	ord: Union[str, int] = 2,
	#  ord: Union[float, int] = float("inf"),
	eps: float = 1e-6,
	derivative: bool = False,
):
	aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
	x = (x - aabb_min) / (aabb_max - aabb_min)
	x = x * 2 - 1  # aabb is at [-1, 1]
	mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
	mask = mag.squeeze(-1) > 1

	if derivative:
		dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
			1 / mag**3 - (2 * mag - 1) / mag**4
		)
		dev[~mask] = 1.0
		dev = torch.clamp(dev, min=eps)
		return dev
	else:
		x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
		x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
		return x