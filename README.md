<!-- # HyRF: Hybrid Radiance Fields for Efficient and High-quality Novel View Synthesis
This the offical code base for "HyRF: Hybrid Radiance Fields for Efficient and High-quality Novel View Synthesis".  -->

<p align="center">

  <h1 align="center">HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=3w7X6NYAAAAJ">Zipeng Wang</a>
    ·
    <a href="https://www.danxurgb.net/">Dan Xu</a>
  </p>
  <h3 align="center">NeurIPS 2025</h3>

  <h3 align="center"><a href="https://arxiv.org/pdf/2509.17083">Paper</a> | <a href="https://arxiv.org/abs/2509.17083">arXiv</a> | <a href="https://wzpscott.github.io/hyrf/">Project Page</a>  | <a href="https://huggingface.co/papers/2509.17083">HuggingFace</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
TLDR: Radiance fields with SOTA quality, NeRF size and 3DGS speed.
</p>
<br>


# Method Overview
<p align="center">
  <a href="">
    <img src="./assets/framework.png"Logo" width="95%">
  </a>
</p>
Our method represents the scene using grid-based neural fields and a set of compact explicit Gaussians storing only 3D position, 3D diffuse color, isotropic scale, and opacity. We encode the point position into a high-dimensional feature using the neural field and decode it into Gaussian properties with tiny MLP. These Gaussian properties are then aggregated with the explicit Gaussians and integrated into the 3DGS rasterizer.

# Installation

1. Clone the repository and create an anaconda environment.
```
git clone https://github.com/wzpscott/hybrid-radiance-fields.git
cd hybrid-radiance-fields

conda create -y -n hyrf python=3.10
conda activate hyrf
```

2. Install pytorch. Choose the version that matches your GPU.
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

3. Install other dependencies.
```
pip install -r requirements.txt
```

3. Install tiny-cuda-nn. Please refer to the [official installation guide](https://github.com/NVlabs/tiny-cuda-nn?tab=readme-ov-file#pytorch-extension) for more details.
```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

5. Install submodules.
```
pip install submodules/diff-gaussian-rasterization-accum
pip install submodules/simple-knn/
pip install submodules/fused-ssim/
```

# Dataset
- Create a ```data/``` folder inside the project path.
- Download public datasets and uncompress them into the ```data/``` folder.
  - **MipNeRF360** dataset is provided by the paper author [here](https://jonbarron.info/mipnerf360/). We test on its entire 9 scenes ```bicycle, bonsai, counter, garden, kitchen, room, stump, flowers, treehill```. 
  - **Tanks&Temples** and **Deep Blending** datasets are hosted by the 3DGS project [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).
  - For **custom scenes**, prepare the dataset using COLMAP following instructions from [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes).
- Organize the dataset into the following structure:
  ```
  data/
   ├── m360/
   │   ├── bicycle/
   │   ├── bonsai/
   │   ├── ...
   ├── tnt/
   │   ├── truck/
   │   ├── train/
   ├── db/
   │   ├── drjohnson/
   │   ├── playroom/
   └── other_custom_scenes/
      ├── scene1/
      ├── scene2/
      ├── ...
  ```

# Training and Evaluation
You can train and evaluate the model using similar commands as in the 3DGS project:
```
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

Moreover, we provide a script `train_standard.py` to train and evaluate models from mipnerf360, tanks&temples, and deep blending datasets:
```
python train_standard.py -s <scene name> -n <experiment name> -d <device id>
```


# Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). Please follow the license of 3DGS. We thank all the authors for their great work and repos. 

# Citation
If you find our code or paper useful, please cite
```bibtex
@article{wang2025hyrf,
  title={HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis},
  author={Zipeng Wang and Dan Xu},
  journal={The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```