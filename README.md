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


# Method
<p align="center">
  <a href="">
    <img src="./assets/framework.png"Logo" width="95%">
  </a>
</p>
Our method represents the scene using grid-based neural fields and a set of compact explicit Gaussians storing only 3D position, 3D diffuse color, isotropic scale, and opacity. We encode the point position into a high-dimensional feature using the neural field and decode it into Gaussian properties with tiny MLP. These Gaussian properties are then aggregated with the explicit Gaussians and integrated into the 3DGS rasterizer.

# Code 
Coming soon. Please stay tuned.
