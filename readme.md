# Welcome to LongiSeg!

## What is LongiSeg?
LongiSeg is an extension of the popular [nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet), designed specifically for longitudinal medical image segmentation. It enhances segmentation performance by leveraging temporal information across multiple timepoints.

LongiSeg introudces several methods for temporal feature merging, including the newly introduced [Difference Weighting Block](https://arxiv.org/abs/2409.13416). For more details on the underlying nnU-Net framework, visit the [nnU-Net repository](https://github.com/MIC-DKFZ/nnUNet).

Please cite the following papers when using LongiSeg in your research:

```bibtex
@article{rokuss2024longitudinal,
  title={Longitudinal segmentation of MS lesions via temporal Difference Weighting},
  author={Rokuss, Maximilian and Kirchhoff, Yannick and Roy, Saikat and Kovacs, Balint and Ulrich, Constantin and Wald, Tassilo and Zenk, Maximilian and Denner, Stefan and Isensee, Fabian and Vollmuth, Philipp and Kleesiek, Jens and Maier-Hein, Klaus},
  journal={arXiv preprint arXiv:2409.13416},
  year={2024}
}
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## Getting started
LongiSeg is easy to use and follows the basic principles of the nnU-Net framework.

In order to get started, we recommend to create a virtual environment with **at least** Python 3.10, e.g. using conda

```bash
conda create -n longiseg python=3.12
```

LongiSeg is not yet available via pip, therefore you need to clone the repository and install it locally. This will also allow for easy customization of the code.

```bash
conda activate longiseg
git clone https://github.com/MIC-DKFZ/LongiSeg.git
cd LongiSeg
pip install -e .
```

Finally, you need to set the paths for raw data, preprocessed data and results
```bash
export LongiSeg_raw="/path_to_data_dir/LongiSeg_raw"
export LongiSeg_preprocessed="/path_to_data_dir/LongiSeg_preprocessed"
export LongiSeg_results="/path_to_experiments_dir/LongiSeg_results"
```

If these are not set, LongiSeg will fall back to the respective nnU-Net paths, ensuring compatability with nnU-Net setups.

For more details on installation requirements and dataset structure, refer to the [nnU-Net installation](documentation/installation_instructions.md) and the [path setup](documentation/setting_up_paths.md) documentation.

## Compatibility with nnU-Net
LongiSeg is fully compatible with nnU-Net and can be installed alongside it in the same environment. This allows users to seamlessly reuse existing nnU-Net structures, datasets, and preprocessing pipelines without modification.

# Acknowledgements
<img src="documentation/assets/dkfz_logo.png" height="100px" />