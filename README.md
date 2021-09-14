# Unseen Object Instance Segmentation for Robotic Environments

<img src="gifs/pipeline.gif" height="200" />

This is a PyTorch-based implementation of our network, UOIS-Net-3D, for unseen object instance segmentation. Our instance segmentation algorithm utilizes a two-stage method to explicitly leverage the strengths of depth and RGB separately for stronger instance segmentation. Surprisingly, our framework is able to learn from synthetic RGB-D data where the RGB is non-photorealistic. Details of the algorithm can be found in our arXiv paper:

[Unseen Object Instance Segmentation for Robotic Environments](https://arxiv.org/abs/2007.08073)<br/>
[Christopher Xie](https://chrisdxie.github.io), [Yu Xiang](https://yuxng.github.io), [Arsalan Mousavian](https://cs.gmu.edu/~amousavi/), [Dieter Fox](https://homes.cs.washington.edu/~fox/) <br/>
IEEE Transactions on Robotics (T-RO), 2021.

## Installation

We highly recommend setting up a virtual environment using [Anaconda](https://www.anaconda.com/distribution/). Here is an example setup using these tools:

```bash
git clone https://github.com/chrisdxie/uois.git
cd uois3d/
conda env create -f env.yml
```

## Models
You can find the models [here](https://drive.google.com/uc?export=download&id=1D-eaiOgFq_mg8OwbLXorgOB5lrxvmgQd). We provide a Depth Seeding Network (DSN) model trained on our synthetic Tabletop Object Dataset (TOD), a Region Refinement Network (RRN) model trained on TOD, and an RRN model trained on real data from the [Google Open Images Dataset (OID)](https://storage.googleapis.com/openimages/web/download.html).

## Data
You can find the Tabletop Object Dataset (TOD) [here](https://drive.google.com/uc?export=download&id=157nWfb4pLbwAfOdMLls6q0lZQyqUCvLY). See the [data loading](src/data_loader.py) and [data augmentation](src/data_augmentation.py) code for more details.

## Train the network
We provide sample training code in [train_DSN.ipynb](train_DSN.ipynb) and [train_RRN.ipynb](train_RRN.ipynb).

## Run the network
See [uois_3D_example.ipynb](uois_3D_example.ipynb) for an example of how to run the network on example images. In order to run this file, Jupyter Notebook must be installed (this is included in `env.yml`). If you haven't used Jupyter Notebooks before, [here](https://www.dataquest.io/blog/jupyter-notebook-tutorial/) is a tutorial to get you up to speed. This repository provides a few images in the [example_images](example_images/) folder. 

Notes:

* Make sure to activate the Anaconda environment before running jupyter. This can be done with ``` conda activate uois3d; jupyter notebook ```
* the notebook should be run in the directory in which it lives (`<ROOT_DIR>`), otherwise the filepaths must be manually adjusted.
* After downloading and unzipping the models, make sure to update `checkpoint_dir` in [uois_3D_example.ipynb](uois_3D_example.ipynb) to point to the directory where the models live.

## Citation
Our code is released under the MIT license.

If you find our work helpful in your research, please cite our work.

```
@article{xie2021unseen,
author    = {Christopher Xie and Yu Xiang and Arsalan Mousavian and Dieter Fox},
title     = {Unseen Object Instance Segmentation for Robotic Environments},
journal   = {IEEE Transactions on Robotics (T-RO)},
year      = {2021}
}
```
