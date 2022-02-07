# Concentric Spherical Neural Network for 3D Representation Learning

## Overview
This library contains a PyTorch implementation of CSNN. It was originally run using Python 3.8, PyTorch 1.9, DGL 0.6.1, and CUDA 11.1.

## Dependencies
The following installs dependencies to Anaconda virtual environment. 
```bash
conda create --name csgnn python=3.8
conda activate csgnn
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia
conda install -c dglteam dgl-cuda11.1
conda install h5py
pip install requests
```

### ModelNet40 experiment
The follow commands are called from the top level directory of this project.

First, retrieve the dataset:
```bash
python -m modelnet40.dataset
```
This downloads the dataset to the path "./modelnet40_ply_hdf5_2048".

There are two pre-trained models: "csgnn-modelnet-z" is trained on z-axis aligned rotations, and "csgnn-modelnet-SO3" is trained on SO3 rotations.
For example, to evaluate the SO3-trained model on SO3-rotated test data, run:
```bash
python -m modelnet40.test modelnet40/saved/csgnn-modelnet-SO3.pkl --rotate_test SO3
```

To train the model according to the paper (with default settings for SO3 training), run
```bash
python -m modelnet40.train
```

### Electronic density of states (DOS) experiment
Extract the carbon dataset as follows:
```bash
tar -xzvf carbon_database.tar.gz
```

In addition to the dependencies listed earlier, the following packages are
required to run the experiment:
```bash
pip install pymatgen
pip install scikit-learn
```

To evaluate pre-trained model for overall error, or error grouped by structure type, run:
```bash
python -m dos.test dos/saved/csgnn-dos.pkl --mode [all/group]
```

To train the model according to the paper, run: 
```bash
python -m dos.train
```
