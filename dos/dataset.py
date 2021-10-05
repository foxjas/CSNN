import csv
import sys
import glob
import os
import re
import numpy as np
import torch
import torch.utils.data
import h5py
import pandas as pd
from pymatgen.io.vasp.outputs import Poscar
from .fp_spherical import ConfigurationFingerprint

CONFIG_FILE_NAME="POSCAR"
REF_FILE_NAME="doscar_s"

class AtomicConfigurations(torch.utils.data.Dataset):


    def __init__(self, root, mode, data_transform=None, target_transform=None, test_files=None, rotate=False):
        self.root = os.path.expanduser(root)
        self.mode = mode
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.config_data = []
        self.dos_data = []
        datasets = []
        self.rotate = rotate

        if mode not in ["train", "validation", "test"]:
            raise Exception("Invalid dataset mode {}".format(mode))
  
        data_info_file = None
        if mode == "train":
            data_info_file = "Train.csv"
            split_key = "database/"
        elif mode == "validation":
            data_info_file = "Val.csv"
            split_key = "C_data_atom_center_smw0.2/"
        if mode == "test":
            config_dirs, config_names = [], []
            for f_path in test_files:
                config_dirs.append(f_path)
                target_part = f_path.split("/")[-3:]
                config_name = "-".join(target_part)
                config_names.append(config_name)
            self.config_dirs, self.config_names = config_dirs, config_names
        else:
            self.config_dirs = []
            data_info_dir = os.path.dirname(os.path.realpath(__file__))
            data_info_path = os.path.join(data_info_dir, data_info_file)
            with open(data_info_path, "r") as f:
                reader = csv.reader(f, delimiter=',')
                next(reader) # skip header
                for row in reader:
                    f_path = row[1]
                    target_part = f_path.split(split_key)[-1]
                    name, temp, config_name = target_part.strip().split("/")
                    if "K" not in temp:
                        target_part = "{}/{}K/{}".format(name, temp, config_name)
                    config_dir = os.path.join(self.root, target_part)
                    self.config_dirs.append(config_dir)

        self.max_atoms = -1
        for config_dir in self.config_dirs:
            config_file = os.path.join(config_dir, CONFIG_FILE_NAME)
            config_data = Poscar.from_file(config_file)
            self.config_data.append(config_data)
            
            n_atoms = len(config_data.structure.sites)
            if n_atoms > self.max_atoms:
                self.max_atoms = n_atoms

            ref_file = os.path.join(config_dir, REF_FILE_NAME)
            ref_data = pd.read_csv(ref_file, delimiter=' ', header=None, usecols=[1])
            dos = np.squeeze(np.array(ref_data[:-1]))

            dos = np.array(dos)/n_atoms
            fdos=0.100*np.cumsum(dos)
            ref_dos = np.concatenate((dos, fdos))
            self.dos_data.append(ref_dos)

        #print("max # atoms: {}".format(self.max_atoms))


    def __getitem__(self, index):
        x = self.config_data[index]
        y = self.dos_data[index] 
        x, atom_mask = self.data_transform(x, self.max_atoms, self.rotate)
        atom_mask = torch.BoolTensor(atom_mask)
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        items = [x, atom_mask, y]
        if self.mode == "test":
            items.append(self.config_names[index])
        return items


    def __len__(self):
        return len(self.config_data)


def get_files_matching(match_expr, base_path):
    search_expr = os.path.join(base_path, "**/{}".format(match_expr))
    matches = glob.glob(search_expr, recursive=True)
    return matches
