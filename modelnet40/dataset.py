import os
import numpy as np
import torch
import torch.utils.data
import argparse
import h5py
import json
import sys


def rotate_point_cloud_so3(points):
    angles = np.random.uniform(0, 1, 3) * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(angles[0]), -np.sin(angles[0])],
                    [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                    [0, 1, 0],
                    [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                    [np.sin(angles[2]), np.cos(angles[2]), 0],
                    [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    points = np.dot(points.reshape((-1, 3)), R)
    return points


def rotate_perturbation_point_cloud(points, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
            Nx3 array, original point clouds
        Return:
            Nx3 array, rotated point clouds
    """
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(angles[0]), -np.sin(angles[0])],
                    [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                    [0, 1, 0],
                    [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                    [np.sin(angles[2]), np.cos(angles[2]), 0],
                    [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    points = np.dot(points.reshape((-1, 3)), R)
    return points


def rotate_point_cloud_z(points):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
            Nx3 array, original point clouds
        Return:
            Nx3 array, rotated point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])

    return np.dot(points.reshape((-1, 3)), rotation_matrix)


def getDataFiles(list_filename, root=''):
    file_names = [os.path.split(line.rstrip())[1] for line in open(list_filename)]
    if root:
        file_names = [os.path.join(root, fname) for fname in file_names]
    return file_names 

def load_h5(h5_filename):
    f = h5py.File(h5_filename, "r")
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def loadDataFileNames(filename, root=''):
    fpath = os.path.join(root, filename)
    with open(fpath) as f:
        names = json.load(f)
    return names


class ModelNet(torch.utils.data.Dataset):

    def __init__(self, data_dir, dataset, train, num_votes=10, reposition=True,
                       rescale=True, rotate_perturb=False, data_transform=None, 
                       target_transform=None, rotate="NR", keep_raw_data=False):

        self.data_transform = data_transform
        self.target_transform = target_transform
        self.num_votes = num_votes
        self.reposition = reposition
        self.rescale = rescale
        self.rotate_perturb = rotate_perturb
        self.rotate = rotate
        self.n_points = 1024
        self.keep_raw_data = keep_raw_data
        file_dir = os.path.join(data_dir, dataset)
        self.train = train
        if train:
            self.files = getDataFiles( \
                os.path.join(data_dir, '{}/train_files.txt'.format(dataset)), root=file_dir)
        else:
            self.files = getDataFiles(\
                os.path.join(data_dir, '{}/test_files.txt'.format(dataset)), root=file_dir)
        self.class_names = [line.rstrip() for line in \
            open(os.path.join(file_dir, 'shape_names.txt'))]

        data, labels = [], []
        for f in self.files:
            current_data, current_labels = loadDataFile(f)
            data.append(current_data)
            labels.append(current_labels)

        self.X = np.vstack(data)
        self.labels = np.vstack(labels)


    def shift_point_cloud(self, points, shift_range=0.1):
        """ 
        Randomly shift point cloud. Shift is per point cloud.
        """
        shifts = np.random.uniform(-shift_range, shift_range, (1, 3))
        return points+shifts


    def random_scale_point_cloud(self, points, scale_low=0.8, scale_high=1.2):
        """ 
        Randomly scale the point cloud. Scale is per point cloud.
        """
        scales = np.random.uniform(scale_low, scale_high, 3)
        return points*scales


    def jitter_points(self, points, noise_sigma=0.01, noise_clip=0.05):
        N, C = points.shape
        noise = np.clip(noise_sigma * np.random.randn(N, C), -1 * noise_clip, noise_clip)
        return points+noise


    def __getitem__(self, index):
        points = self.X[index]
        label = [self.labels[index]]
        
        if self.train:
            points = points[:self.n_points, :]
            if self.rescale:
                points = self.random_scale_point_cloud(points)
            if self.rotate == "z":
                points = rotate_point_cloud_z(points)
            elif self.rotate == "SO3":
                points = rotate_point_cloud_so3(points)
            if self.rotate_perturb:
                points = rotate_perturbation_point_cloud(points)
            if self.reposition:
                points = self.shift_point_cloud(points)
                points = self.jitter_points(points)
            x = self.data_transform(points)
        else:
            points = points[:self.n_points, :]
            test_data = []
            for vote_idx in range(self.num_votes):
                if self.rotate == "z":
                    points_rot = rotate_point_cloud_z(points)
                elif self.rotate == "SO3":
                    points_rot = rotate_point_cloud_so3(points)
                x_inst = self.data_transform(points_rot)
                test_data.append(x_inst)
            x = np.stack(test_data)

        if self.target_transform:
            label = self.target_transform(x)
         
        x = torch.FloatTensor(x)
        label = torch.LongTensor(label).squeeze()

        items = [x, label]
        if self.keep_raw_data:
            x_raw = self.X[index]
            items.append(x_raw)
        return items

    def __len__(self):
        return len(self.X)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default=".", help="Target data directory")
    args = parser.parse_args()

    # Download ModelNet40 dataset for point cloud classification
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    if not os.path.exists(os.path.join(args.data_dir, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], args.data_dir))
