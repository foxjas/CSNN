import torch
import torch.nn.functional as F
from torch import nn
import argparse
import os
import sys
import time
import numpy as np
from .model import CSGNN
from .dataset import ModelNet
from mesh.mesh import icosphere
from .mapping import IcosphereSignalConstant, IcosphereSignalKernel
from .train import model_dims, worker_init_fn, N_CLASS


@torch.inference_mode()
def test_step(model, data, target, num_votes):
    model.eval()
    data, target = data.cuda(), target.cuda()

    pred_sum = torch.zeros(data.shape[0], N_CLASS)
    total_loss = 0
    for vote_idx in range(num_votes):
        points = data[:, vote_idx, ...]
        output = model(points)
        prediction = F.softmax(output, dim=1)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item()

    pred_sum += prediction.data.cpu()

    prediction = pred_sum.max(1)[1]
    target = target.data.long().cpu()
   
    correct = prediction.eq(target)
    total_loss /= num_votes

    return total_loss, correct


def prepare_data(args, **kwargs):
    R = kwargs.get("R", args.R)
    num_votes = kwargs.get("num_votes", args.num_votes)
    data_transform = IcosphereSignalKernel(args.l, R, args.R_expand, T=args.kernel_thresh)
    test_data = ModelNet(args.data_dir, args.dataset, False, num_votes=num_votes, data_transform=data_transform, rotate=args.rotate_test)
    num_workers = args.workers
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    return test_loader


def model_setup(args, **kwargs):
    R = kwargs.get("R", args.R)
    dropout = args.dropout
    mapping = args.m
    param_factor = args.F
    n_hidden = 2
    concatenate = True
    gconv = args.g

    mesh = icosphere(args.l)

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    pool_type = args.p
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.k_radial < 1:
        print("invalid R ({}) or K_r ({})".format(R, args.k_radial))
        sys.exit()

    BASE_DIM = 8
    conv_block_dims = model_dims(args.l, BASE_DIM, param_factor)
    conv_radial_dims = model_dims(args.l, BASE_DIM, param_factor*2)

    if args.l == 3:
        conv_depth_dims = [1, 1, 1, 0]
        conv_radial_layers = [2, 2, 2, 1]
    elif args.l == 4:
        conv_depth_dims = [1, 1, 1, 1, 0]
        conv_radial_layers = [2, 2, 2, 1, 1]
    elif args.l == 5:
        conv_depth_dims = [1, 1, 1, 1, 0, 0]
        conv_radial_layers = [2, 2, 2, 1, 1, 1]

    output_hidden_dims = []
    if n_hidden == 1:
        output_hidden_dims = [512]
    if n_hidden == 2:
        output_hidden_dims = [512, 256]

    pad = int((args.k_radial-1)/2)
    out_dim = 40
    if mapping == "kernel":
        in_feat_dim = args.R_expand
    else:
        in_feat_dim = 1

    model = CSGNN(mesh, R, gconv, conv_block_dims, conv_depth_dims, output_hidden_dims, in_feat_dim, out_dim, pool_type=pool_type, conv_radial_layers=conv_radial_layers, conv_radial_dims=conv_radial_dims, k_radial=args.k_radial, pad=pad, dropout=dropout, concatenate=concatenate)

    model.cuda()
    return model


def test(args):
    R = args.R
    num_votes = args.num_votes
    if "modelnet-z" in args.model_path:
        R = 15
        num_votes = 1
        if args.rotate_test == "SO3":
            num_votes=5
    elif "modelnet-SO3" in args.model_path:
        R = 20
        num_votes = 1

    test_loader = prepare_data(args, R=R, num_votes=num_votes)
    model = model_setup(args, R=R)

    print("Loading model {}".format(args.model_path))
    model.load_state_dict(torch.load(args.model_path))

    # test
    total_loss = 0
    total_correct = 0
    n_batches = 0
    n_seen = 0

    total_seen_class = np.zeros(N_CLASS)
    total_correct_class = np.zeros(N_CLASS)
    time_before_eval = time.perf_counter()
    for batch_idx, (data, target) in enumerate(test_loader):
        loss, correct = test_step(model, data, target, num_votes)
        total_loss += loss
        n_correct = correct.sum().item()
        total_correct += n_correct
        n_batches += 1
        n_seen += len(data)

        for l_ref, pred_val in zip(target.data.cpu().numpy(), correct):
            total_correct_class[l_ref] += pred_val
            total_seen_class[l_ref] += 1

    total_loss_avg = total_loss / n_batches
    test_acc = total_correct / n_seen
    test_class_acc = np.mean(total_correct_class/total_seen_class)

    results_str = "<LOSS>={:.3} <ACC>={:.4f} <C_ACC>={:.4f} time={:.3}".format(
            total_loss_avg,
            test_acc,
            test_class_acc,
            time.perf_counter() - time_before_eval)

    print(results_str)


if __name__ == "__main__":

    DATASET = 'modelnet40_ply_hdf5_2048'

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to saved model")
    parser.add_argument("-d", "--data_dir", default=".", help="Data directory")
    parser.add_argument('-R', type=int, help='Number of radial levels', default=20)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument('-l', '--level', type=int, dest='l', default=4, choices=[3, 4, 5], help='Level of mesh refinement')
    parser.add_argument("-p", default="max", choices=["max", "mean", "sum"], help="Pooling type")
    parser.add_argument("-m", choices=["constant", "kernel"], default="kernel", help="Data mapping type")
    parser.add_argument("-k_radial", type=int, help="Radial kernel size", default=3)
    parser.add_argument("-F", type=int, help="Controls model layer parameters", default=4)
    parser.add_argument("-g", help="Graph convolution type", choices=["gcn", "graphsage"], default="gcn")
    parser.add_argument('-e', '--R_expand', help='Radial expansion factor for kernel', type=int, default=8)
    parser.add_argument("--dataset", default='modelnet40_ply_hdf5_2048', help="Dataset name") 
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, help='random seed', default=24)
    parser.add_argument("--rotate_train", default="SO3", choices=["NR", "z", "SO3"])
    parser.add_argument("--rotate_test", default="SO3", choices=["NR", "z", "SO3"])
    parser.add_argument("--num-votes", type=int, help='Number of votes during testing (rotation only)', default=1)
    parser.add_argument('-T', '--kernel-thresh', help='Threshold for kernel mapping', type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument('--device', default=0, type=int, help='GPU device ID')
    parser.add_argument("--workers", type=int, default=6, help="Number of processes for data loading")

    args = parser.parse_args()
    test(args)


