import torch
import torch.nn.functional as F
from torch import nn
import argparse
import os
import logging
import sys
import time
import numpy as np
from .model import CSGNN
from .dataset import ModelNet
from mesh.mesh import icosphere
from .mapping import IcosphereSignalConstant, IcosphereSignalKernel

N_CLASS = 40

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_step(model, data, target, optimizer):
    model.train()
    data, target = data.cuda(), target.cuda()

    output = model(data)
    prediction = F.log_softmax(output, dim=-1)
    loss = F.nll_loss(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    correct = prediction.data.max(1)[1].eq(target.data).long().cpu().sum()

    return loss.item(), correct.item()


@torch.inference_mode()
def test_step(model, data, target, num_votes, n_classes=40):
    model.eval()
    data, target = data.cuda(), target.cuda()

    pred_sum = torch.zeros(data.shape[0], n_classes)
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


def get_log_name(args):
    out_dir = os.path.join(args.data_dir, "modelnet40_out")
    log_dir = os.path.join(out_dir, "logs")
    if args.out:
        log_dir = os.path.join(log_dir, args.out)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    R = args.R
    lr = args.lr
    dropout = args.dropout
    weight_decay = args.weight_decay
    param_factor = args.F
    mapping = args.m
    
    log_name = "l{}-R{}-kr{}-F{}-m_{}-b{}-lr{:.2e}-{}".format(args.l, R, args.k_radial, param_factor, mapping, args.batch_size, lr, args.rotate_train)
    if mapping == "kernel":
        log_name += "-e{}".format(args.R_expand)
        log_name += "-T{}".format(args.kernel_thresh)
    if args.p != "max":
        log_name += "-p_{}".format(args.p)
    if dropout:
        log_name += "-drop{:.2f}".format(dropout)
    if weight_decay:
        log_name += "-wd{:.2e}".format(weight_decay)
    if args.seed:
        log_name += "-s{}".format(args.seed)
    if args.identifier:
        log_name += "-{}".format(args.identifier)

    return log_dir, log_name


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def model_dims(n_levels, base_dim, factor=1):
    dims_init = [base_dim*(2**i) for i in range(n_levels+1)]
    dims = [factor*dim for dim in dims_init]
    return dims


def model_setup(args, **kwargs):
    R = args.R
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


def prepare_data(args):

    mapping = args.m
    R = args.R
    data_transform = None
    if mapping == "constant":
        data_transform = IcosphereSignalConstant(args.l, R)
    elif mapping == "kernel":
        data_transform = IcosphereSignalKernel(args.l, R, args.R_expand, T=args.kernel_thresh)

    train_data = ModelNet(args.data_dir, args.dataset, True, data_transform=data_transform, rotate=args.rotate_train)
    valid_data = ModelNet(args.data_dir, args.dataset, False, num_votes=args.num_votes, data_transform=data_transform, rotate=args.rotate_test)

    num_workers = args.workers
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    n_train, n_valid = len(train_data), len(valid_data)

    return train_loader, valid_loader


def train(args):

    dropout = args.dropout
    lr = args.lr
    wd = args.weight_decay
    param_factor = args.F

    train_loader, valid_loader = prepare_data(args)
    model = model_setup(args)

    """ logging """
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    console = logging.StreamHandler(sys.stdout)
    logger.addHandler(console)
    log_dir, log_name = get_log_name(args) 
    save_model_name = log_name + ".pkl"
    log_fname = "{}.txt".format(log_name)
    log_path = os.path.join(log_dir, log_fname)

    fh = logging.FileHandler(os.path.join(log_dir, log_fname))
    logger.addHandler(fh)

    logger.info("Parameters: {}".format(count_parameters(model)))
    logger.info("{}\n".format(model))

    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    DECAY_FACTOR = 0.1
    stop_thresh = 1e-3
    decay_patience = 4
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                  factor=DECAY_FACTOR,
                                  patience=decay_patience, threshold=stop_thresh)

    best_val_loss = 1e10
    best_val_acc = -1
    best_val_class_acc = -1
    for epoch in range(args.epochs):
        total_loss = 0
        total_correct = 0
        time_before_load = time.perf_counter()
        n_batches = 0
        n_seen = 0

        time_before_train = time.perf_counter()

        for batch_idx, (data, target) in enumerate(train_loader):
            time_after_load = time.perf_counter()
            n_batches += 1
            n_seen += len(data)
            time_before_step = time.perf_counter()
            loss, correct = train_step(model, data, target, optimizer)
            total_loss += loss
            total_correct += correct
            
            logger.info("[{}:{}/{}] LOSS={:.3} <LOSS>={:.3} ACC={:.3} <ACC>={:.3} time={:.2}+{:.2}".format(
                epoch, batch_idx, len(train_loader),
                loss, total_loss / n_batches,
                correct / len(data), total_correct / n_seen,
                time_after_load - time_before_load,
                time.perf_counter() - time_before_step))

            time_before_load = time.perf_counter()
            if args.debug:
                break

        logger.info("mode=training epoch={} lr={:.2e} <LOSS>={:.3} <ACC>={:.3f} time={:.3f}".format(
                epoch,
                lr,
                total_loss / n_batches,
                total_correct / n_seen,
                time.perf_counter() - time_before_train))


        # test
        total_loss = 0
        total_correct = 0
        n_batches = 0
        n_seen = 0
        total_seen_class = np.zeros(N_CLASS)
        total_correct_class = np.zeros(N_CLASS)
        time_before_eval = time.perf_counter()
        for batch_idx, (data, target) in enumerate(valid_loader):
            loss, correct = test_step(model, data, target, args.num_votes) 
            total_loss += loss
            n_correct = correct.sum().item()
            total_correct += n_correct
            n_batches += 1
            n_seen += len(data)
            for l_ref, pred_val in zip(target.data.cpu().numpy(), correct):
                total_correct_class[l_ref] += pred_val
                total_seen_class[l_ref] += 1

            if args.debug:
                break

        total_loss_avg = total_loss / n_batches
        if total_loss_avg < best_val_loss:
            best_val_loss = total_loss_avg

        val_acc = total_correct / n_seen
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), os.path.join(log_dir, save_model_name))
            best_val_acc = val_acc

        val_class_acc = np.mean(total_correct_class/total_seen_class)
        if val_class_acc > best_val_class_acc:
            best_val_class_acc = val_class_acc

        logger.info("mode=validation epoch={} lr={:.3} <LOSS>={:.3} *LOSS={:.3} <ACC>={:.3f} *ACC={:.3f} <C_ACC>={:.3f} time={:.3}".format(
                epoch,
                lr,
                total_loss_avg,
                best_val_loss,
                val_acc,
                best_val_acc,
                val_class_acc,
                time.perf_counter() - time_before_eval))

        lr_curr = optimizer.param_groups[0]['lr']
        if lr_curr < 1e-5:
            logger.info("Early termination at LR={}".format(lr_curr))
            break

        scheduler.step(total_loss)

    logger.info("Final: *ACC={:3f} *C_ACC={:.3f}".format(best_val_acc, best_val_class_acc))

    return best_val_loss


if __name__ == "__main__":

    DATASET = 'modelnet40_ply_hdf5_2048'

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default=".", help="Data directory")
    parser.add_argument("-o", "--out", help="Name of output directory") 
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument('-l', '--level', type=int, dest='l', default=4, help='Level of mesh refinement')
    parser.add_argument('-R', type=int, help='Number of radial levels', default=20)
    parser.add_argument("-p", default="max", choices=["max", "mean", "sum"], help="Pooling type: [max, sum, mean]")
    parser.add_argument("-m", default="kernel", choices=["constant", "kernel"], help="Data mapping type")
    parser.add_argument("-k_radial", type=int, help="Radial kernel size", default=3)
    parser.add_argument("-F", type=int, help="Controls model layer parameters", default=4)
    parser.add_argument("-g", help="Graph convolution type", choices=["gcn", "graphsage"], default="gcn")
    parser.add_argument('-e', '--R_expand', help='Radial expansion factor for kernel', type=int, default=8)
    parser.add_argument('-T', '--kernel-thresh', help='Threshold for kernel mapping', type=float, default=0.01)
    parser.add_argument("--dataset", default=DATASET, help="Dataset name") 
    parser.add_argument("--lr", type=float, default=2.2e-4)
    parser.add_argument("--dropout", type=float, default=0.14)
    parser.add_argument("--weight-decay", type=float, default=2.7e-7)
    parser.add_argument("--seed", type=int, help='random seed', default=24)
    parser.add_argument("--num-votes", type=int, help='Number of votes during testing (rotation only)', default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--rotate_train", default="SO3", choices=["NR", "z", "SO3"])
    parser.add_argument("--rotate_test", default="SO3", choices=["NR", "z", "SO3"])
    parser.add_argument('--device', default=0, type=int, help='GPU device ID')
    parser.add_argument("--workers", type=int, default=6, help="Number of processes for data loading")
    parser.add_argument("--debug", action='store_true', help="Debug mode")
    parser.add_argument("--identifier", help="user-defined string to add to log name")
    args = parser.parse_args()

    train(args)
