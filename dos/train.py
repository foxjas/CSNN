import torch
import torch.nn.functional as F
from torch import nn
import argparse
import os
import logging
import sys
import time
import numpy as np
import csv
from .model import CSGNN
from .dataset import AtomicConfigurations
from .fp_spherical import ConfigurationFingerprint
from mesh.mesh import icosphere


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_step(model, data, mask, target, optimizer, weight):
    model.train()
    data, mask, target = data.cuda(), mask.cuda(), target.cuda()
    dos, fdos = model(data, mask)
    ref_dos = target[:, :310]
    ref_fdos = target[:, 310:]
    loss_dos = F.mse_loss(dos, ref_dos)
    loss_fdos = F.mse_loss(fdos, ref_fdos)
    loss = weight*loss_dos + (1-weight)*loss_fdos

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss_dos.item(), loss_fdos.item(), loss.item()


@torch.inference_mode()
def test_step(model, data, mask, target, weight):

    model.eval()
    data, mask, target = data.cuda(), mask.cuda(), target.cuda()

    dos, fdos = model(data, mask)
    ref_dos = target[:, :310]
    ref_fdos = target[:, 310:]
    loss_dos = F.mse_loss(dos, ref_dos)
    loss_fdos = F.mse_loss(fdos, ref_fdos)
    loss = weight*loss_dos + (1-weight)*loss_fdos

    return loss_dos.item(), loss_fdos.item(), loss.item()


def get_log_name(args):
    log_name = "dos-l{}-B{}-R{}-kr{}-rcut{}-{}-m_{}-w{}_sc_{}-p{}-lr{}".format(args.l, args.batch_size, args.R, args.k_radial, args.rcut, args.g, args.m, args.loss_weight, args.scaling, args.p, args.lr)
    if args.m == "kernel":
        log_name += "-T{}".format(args.T)
    if args.rotate_train:
        log_name += "-rtrain"
    if args.dropout:
        log_name += "-drop{}".format(args.dropout)
    if args.weight_decay:
        log_name += "-wd{}".format(args.weight_decay)
    if args.seed:
        log_name += "-s{}".format(args.seed)
    if hasattr(args, "debug") and args.debug:
        log_name += "-debug"
    if args.identifier:
        log_name += "-{}".format(args.identifier)

    return log_name


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def model_setup(args, **kwargs):

    pool_type = args.p
    if pool_type not in ["sum", "max", "mean"]:
        print("invalid pooling type: {}".format(pool_type))
        sys.exit()

    mesh = kwargs["mesh"]
    conv_depth_dims = [1, 1, 1, 0]
    conv_block_dims = [32, 64, 128, 256]

    if args.R == 1:
        conv_radial_dims = [64, 128, 256, 512]
        if args.k_radial:
            conv_radial_layers = [2, 2, 2, 1]
        else:
            conv_radial_layers = []
    elif args.R > 1:
        conv_radial_dims = [64, 128, 256, 512]
        conv_radial_layers = [2, 2, 2, 1]
    else:
        print("invalid R ({}) or K_r ({})".format(args.R, args.k_radial))
        sys.exit()

    output_hidden_dims = [512, 512]
    
    pad = int((args.k_radial-1)/2)
    out_dim = 310
    in_feat_dim = args.R

    model = CSGNN(mesh, args.R, args.g, conv_block_dims, conv_depth_dims, output_hidden_dims, in_feat_dim, out_dim, pool_type=pool_type, conv_radial_layers=conv_radial_layers, conv_radial_dims=conv_radial_dims, k_radial=args.k_radial, pad=pad, dropout=args.dropout, classify=False)

    if hasattr(args, "model_path") and args.model_path:
        print("Loading model {}".format(args.model_path))
        model.load_state_dict(torch.load(args.model_path))
    else:
        print("Initializing new model")

    model.cuda()
    return model


def train(args, model, **kwargs):

    data_path = os.path.join(args.data_dir, args.dataset) 

    fp_transform = ConfigurationFingerprint(args.rcut, args.R, args.l, args.m, args.scaling, T=args.T)
    train_data = AtomicConfigurations(data_path, "train", data_transform=fp_transform, rotate=args.rotate_train)
    valid_data = AtomicConfigurations(data_path, "validation", data_transform=fp_transform, rotate=args.rotate_test)
    n_train, n_valid = len(train_data), len(valid_data)
    print("n_train: {}, n_valid: {}".format(n_train, n_valid))

    num_workers = args.workers
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    """ logging """
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    console = logging.StreamHandler(sys.stdout)
    logger.addHandler(console)
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(args.data_dir, "dos_out")
    log_dir = os.path.join(out_dir, "logs")
    if args.out:
        log_dir = os.path.join(log_dir, args.out)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_name = get_log_name(args)
    save_model_name = log_name + ".pkl"
    log_fname = "{}.txt".format(log_name)
    log_path = os.path.join(log_dir, log_fname)

    fh = logging.FileHandler(os.path.join(log_dir, log_fname))
    logger.addHandler(fh)

    init_lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=args.weight_decay)
    DECAY_FACTOR = 0.1
    stop_thresh = 5e-3
    decay_patience = 99
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                  factor=DECAY_FACTOR,
                                  patience=decay_patience, threshold=stop_thresh)
    best_val_loss = 1e10
    saved_train_loss = 1e10
    saved_dos_loss = 1e10
    saved_fdos_loss = 1e10

    logger.info("Parameters: {}".format(count_parameters(model)))
    logger.info("{}\n".format(model))
    for epoch in range(args.epochs):

        lr_curr = optimizer.param_groups[0]['lr']
        total_dos_loss = 0
        total_fdos_loss = 0
        total_loss = 0
        n_batches = 0

        time_before_train = time.perf_counter()
        time_before_load = time.perf_counter()
        for batch_idx, (data, mask, target) in enumerate(train_loader):
            time_after_load = time.perf_counter()
            n_batches += 1
            time_before_step = time.perf_counter()
            dos_loss, fdos_loss, loss = train_step(model, data, mask, target, optimizer, args.loss_weight)
            total_dos_loss += dos_loss
            total_fdos_loss += fdos_loss
            total_loss += loss
            
            logger.info("[{}:{}/{}] DOS={:.3} FDOS={:.3} LOSS={:.3} time={:.2f}+{:.2f}".format(
                epoch, batch_idx, len(train_loader),
                dos_loss, fdos_loss, loss,
                time_after_load - time_before_load,
                time.perf_counter() - time_before_step))

            time_before_load = time.perf_counter()
            if args.debug:
                break

        total_loss /= n_batches
        logger.info("mode=training epoch={} lr={:.3} <DOS>={:.3} <FDOS>={:.3} <LOSS>={:.3} time={:.3f}".format(
                epoch, 
                lr_curr,
                total_dos_loss / n_batches,
                total_fdos_loss / n_batches,
                total_loss,
                time.perf_counter() - time_before_train))


        # validation
        total_val_loss = 0
        total_val_dos_loss = 0
        total_val_fdos_loss = 0

        n_batches = 0
        time_before_eval = time.perf_counter()
        for batch_idx, (data, mask, target) in enumerate(valid_loader):
            dos_loss, fdos_loss, loss = test_step(model, data, mask, target, args.loss_weight)
            total_val_dos_loss += dos_loss
            total_val_fdos_loss += fdos_loss
            total_val_loss += loss
            n_batches += 1
            if args.debug:
                break

        total_val_dos_loss /= n_batches
        total_val_fdos_loss /= n_batches
        total_val_loss /= n_batches
        if total_val_loss <= (1-stop_thresh)*best_val_loss:
            best_val_loss = total_val_loss
            saved_dos_loss = total_val_dos_loss
            saved_fdos_loss = total_val_fdos_loss
            saved_train_loss = total_loss
            torch.save(model.state_dict(), os.path.join(log_dir, save_model_name))

        logger.info("mode=validation epoch={} lr={:.3} <DOS>={:.3} <FDOS>={:.3} <LOSS>={:.3} *LOSS={:.3} time={:.3f}".format(
                epoch,
                lr_curr,
                total_val_dos_loss,
                total_val_fdos_loss,
                total_val_loss,
                best_val_loss,
                time.perf_counter() - time_before_eval))

        if lr_curr < 1e-5:
            logger.info("Early termination at LR={}".format(lr_curr))
            break

        scheduler.step(total_val_loss)
        if args.debug:
            break
    
    results = {"train loss": saved_train_loss, 
               "val loss": best_val_loss, 
               "val dos loss": saved_dos_loss,
               "val fdos loss": saved_fdos_loss}

    return results, logger


def write_results(val_results, test_results, outpath, args):
    """
    Writes header (if file doesn't exist) and results row containing
    both experiment hyperparameters and result metrics.
    """
    model_fields = ["B", "lr", "L", "R", "w", "wd", "scaling", "epochs", "custom"]
    metric_fields = ["train loss", "val loss", "val dos loss", "val fdos loss", "dos", "fdos", "fermi", "energy"]  
    fields = model_fields + metric_fields
    try:
        with open(outpath, 'x') as f:
            header_writer = csv.writer(f, delimiter=",")
            header_writer.writerow(fields)
    except FileExistsError:
        print('File already exists. Skipping writing of header')

    with open(outpath, "a+", newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        results = {**val_results, **test_results}
        metric_vals = [str(results[k]) for k in metric_fields]
        model_vals = [args.batch_size, args.lr, args.l, args.R, args.loss_weight, args.weight_decay, args.scaling, args.epochs, args.identifier] 
        entries = model_vals + metric_vals
        csvwriter.writerow(entries)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="carbon_database", help="Dataset name")
    parser.add_argument("-d", "--data_dir", default=".", help="Data directory")
    parser.add_argument("-o", "--out", default='', help="Name of output directory") 
    parser.add_argument('-l', '--level', help='Level of mesh refinement', type=int, dest='l', default=3)
    parser.add_argument('-R', type=int, default=32, help='Number of radial levels')
    parser.add_argument('-rcut', help='Neighborhood radius for molecular environment', type=float, default=7.0)
    parser.add_argument("-p", default="mean", help="Pooling type: [max, sum, mean]")
    parser.add_argument("-m", default="linear", help="Data mapping type")
    parser.add_argument('-w', "--loss-weight", help='Weight assigned to DOS loss', type=float, default=0.1)
    parser.add_argument("-scaling", default="inverse", choices=["none", "inverse", "inverse_sq"], help="Distance scaling type")
    parser.add_argument("-k_radial", type=int, help="Radial kernel size", default=1)
    parser.add_argument("-g", help="Graph convolution type", choices=["gcn", "graphsage"], default="gcn")
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument('-T', help='Smoothing cut-off', type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, help='random seed', default=24)
    parser.add_argument("--rotate-train", action='store_true')
    parser.add_argument("--rotate-test", action='store_true')
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--debug", action='store_true', help="Debug mode: single batch, single epoch")
    parser.add_argument("--identifier", help="user-defined string to add to log name")
    parser.add_argument('--device', default=0, type=int, help='GPU device ID')
    parser.add_argument("--workers", type=int, default=6, help="Number of processes for data loading")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    icosphere = icosphere(args.l)
    model = model_setup(args, mesh=icosphere)
    val_results, logger = train(args, model)
