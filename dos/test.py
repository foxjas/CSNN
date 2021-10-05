import torch
import torch.nn.functional as F
import argparse
import os
import logging
import sys
import numpy as np
import csv
from sklearn.metrics import mean_squared_error
from scipy import interpolate, integrate
from .dataset import AtomicConfigurations, get_files_matching
from .fp_spherical import ConfigurationFingerprint
from .train import model_setup, worker_init_fn
from mesh.mesh import icosphere

CARBON_GROUPS = ["Graphene", "Graphite", "C20", "C40", "C60", "C_6_4", "C_9_9", "C_8_0"]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.inference_mode()
def test_step(model, data, mask, target):

    model.eval()
    data, mask, target = data.cuda(), mask.cuda(), target.cuda()

    dos_pred, fdos_pred = model(data, mask)

    return dos_pred.detach().cpu().numpy(), fdos_pred.detach().cpu().numpy()


def predict_fermi_and_energy(pred_DOS, ref_DOS, n_atoms):
    """
    Input: config info, predicted & target result
    Compute DOS bins, Fermi level
    """
    total_elec = 4*n_atoms
    x_axis = np.arange(-30, 1, 0.1)
    fine_grid = np.arange(x_axis[0], x_axis[-2], 0.01)

    # reference fermi level
    f_prop = interpolate.interp1d(x_axis, ref_DOS, kind='cubic')
    dos_fine_grid_prop = f_prop(fine_grid)
    fermi_x_ref = 0
    integral = integrate.cumtrapz(dos_fine_grid_prop, fine_grid)
    for x in range(1, len(fine_grid)):
        if integral[x]>total_elec-0.005:
            fermi_x_ref = fine_grid[x-1]
            break

    # reference energy
    energy_fine_grid_prop=dos_fine_grid_prop*fine_grid
    integral_ener=integrate.cumtrapz(energy_fine_grid_prop,fine_grid)
    energy_ref = integral_ener[x-1]
    energy_ref /= n_atoms

    # predicted fermi level 
    f_pred = interpolate.interp1d(x_axis, pred_DOS, kind='cubic')
    dos_fine_grid_pred = f_pred(fine_grid)
    fermi_x_pred = 0
    fermi_x_pred_cond1=-200
    integral = integrate.cumtrapz(dos_fine_grid_pred, fine_grid)

    for x in range(1, len(fine_grid)-2):
        if integral[x]>total_elec-0.2 and integral[x]<total_elec+0.2:
            p=abs(fine_grid[x]-fermi_x_ref)
            if p < 0.5 :
                fermi_x_pred=fine_grid[x]
                if abs(fermi_x_pred-fermi_x_ref)<abs(fermi_x_pred_cond1-fermi_x_ref):
                    fermi_x_pred_cond1=fermi_x_pred
                else:
                    break

    for x in range(1, len(fine_grid)):
        if integral[x] > total_elec-0.005:
            fermi_x_pred_cond2 = fine_grid[x-1]
            break

    fermi_x_pred=fermi_x_pred_cond2

    # predicted energy 
    energy_fine_grid_pred=dos_fine_grid_pred*fine_grid
    integral_ener=integrate.cumtrapz(energy_fine_grid_pred,fine_grid)
    energy_pred = integral_ener[x]
    energy_pred /= n_atoms

    return fermi_x_ref, fermi_x_pred, energy_ref, energy_pred


def evaluate_groups(args, model, groups, config_data, write_path=None, **kwargs):

    data_path = os.path.join(args.data_dir, args.dataset)
    fp_transform = ConfigurationFingerprint(args.rcut, args.R, args.l, args.m, args.scaling, T=args.T)
    num_workers = args.workers
    all_results = []
    model_fields = ["B", "lr", "L", "R", "w", "wd", "scaling", "custom", "metric"]
    fields = model_fields + groups
    dos_results = ["dos"]
    fdos_results = ["fdos"]
    fermi_results = ["fermi"]
    energy_results = ["energy"]
    for group_name in groups:
        print(group_name)
        config_files = config_data[group_name]
        test_data = AtomicConfigurations(data_path, "test", data_transform=fp_transform, test_files=config_files, rotate=args.rotate_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
        total_rmse_dos = 0
        total_rmse_fdos = 0
        total_error_fermi = 0
        total_error_energy = 0
        n_samples = 0
        for batch_idx, (data, mask, target, config_name) in enumerate(test_loader):
            dos_pred, fdos_pred = test_step(model, data, mask, target)
            dos_pred = np.array(dos_pred)
            fdos_pred = np.array(fdos_pred)
            ref_dos = np.array(target[:, :310])
            ref_fdos = np.array(target[:, 310:])

            for config_name_inst, dos_pred_inst, fdos_pred_inst, dos_ref_inst, fdos_ref_inst, mask_inst in zip(config_name, dos_pred, fdos_pred, ref_dos, ref_fdos, mask):
                rmse_dos = np.sqrt(mean_squared_error(dos_ref_inst, dos_pred_inst))
                total_rmse_dos += rmse_dos
                
                n_atoms = np.sum(mask_inst.data.cpu().numpy())
                fdos_ref_inst *= n_atoms
                fdos_pred_inst *= n_atoms
                rmse_fdos = np.sqrt(mean_squared_error(fdos_ref_inst, fdos_pred_inst))
                total_rmse_fdos += rmse_fdos

                dos_ref_inst *= n_atoms
                dos_pred_inst *= n_atoms
                fermi_ref, fermi_pred, energy_ref, energy_pred = predict_fermi_and_energy(dos_pred_inst, dos_ref_inst, n_atoms)
                total_error_fermi += abs(fermi_ref-fermi_pred)
                total_error_energy += abs(energy_ref-energy_pred)
                n_samples += 1

        print(n_samples)

        dos_results.append(total_rmse_dos/n_samples)
        fdos_results.append(total_rmse_fdos/n_samples)
        fermi_results.append(total_error_fermi/n_samples)
        energy_results.append(total_error_energy/n_samples)

    if write_path:
        model_vals = [args.batch_size, args.lr, args.l, args.R, args.loss_weight, args.weight_decay, args.scaling, args.identifier]
        write_results_all(write_path, fields, model_vals + dos_results)
        write_results_all(write_path, fields, model_vals + fdos_results)
        write_results_all(write_path, fields, model_vals + fermi_results)
        write_results_all(write_path, fields, model_vals + energy_results)
    else:
        print(",".join([str(x) for x in dos_results]))
        print(",".join([str(x) for x in fdos_results]))
        print(",".join([str(x) for x in fermi_results]))
        print(",".join([str(x) for x in energy_results]))


def evaluate_all(args, model, config_data=None, write_path=None, **kwargs):

    data_path = os.path.join(args.data_dir, args.dataset)
    fp_transform = ConfigurationFingerprint(args.rcut, args.R, args.l, args.m, args.scaling, T=args.T)
    num_workers = args.workers

    data_mode = "test"
    test_data = AtomicConfigurations(data_path, data_mode, data_transform=fp_transform, test_files=config_data, rotate=args.rotate_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
    
    model_fields = ["B", "lr", "L", "R", "w", "wd", "scaling", "trial", "custom"]
    metric_fields = ["dos", "dos std", "fdos", "fdos std", "fermi", "fermi std", "energy", "energy std"]
    fields = model_fields + metric_fields

    dos_results = []
    fdos_results = []
    fermi_results = []
    energy_results = []

    for trial in range(1, args.test_trials+1):
        total_rmse_dos = 0
        total_rmse_fdos = 0
        total_error_fermi = 0
        total_error_energy = 0
        n_samples = 0
        for batch_idx, batch_items in enumerate(test_loader):
            data, mask, target = batch_items[:3]
            config_name = None
            if hasattr(args, "mode") and args.mode == "test":
                config_name = batch_items[4]
            dos_pred, fdos_pred = test_step(model, data, mask, target)
            dos_pred = np.array(dos_pred)
            fdos_pred = np.array(fdos_pred)
            ref_dos = np.array(target[:, :310])
            ref_fdos = np.array(target[:, 310:])

            for dos_pred_inst, fdos_pred_inst, dos_ref_inst, fdos_ref_inst, mask_inst in zip(dos_pred, fdos_pred, ref_dos, ref_fdos, mask):
                rmse_dos = np.sqrt(mean_squared_error(dos_ref_inst, dos_pred_inst))
                total_rmse_dos += rmse_dos
                
                n_atoms = np.sum(mask_inst.data.cpu().numpy())
                fdos_ref_inst *= n_atoms
                fdos_pred_inst *= n_atoms
                rmse_fdos = np.sqrt(mean_squared_error(fdos_ref_inst, fdos_pred_inst))
                total_rmse_fdos += rmse_fdos

                dos_ref_inst *= n_atoms
                dos_pred_inst *= n_atoms
                if args.debug: # to avoid triggering possible out of bounds exception
                    fermi_ref, fermi_pred, energy_ref, energy_pred = 0, 0, 0, 0
                else:
                    fermi_ref, fermi_pred, energy_ref, energy_pred = predict_fermi_and_energy(dos_pred_inst, dos_ref_inst, n_atoms)
                total_error_fermi += abs(fermi_ref-fermi_pred)
                total_error_energy += abs(energy_ref-energy_pred)
                n_samples += 1

        dos_results.append(total_rmse_dos/n_samples)
        fdos_results.append(total_rmse_fdos/n_samples)
        fermi_results.append(total_error_fermi/n_samples)
        energy_results.append(total_error_energy/n_samples)

    all_results = dict()
    all_results["dos"] = np.average(dos_results)
    all_results["dos std"] = np.std(dos_results)
    all_results["fdos"] = np.average(fdos_results)
    all_results["fdos std"] = np.std(fdos_results)
    all_results["fermi"] = np.average(fermi_results)
    all_results["fermi std"] = np.std(fermi_results)
    all_results["energy"] = np.average(energy_results)
    all_results["energy std"] = np.std(energy_results)

    if write_path:
        metric_vals = [str(all_results[k]) for k in metric_fields]
        model_vals = [args.batch_size, args.lr, args.l, args.R, args.loss_weight, args.weight_decay, args.scaling, trial, args.identifier] 
        entries = model_vals + metric_vals
        write_results_all(write_path, fields, entries)

    print(all_results)

    return all_results


def get_test_files_groups(dataset, groups):
    config_paths = dict()
    for group_name in groups:
        match_expr = "Test*{}*".format(group_name)
        config_paths[group_name] = get_files_matching(match_expr, dataset)
    return config_paths


def get_test_files(dataset, groups):
    config_files = []
    for group_name in groups:
        match_expr = "Test*{}*".format(group_name)
        config_files.extend(get_files_matching(match_expr, dataset))
    print("number of files: {}".format(len(config_files)))
    return config_files


def write_results_all(outpath, fields, values):
    """
    Writes header (if file doesn't exist) and results row containing
    both experiment hyperparameters and result metrics.
    """
    try:
        with open(outpath, 'x') as f:
            header_writer = csv.writer(f, delimiter=",")
            header_writer.writerow(fields)
    except FileExistsError:
        print('File already exists. Skipping writing of header')

    with open(outpath, "a+", newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        csvwriter.writerow(values)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to saved model")
    parser.add_argument("-dataset", default="carbon_database", help="Dataset name")
    parser.add_argument("-d", "--data_dir", default=".", help="Data directory")
    parser.add_argument("-o", "--out", default='', help="Name of output directory")
    parser.add_argument("-t", "--test-out", default='', help="Name of test results file")
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
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, help='random seed', default=24)
    parser.add_argument("--test-seed", type=int, help='seed for test evaluation')
    parser.add_argument("--rotate-train", action='store_true')
    parser.add_argument("--rotate-test", action='store_true')
    parser.add_argument("--test-trials", type=int, default=5, help="Number of test trials")
    parser.add_argument('--device', default=0, type=int, help='GPU device ID')
    parser.add_argument("--workers", type=int, default=6, help="Number of processes for data loading")
    parser.add_argument('--mode', default='all', choices=['group', 'all'], help='Evaluation mode')
    parser.add_argument("--debug", action='store_true', help="Debug mode: single batch, single epoch")
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--identifier", help="user-defined string to add to log name")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    icosphere = icosphere(args.l)
    model = model_setup(args, mesh=icosphere)
    if args.test_seed:
        np.random.seed(args.test_seed)
        torch.manual_seed(args.test_seed)
    elif args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    write_path = None
    if args.test_out:
        log_dir = os.path.join(os.environ['DATA_DIR'], "dos_out/logs")
        if args.out:
            log_dir = os.path.join(log_dir, args.out)
        write_path = os.path.join(log_dir, args.test_out)

    config_data = None
    data_path = os.path.join(args.data_dir, args.dataset)
    if args.mode == "group":
        config_data = get_test_files_groups(data_path, groups=CARBON_GROUPS)
    elif args.mode == "all":
        config_data = get_test_files(data_path, groups=CARBON_GROUPS)

    if args.mode == "group":
        groups = CARBON_GROUPS
        evaluate_groups(args, model, groups, config_data=config_data, write_path=write_path)
    elif args.mode == "all": 
        evaluate_all(args, model, config_data=config_data, write_path=write_path)
