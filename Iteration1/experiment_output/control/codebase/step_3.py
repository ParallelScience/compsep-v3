# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('codebase'))
import utils

def process_planck_group_constrained(args):
    i_planck, patch_list, i_so_list, S_ell_obs, N_ell, ell = args
    BASE = '/home/node/data/compsep_data/cut_maps'
    pn_353_full = np.load(BASE + '/planck_noise/planck_noise_353_' + str(i_planck) + '.npy', mmap_mode='r')
    pn_545_full = np.load(BASE + '/planck_noise/planck_noise_545_' + str(i_planck) + '.npy', mmap_mode='r')
    pn_857_full = np.load(BASE + '/planck_noise/planck_noise_857_' + str(i_planck) + '.npy', mmap_mode='r')
    results = []
    nx, ny = 256, 256
    fov_deg = 5.0
    dx = fov_deg * np.pi / 180.0 / nx
    dy = fov_deg * np.pi / 180.0 / ny
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    ell_2d = np.sqrt(kx**2 + ky**2)
    for idx, i_patch in enumerate(patch_list):
        i_so = i_so_list[idx]
        tsz_true = np.load(BASE + '/tsz.npy', mmap_mode='r')[i_patch]
        sig_90 = np.load(BASE + '/stacked_90.npy', mmap_mode='r')[i_patch]
        sig_150 = np.load(BASE + '/stacked_150.npy', mmap_mode='r')[i_patch]
        sig_217 = np.load(BASE + '/stacked_217.npy', mmap_mode='r')[i_patch]
        sig_353 = np.load(BASE + '/stacked_353.npy', mmap_mode='r')[i_patch]
        sig_545 = np.load(BASE + '/stacked_545.npy', mmap_mode='r')[i_patch]
        sig_857 = np.load(BASE + '/stacked_857.npy', mmap_mode='r')[i_patch]
        sigs = [sig_90, sig_150, sig_217, sig_353, sig_545, sig_857]
        P_ell_patch = np.zeros((6, len(ell)))
        for i in range(6):
            _, cl = utils.powers(tsz_true, sigs[i], ps=5, window_alpha=0.5)
            P_ell_patch[i] = cl
        W_ell = np.zeros((6, len(ell)))
        epsilon = 1e-5
        for k in range(len(ell)):
            C = S_ell_obs[:, :, k] + N_ell[:, :, k]
            reg = epsilon * np.diag(np.diag(C))
            if np.all(C == 0):
                continue
            try:
                C_inv = np.linalg.inv(C + reg)
                W_ell[:, k] = C_inv @ P_ell_patch[:, k]
            except np.linalg.LinAlgError:
                C_inv = np.linalg.pinv(C + reg)
                W_ell[:, k] = C_inv @ P_ell_patch[:, k]
        W_2D = np.zeros((6, nx, ny))
        for i in range(6):
            W_2D[i] = np.interp(ell_2d, ell, W_ell[i], left=W_ell[i, 0], right=0)
        W_2D[:, 0, 0] = 0
        so_90 = np.load(BASE + '/so_noise/90.npy', mmap_mode='r')[i_so]
        so_150 = np.load(BASE + '/so_noise/150.npy', mmap_mode='r')[i_so]
        so_217 = np.load(BASE + '/so_noise/217.npy', mmap_mode='r')[i_so]
        n_353 = pn_353_full[i_patch] * 1e6
        n_545 = pn_545_full[i_patch] * 1e6 * utils.jysr2uk(545)
        n_857 = pn_857_full[i_patch] * 1e6 * utils.jysr2uk(857)
        obs_90 = sig_90 + so_90
        obs_150 = sig_150 + so_150
        obs_217 = sig_217 + so_217
        obs_353 = sig_353 + n_353
        obs_545 = sig_545 + n_545
        obs_857 = sig_857 + n_857
        obs_maps = np.stack([obs_90, obs_150, obs_217, obs_353, obs_545, obs_857])
        obs_fft = np.fft.fft2(obs_maps, axes=(1, 2))
        y_pred_fft = np.sum(W_2D * obs_fft, axis=0)
        y_pred = np.real(np.fft.ifft2(y_pred_fft))
        _, cl_pred_true = utils.powers(y_pred, tsz_true, ps=5, window_alpha=0.5)
        _, cl_true_true = utils.powers(tsz_true, tsz_true, ps=5, window_alpha=0.5)
        results.append((i_patch, y_pred, cl_pred_true, cl_true_true, tsz_true))
    return results

def compute_standard_tf(args):
    y_pred_std, tsz_true = args
    _, cl_pred_true = utils.powers(y_pred_std, tsz_true, ps=5, window_alpha=0.5)
    return cl_pred_true

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '16'
    print('Starting Step 3: Constrained MWF for Spatial Fidelity Benchmark')
    n_patch = 1523
    n_planck = 100
    n_so = 3000
    rng = np.random.default_rng(seed=42)
    i_so_indices = rng.integers(0, n_so, size=n_patch)
    i_planck_indices = rng.integers(0, n_planck, size=n_patch)
    S_ell_obs = np.load('data/S_ell_obs.npy')
    N_ell = np.load('data/N_ell.npy')
    ell = np.load('data/ell.npy')
    patches_by_planck = {}
    for i_patch in range(n_patch):
        i_p = i_planck_indices[i_patch]
        if i_p not in patches_by_planck:
            patches_by_planck[i_p] = []
        patches_by_planck[i_p].append(i_patch)
    args_list = []
    for i_planck, patch_list in patches_by_planck.items():
        i_so_list = [i_so_indices[p] for p in patch_list]
        args_list.append((i_planck, patch_list, i_so_list, S_ell_obs, N_ell, ell))
    print('Processing ' + str(n_patch) + ' patches for Constrained MWF using multiprocessing...')
    start_time = time.time()
    with mp.Pool(processes=16) as pool:
        results = pool.map(process_planck_group_constrained, args_list)
    print('Constrained MWF processing completed in ' + str(round(time.time() - start_time, 2)) + ' seconds.')
    results_flat = [item for sublist in results for item in sublist]
    y_pred_constrained_all = np.zeros((n_patch, 256, 256), dtype=np.float32)
    tsz_true_all = np.zeros((n_patch, 256, 256), dtype=np.float32)
    cl_pred_true_constrained_list = []
    cl_true_true_list = []
    for res in results_flat:
        i_patch, y_pred, cl_pred_true, cl_true_true, tsz_true = res
        y_pred_constrained_all[i_patch] = y_pred
        tsz_true_all[i_patch] = tsz_true
        cl_pred_true_constrained_list.append(cl_pred_true)
        cl_true_true_list.append(cl_true_true)
    cl_pred_true_constrained_mean = np.mean(cl_pred_true_constrained_list, axis=0)
    cl_true_true_mean = np.mean(cl_true_true_list, axis=0)
    print('Computing transfer function for Standard MWF...')
    y_pred_std_all = np.load('data/y_pred_mwf.npy')
    args_std = [(y_pred_std_all[i], tsz_true_all[i]) for i in range(n_patch)]
    with mp.Pool(processes=16) as pool:
        cl_pred_true_std_list = pool.map(compute_standard_tf, args_std)
    cl_pred_true_std_mean = np.mean(cl_pred_true_std_list, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        T_ell_constrained = np.where(cl_true_true_mean > 0, cl_pred_true_constrained_mean / cl_true_true_mean, 0)
        T_ell_std = np.where(cl_true_true_mean > 0, cl_pred_true_std_mean / cl_true_true_mean, 0)
    print('Computing pixel-level correlation coefficients...')
    y_pred_constrained_flat = y_pred_constrained_all.flatten()
    tsz_true_flat = tsz_true_all.flatten()
    y_pred_std_flat = y_pred_std_all.flatten()
    corr_constrained = np.corrcoef(y_pred_constrained_flat, tsz_true_flat)[0, 1]
    corr_std = np.corrcoef(y_pred_std_flat, tsz_true_flat)[0, 1]
    print('--- Benchmark Metrics ---')
    print('Global Pixel Correlation (Constrained MWF): ' + str(round(corr_constrained, 4)))
    print('Global Pixel Correlation (Standard MWF):    ' + str(round(corr_std, 4)))
    print('Mean T(ell) [ell < 3000] (Constrained MWF): ' + str(round(np.mean(T_ell_constrained[ell < 3000]), 4)))
    print('Mean T(ell) [ell < 3000] (Standard MWF):    ' + str(round(np.mean(T_ell_std[ell < 3000]), 4)))
    print('-------------------------')
    np.save('data/y_pred_constrained.npy', y_pred_constrained_all)
    np.savez('data/transfer_functions.npz', ell=ell, T_ell_constrained=T_ell_constrained, T_ell_std=T_ell_std)
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    plt.plot(ell[1:], T_ell_constrained[1:], label='Constrained MWF (Upper Limit)', color='red', linewidth=2)
    plt.plot(ell[1:], T_ell_std[1:], label='Standard MWF (Blind)', color='blue', linewidth=2, linestyle='--')
    plt.axhline(1.0, color='black', linestyle=':', alpha=0.7)
    plt.xlabel('Multipole ell')
    plt.ylabel('Transfer Function T(ell)')
    plt.title('Spatial Fidelity Benchmark: Transfer Function')
    plt.xscale('log')
    plt.xlim(100, 10000)
    plt.ylim(-0.1, 1.2)
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_path = 'data/transfer_function_benchmark_' + str(timestamp) + '.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print('Transfer function plot saved to ' + plot_path)
    print('Step 3 completed successfully. Output shape: ' + str(y_pred_constrained_all.shape) + ', saved to data/y_pred_constrained.npy')