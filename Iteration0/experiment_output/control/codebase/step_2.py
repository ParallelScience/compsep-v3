# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, '/home/node/data/compsep_data')
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing as mp
import utils

def get_planck_noise(freq, i_planck, i_patch, base_path):
    filepath = base_path + '/planck_noise/planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy'
    data = np.load(filepath, mmap_mode='r')
    return data[i_patch]

def process_patch(args):
    i_patch, i_so, i_planck, base_dir = args
    s_90 = np.load(base_dir + '/stacked_90.npy', mmap_mode='r')[i_patch]
    s_150 = np.load(base_dir + '/stacked_150.npy', mmap_mode='r')[i_patch]
    s_217 = np.load(base_dir + '/stacked_217.npy', mmap_mode='r')[i_patch]
    s_353 = np.load(base_dir + '/stacked_353.npy', mmap_mode='r')[i_patch]
    s_545 = np.load(base_dir + '/stacked_545.npy', mmap_mode='r')[i_patch]
    s_857 = np.load(base_dir + '/stacked_857.npy', mmap_mode='r')[i_patch]
    tsz = np.load(base_dir + '/tsz.npy', mmap_mode='r')[i_patch]
    signals = [s_90, s_150, s_217, s_353, s_545, s_857, tsz]
    n_90 = np.load(base_dir + '/so_noise/90.npy', mmap_mode='r')[i_so]
    n_150 = np.load(base_dir + '/so_noise/150.npy', mmap_mode='r')[i_so]
    n_217 = np.load(base_dir + '/so_noise/217.npy', mmap_mode='r')[i_so]
    pn_353_raw = get_planck_noise(353, i_planck, i_patch, base_dir)
    pn_545_raw = get_planck_noise(545, i_planck, i_patch, base_dir)
    pn_857_raw = get_planck_noise(857, i_planck, i_patch, base_dir)
    n_353 = pn_353_raw * 1e6
    n_545 = pn_545_raw * 1e6 * utils.jysr2uk(545)
    n_857 = pn_857_raw * 1e6 * utils.jysr2uk(857)
    noises = [n_90, n_150, n_217, n_353, n_545, n_857]
    ps_val = 300.0 / 256.0
    cl, ell = utils.powers(signals[0], signals[0], ps=ps_val, window_alpha=0.5)
    n_ell = len(ell)
    S = np.zeros((7, 7, n_ell))
    for i in range(7):
        for j in range(i, 7):
            cl, _ = utils.powers(signals[i], signals[j], ps=ps_val, window_alpha=0.5)
            S[i, j] = cl
            if i != j:
                S[j, i] = cl
    N = np.zeros((6, 6, n_ell))
    for i in range(6):
        for j in range(i, 6):
            if i == j or (i == 0 and j == 1):
                cl, _ = utils.powers(noises[i], noises[j], ps=ps_val, window_alpha=0.5)
                N[i, j] = cl
                if i != j:
                    N[j, i] = cl
    return S, N, ell

def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    base_dir = '/home/node/data/compsep_data/cut_maps'
    n_patch_total = 1523
    n_patches_to_use = 500
    rng = np.random.default_rng(seed=42)
    patch_indices = rng.choice(n_patch_total, size=n_patches_to_use, replace=False)
    i_so_indices = rng.integers(0, 3000, size=n_patch_total)
    i_planck_indices = rng.integers(0, 100, size=n_patch_total)
    args_list = [(p, i_so_indices[p], i_planck_indices[p], base_dir) for p in patch_indices]
    num_workers = 16
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(process_patch, args_list)
    ell = results[0][2]
    n_ell = len(ell)
    S_sum = np.zeros((7, 7, n_ell))
    N_sum = np.zeros((6, 6, n_ell))
    for S, N, _ in results:
        S_sum += S
        N_sum += N
    S_mean = S_sum / n_patches_to_use
    N_mean = N_sum / n_patches_to_use
    sort_idx = np.argsort(ell)
    ell = ell[sort_idx]
    S_mean = S_mean[:, :, sort_idx]
    N_mean = N_mean[:, :, sort_idx]
    C_total = np.zeros((7, 7, n_ell))
    C_total[:, :, :] = S_mean
    C_total[:6, :6, :] += N_mean
    C_obs = C_total[:6, :6, :]
    C_cross = C_total[:6, 6, :]
    epsilon = 1e-5
    W = np.zeros((6, n_ell))
    for k in range(n_ell):
        C_k = C_obs[:, :, k]
        reg_term = epsilon * np.diag(np.diag(C_k))
        C_k_reg = C_k + reg_term
        d = np.sqrt(np.diag(C_k_reg))
        D_inv = np.diag(1.0 / d)
        R_k = D_inv @ C_k_reg @ D_inv
        inv_R_k = np.linalg.inv(R_k)
        inv_C_k = D_inv @ inv_R_k @ D_inv
        W[:, k] = inv_C_k @ C_cross[:, k]
    out_filepath = 'data/mwf_weights.npz'
    np.savez(out_filepath, W=W, ell=ell, C_total=C_total, S_mean=S_mean, N_mean=N_mean)
    plt.rcParams['text.usetex'] = False
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    channels = ['90 GHz', '150 GHz', '217 GHz', '353 GHz', '545 GHz', '857 GHz', 'tSZ']
    for i in range(6):
        axs[0].plot(ell, C_total[i, i, :], label=channels[i])
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].set_xlabel('Multipole ell')
    axs[0].set_ylabel('Power Spectrum C_ell [µK_CMB^2]')
    axs[0].set_title('Frequency Auto-Spectra')
    axs[0].legend(fontsize=8)
    axs[0].grid(True, which='both', ls='--', alpha=0.5)
    pairs_freq = [(0, 1, '90 x 150'), (1, 2, '150 x 217'), (3, 5, '353 x 857')]
    for i, j, label in pairs_freq:
        cross = C_total[i, j, :]
        axs[1].plot(ell, np.abs(cross), label='|' + label + '|')
    axs[1].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].set_xlabel('Multipole ell')
    axs[1].set_ylabel('Absolute Power Spectrum |C_ell| [µK_CMB^2]')
    axs[1].set_title('Frequency Cross-Spectra')
    axs[1].legend(fontsize=8)
    axs[1].grid(True, which='both', ls='--', alpha=0.5)
    axs[2].plot(ell, C_total[6, 6, :], label='tSZ Auto (dimensionless)', color='black', linewidth=2)
    pairs_tsz = [(0, 6, '90 x tSZ'), (1, 6, '150 x tSZ'), (5, 6, '857 x tSZ')]
    for i, j, label in pairs_tsz:
        cross = C_total[i, j, :]
        axs[2].plot(ell, np.abs(cross), label='|' + label + '| (µK_CMB)')
    axs[2].set_yscale('log')
    axs[2].set_xscale('log')
    axs[2].set_xlabel('Multipole ell')
    axs[2].set_ylabel('Absolute Power Spectrum')
    axs[2].set_title('tSZ Auto and Cross-Spectra')
    axs[2].legend(fontsize=8)
    axs[2].grid(True, which='both', ls='--', alpha=0.5)
    fig.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = 'data/covariance_spectra_2_' + timestamp + '.png'
    fig.savefig(plot_filename, dpi=300)
    print('Diagnostic plot saved to ' + plot_filename)
    print('Step 2 completed successfully. Output shape: ' + str(W.shape) + ', saved to ' + out_filepath)

if __name__ == '__main__':
    main()