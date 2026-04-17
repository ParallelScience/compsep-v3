# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import multiprocessing as mp
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.abspath('codebase'))
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
def process_planck_group(args):
    i_planck, patch_list, i_so_list, W_2D = args
    BASE = '/home/node/data/compsep_data/cut_maps'
    pn_353_full = np.load(BASE + '/planck_noise/planck_noise_353_' + str(i_planck) + '.npy', mmap_mode='r')
    pn_545_full = np.load(BASE + '/planck_noise/planck_noise_545_' + str(i_planck) + '.npy', mmap_mode='r')
    pn_857_full = np.load(BASE + '/planck_noise/planck_noise_857_' + str(i_planck) + '.npy', mmap_mode='r')
    results = []
    for idx, i_patch in enumerate(patch_list):
        i_so = i_so_list[idx]
        sig_90 = np.load(BASE + '/stacked_90.npy', mmap_mode='r')[i_patch]
        sig_150 = np.load(BASE + '/stacked_150.npy', mmap_mode='r')[i_patch]
        sig_217 = np.load(BASE + '/stacked_217.npy', mmap_mode='r')[i_patch]
        sig_353 = np.load(BASE + '/stacked_353.npy', mmap_mode='r')[i_patch]
        sig_545 = np.load(BASE + '/stacked_545.npy', mmap_mode='r')[i_patch]
        sig_857 = np.load(BASE + '/stacked_857.npy', mmap_mode='r')[i_patch]
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
        noise_maps = np.stack([so_90, so_150, so_217, n_353, n_545, n_857])
        obs_fft = np.fft.fft2(obs_maps, axes=(1, 2))
        y_pred_fft = np.sum(W_2D * obs_fft, axis=0)
        y_pred = np.real(np.fft.ifft2(y_pred_fft))
        noise_fft = np.fft.fft2(noise_maps, axes=(1, 2))
        y_noise_fft = np.sum(W_2D * noise_fft, axis=0)
        y_noise = np.real(np.fft.ifft2(y_noise_fft))
        sigma_noise = np.std(y_noise)
        tsz_true = np.load(BASE + '/tsz.npy', mmap_mode='r')[i_patch]
        cib_857_true = np.load(BASE + '/cib_857.npy', mmap_mode='r')[i_patch]
        cib_smoothed = gaussian_filter(cib_857_true, sigma=1.08)
        results.append((i_patch, y_pred, sigma_noise, cib_smoothed, tsz_true))
    return results
if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '16'
    print('Starting Step 2: Bias-Corrected MWF Construction')
    n_patch = 1523
    n_planck = 100
    n_so = 3000
    rng = np.random.default_rng(seed=42)
    i_so_indices = rng.integers(0, n_so, size=n_patch)
    i_planck_indices = rng.integers(0, n_planck, size=n_patch)
    S_ell_obs = np.load('data/S_ell_obs.npy')
    N_ell = np.load('data/N_ell.npy')
    S_ell_tSZ_obs = np.load('data/S_ell_tSZ_obs.npy')
    ell = np.load('data/ell.npy')
    n_ell = len(ell)
    W_ell = np.zeros((6, n_ell))
    epsilon = 1e-5
    print('Computing 1D MWF operator...')
    for k in range(n_ell):
        C = S_ell_obs[:, :, k] + N_ell[:, :, k]
        reg = epsilon * np.diag(np.diag(C))
        if np.all(C == 0):
            continue
        try:
            C_inv = np.linalg.inv(C + reg)
            W_ell[:, k] = C_inv @ S_ell_tSZ_obs[:, k]
        except np.linalg.LinAlgError:
            C_inv = np.linalg.pinv(C + reg)
            W_ell[:, k] = C_inv @ S_ell_tSZ_obs[:, k]
    nx, ny = 256, 256
    fov_deg = 5.0
    dx = fov_deg * np.pi / 180.0 / nx
    dy = fov_deg * np.pi / 180.0 / ny
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    ell_2d = np.sqrt(kx**2 + ky**2)
    W_2D = np.zeros((6, nx, ny))
    for i in range(6):
        W_2D[i] = np.interp(ell_2d, ell, W_ell[i], left=W_ell[i, 0], right=0)
    W_2D[:, 0, 0] = 0
    patches_by_planck = {}
    for i_patch in range(n_patch):
        i_p = i_planck_indices[i_patch]
        if i_p not in patches_by_planck:
            patches_by_planck[i_p] = []
        patches_by_planck[i_p].append(i_patch)
    args_list = []
    for i_planck, patch_list in patches_by_planck.items():
        i_so_list = [i_so_indices[p] for p in patch_list]
        args_list.append((i_planck, patch_list, i_so_list, W_2D))
    print('Processing ' + str(n_patch) + ' patches using multiprocessing...')
    start_time = time.time()
    with mp.Pool(processes=16) as pool:
        results = pool.map(process_planck_group, args_list)
    print('Processing completed in ' + str(round(time.time() - start_time, 2)) + ' seconds.')
    results_flat = [item for sublist in results for item in sublist]
    y_pred_all = np.zeros((n_patch, 256, 256), dtype=np.float32)
    sigma_noise_all = np.zeros(n_patch, dtype=np.float32)
    cib_all = np.zeros((n_patch, 256, 256), dtype=np.float32)
    y_true_all = np.zeros((n_patch, 256, 256), dtype=np.float32)
    for res in results_flat:
        i_patch, y_pred, sigma_noise, cib_smoothed, y_true = res
        y_pred_all[i_patch] = y_pred
        sigma_noise_all[i_patch] = sigma_noise
        cib_all[i_patch] = cib_smoothed
        y_true_all[i_patch] = y_true
    print('Saving reconstructed y-maps...')
    np.save('data/y_pred_mwf.npy', y_pred_all)
    np.save('data/sigma_noise_mwf.npy', sigma_noise_all)
    print('Building multi-dimensional bias-correction lookup table...')
    y_pred_flat = y_pred_all.flatten()
    cib_flat = cib_all.flatten()
    y_true_flat = y_true_all.flatten()
    sigma_noise_flat = np.repeat(sigma_noise_all, 256 * 256)
    y_pred_min = np.min(y_pred_flat)
    y_pred_max = np.max(y_pred_flat)
    cib_min = max(1e-2, np.min(cib_flat))
    cib_max = np.max(cib_flat)
    sigma_noise_min = np.min(sigma_noise_all)
    sigma_noise_max = np.max(sigma_noise_all)
    y_pred_bins = np.linspace(y_pred_min, y_pred_max, 501)
    y_pred_bins[0] = -np.inf
    y_pred_bins[-1] = np.inf
    cib_bins = np.logspace(np.log10(cib_min), np.log10(cib_max), 21)
    cib_bins[0] = -np.inf
    cib_bins[-1] = np.inf
    if sigma_noise_min == sigma_noise_max:
        sigma_noise_bins = np.linspace(sigma_noise_min - 1e-6, sigma_noise_max + 1e-6, 21)
    else:
        sigma_noise_bins = np.linspace(sigma_noise_min, sigma_noise_max, 21)
    sigma_noise_bins[0] = -np.inf
    sigma_noise_bins[-1] = np.inf
    sample = np.vstack([y_pred_flat, sigma_noise_flat, cib_flat]).T
    H_sum, edges = np.histogramdd(sample, bins=[y_pred_bins, sigma_noise_bins, cib_bins], weights=y_true_flat)
    H_count, _ = np.histogramdd(sample, bins=[y_pred_bins, sigma_noise_bins, cib_bins])
    with np.errstate(divide='ignore', invalid='ignore'):
        lookup_table = H_sum / H_count
        lookup_table[H_count == 0] = 0
    np.savez('data/bias_correction_table.npz', lookup_table=lookup_table, H_count=H_count, y_pred_bins=y_pred_bins, sigma_noise_bins=sigma_noise_bins, cib_bins=cib_bins)
    print('Lookup table shape: ' + str(lookup_table.shape))
    print('Step 2 completed successfully. Output shape: ' + str(y_pred_all.shape) + ', saved to data/y_pred_mwf.npy')