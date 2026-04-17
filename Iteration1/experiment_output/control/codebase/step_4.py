# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, '/home/node/data/compsep_data/')
import time
import numpy as np
import multiprocessing as mp
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils

def process_planck_group_noise(args):
    i_planck, patch_list, i_so_list, W_2D = args
    BASE = '/home/node/data/compsep_data/cut_maps'
    pn_353_full = np.load(BASE + '/planck_noise/planck_noise_353_' + str(i_planck) + '.npy', mmap_mode='r')
    pn_545_full = np.load(BASE + '/planck_noise/planck_noise_545_' + str(i_planck) + '.npy', mmap_mode='r')
    pn_857_full = np.load(BASE + '/planck_noise/planck_noise_857_' + str(i_planck) + '.npy', mmap_mode='r')
    results = []
    for idx, i_patch in enumerate(patch_list):
        i_so = i_so_list[idx]
        so_90 = np.load(BASE + '/so_noise/90.npy', mmap_mode='r')[i_so]
        so_150 = np.load(BASE + '/so_noise/150.npy', mmap_mode='r')[i_so]
        so_217 = np.load(BASE + '/so_noise/217.npy', mmap_mode='r')[i_so]
        n_353 = pn_353_full[i_patch] * 1e6
        n_545 = pn_545_full[i_patch] * 1e6 * utils.jysr2uk(545)
        n_857 = pn_857_full[i_patch] * 1e6 * utils.jysr2uk(857)
        noise_maps = np.stack([so_90, so_150, so_217, n_353, n_545, n_857])
        noise_fft = np.fft.fft2(noise_maps, axes=(1, 2))
        y_noise_fft = np.sum(W_2D * noise_fft, axis=0)
        y_noise = np.real(np.fft.ifft2(y_noise_fft))
        sigma_noise = np.std(y_noise)
        local_max = ndimage.maximum_filter(y_noise, size=5) == y_noise
        peaks = y_noise[local_max & (y_noise > 0)]
        snr_peaks = peaks / sigma_noise
        results.append(snr_peaks)
    return results

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '16'
    print('Starting Step 4: Empirical False Detection Rate (FDR) Calibration')
    n_patch_total = 1523
    n_use = 500
    n_planck = 100
    n_so = 3000
    rng = np.random.default_rng(seed=42)
    i_so_indices = rng.integers(0, n_so, size=n_patch_total)
    i_planck_indices = rng.integers(0, n_planck, size=n_patch_total)
    S_ell_obs = np.load('data/S_ell_obs.npy')
    N_ell = np.load('data/N_ell.npy')
    S_ell_tSZ_obs = np.load('data/S_ell_tSZ_obs.npy')
    ell = np.load('data/ell.npy')
    n_ell = len(ell)
    W_ell = np.zeros((6, n_ell))
    epsilon = 1e-5
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
    for i_patch in range(n_use):
        i_p = i_planck_indices[i_patch]
        if i_p not in patches_by_planck:
            patches_by_planck[i_p] = []
        patches_by_planck[i_p].append(i_patch)
    args_list = []
    for i_planck, patch_list in patches_by_planck.items():
        i_so_list = [i_so_indices[p] for p in patch_list]
        args_list.append((i_planck, patch_list, i_so_list, W_2D))
    print('Processing ' + str(n_use) + ' pure noise patches using multiprocessing...')
    start_time = time.time()
    with mp.Pool(processes=16) as pool:
        results = pool.map(process_planck_group_noise, args_list)
    print('Processing completed in ' + str(round(time.time() - start_time, 2)) + ' seconds.')
    results_flat = [item for sublist in results for item in sublist]
    all_snr_peaks = np.concatenate(results_flat)
    if len(all_snr_peaks) > 0:
        max_snr = np.max(all_snr_peaks)
    else:
        max_snr = 0.0
    print('Maximum SNR found in pure noise patches: ' + str(round(max_snr, 2)))
    snr_thresholds = np.linspace(3.0, 8.0, 501)
    expected_false_detections = np.zeros_like(snr_thresholds)
    for i, th in enumerate(snr_thresholds):
        count = np.sum(all_snr_peaks > th)
        expected_false_detections[i] = count * (n_patch_total / n_use)
    target_fdrs = [100, 50, 10, 5, 1]
    print('--- False Detection Rate Calibration ---')
    for target in target_fdrs:
        idx = np.where(expected_false_detections <= target)[0]
        if len(idx) > 0:
            th = snr_thresholds[idx[0]]
            print('SNR Threshold for <= ' + str(target) + ' false detections across ' + str(n_patch_total) + ' patches: ' + str(round(th, 2)))
        else:
            print('SNR Threshold for <= ' + str(target) + ' false detections across ' + str(n_patch_total) + ' patches: > 8.0')
    idx_5 = np.where(expected_false_detections <= 5)[0]
    if len(idx_5) > 0:
        chosen_snr = snr_thresholds[idx_5[0]]
    else:
        chosen_snr = 5.0
    print('Chosen SNR threshold for subsequent steps: ' + str(round(chosen_snr, 2)))
    np.savez('data/fdr_calibration.npz', snr_thresholds=snr_thresholds, expected_false_detections=expected_false_detections, chosen_snr=chosen_snr)
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    valid = expected_false_detections > 0
    if np.any(valid):
        plt.plot(snr_thresholds[valid], expected_false_detections[valid], color='blue', linewidth=2)
    plt.axhline(5.0, color='red', linestyle='--', label='5 False Detections')
    plt.axvline(chosen_snr, color='green', linestyle=':', label='Chosen SNR: ' + str(round(chosen_snr, 2)))
    plt.yscale('log')
    plt.xlabel('SNR Threshold')
    plt.ylabel('Expected False Detections (per 1523 patches)')
    plt.title('Empirical False Detection Rate (FDR) Calibration')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    timestamp = int(time.time())
    plot_path = 'data/fdr_calibration_' + str(timestamp) + '.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print('FDR curve plot saved to ' + plot_path)
    print('Step 4 completed successfully. Output shape: ' + str(expected_false_detections.shape) + ', saved to data/fdr_calibration.npz')