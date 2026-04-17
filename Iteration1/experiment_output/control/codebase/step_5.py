# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import multiprocessing as mp
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils

def recompute_patch(args):
    i_patch, i_so, i_p, W_2D, f_tSZ = args
    BASE = '/home/node/data/compsep_data/cut_maps'
    sig_90 = np.load(BASE + '/stacked_90.npy', mmap_mode='r')[i_patch]
    sig_150 = np.load(BASE + '/stacked_150.npy', mmap_mode='r')[i_patch]
    sig_217 = np.load(BASE + '/stacked_217.npy', mmap_mode='r')[i_patch]
    sig_353 = np.load(BASE + '/stacked_353.npy', mmap_mode='r')[i_patch]
    sig_545 = np.load(BASE + '/stacked_545.npy', mmap_mode='r')[i_patch]
    sig_857 = np.load(BASE + '/stacked_857.npy', mmap_mode='r')[i_patch]
    so_90 = np.load(BASE + '/so_noise/90.npy', mmap_mode='r')[i_so]
    so_150 = np.load(BASE + '/so_noise/150.npy', mmap_mode='r')[i_so]
    so_217 = np.load(BASE + '/so_noise/217.npy', mmap_mode='r')[i_so]
    pn_353 = np.load(BASE + '/planck_noise/planck_noise_353_' + str(i_p) + '.npy', mmap_mode='r')[i_patch]
    pn_545 = np.load(BASE + '/planck_noise/planck_noise_545_' + str(i_p) + '.npy', mmap_mode='r')[i_patch]
    pn_857 = np.load(BASE + '/planck_noise/planck_noise_857_' + str(i_p) + '.npy', mmap_mode='r')[i_patch]
    obs_maps = np.stack([sig_90 + so_90, sig_150 + so_150, sig_217 + so_217, sig_353 + pn_353 * 1e6, sig_545 + pn_545 * 1e6 * utils.jysr2uk(545), sig_857 + pn_857 * 1e6 * utils.jysr2uk(857)])
    obs_fft = np.fft.fft2(obs_maps, axes=(1, 2))
    y_pred = np.real(np.fft.ifft2(np.sum(W_2D * obs_fft, axis=0)))
    tsz_map = np.load(BASE + '/tsz.npy', mmap_mode='r')[i_patch]
    obs_no_tsz = np.stack([obs_maps[i] - tsz_map * f_tSZ[i] for i in range(6)])
    y_null = np.real(np.fft.ifft2(np.sum(W_2D * np.fft.fft2(obs_no_tsz, axes=(1, 2)), axis=0)))
    return i_patch, y_pred, np.median(np.abs(y_null - np.median(y_null))) / 0.6745, y_null

def get_all_truth_peaks_worker(args):
    tsz_map, i_p = args
    local_max = ndimage.maximum_filter(tsz_map, size=5) == tsz_map
    mask = local_max & (tsz_map > 1e-6)
    y_idx, x_idx = np.where(mask)
    return [(x, y, tsz_map[y, x], i_p) for x, y in zip(x_idx, y_idx)]

def get_all_detected_peaks_worker(args):
    y_map, sigma, snr_th, i_p = args
    if sigma <= 0 or np.isnan(sigma): return []
    local_max = ndimage.maximum_filter(y_map, size=5) == y_map
    mask = local_max & (y_map > snr_th * sigma)
    y_idx, x_idx = np.where(mask)
    return [(x, y, y_map[y, x] / sigma, i_p) for x, y in zip(x_idx, y_idx)]

def get_all_detected_peaks_filt_worker(args):
    y_map, filt_fft, snr_th, i_p = args
    y_filt = np.real(np.fft.ifft2(np.fft.fft2(y_map) * filt_fft))
    sigma = np.median(np.abs(y_filt - np.median(y_filt))) / 0.6745
    if sigma <= 0 or np.isnan(sigma): return [], sigma, i_p
    local_max = ndimage.maximum_filter(y_filt, size=5) == y_filt
    mask = local_max & (y_filt > snr_th * sigma)
    y_idx, x_idx = np.where(mask)
    return [(x, y, y_filt[y, x] / sigma, i_p) for x, y in zip(x_idx, y_idx)], sigma, i_p

def cross_match_all_idx(detected_peaks, truth_peaks, dist_th_pixels=2.13):
    from collections import defaultdict
    det_by_patch, truth_by_patch = defaultdict(list), defaultdict(list)
    for i, d in enumerate(detected_peaks): det_by_patch[d[3]].append((i, d))
    for i, t in enumerate(truth_peaks): truth_by_patch[t[3]].append((i, t))
    tp_det_idx, tp_truth_idx, fp_det_idx, matched_truth_indices = [], [], [], set()
    for i_p in range(1523):
        dets = sorted(det_by_patch[i_p], key=lambda x: x[1][2], reverse=True)
        truths = truth_by_patch[i_p]
        for det_idx, det in dets:
            x_d, y_d, snr, _ = det
            best_dist, best_truth_idx = float('inf'), -1
            for truth_idx, t in truths:
                if truth_idx in matched_truth_indices: continue
                dist = np.sqrt((x_d - t[1][0])**2 + (y_d - t[1][1])**2)
                if dist < best_dist: best_dist, best_truth_idx = dist, truth_idx
            if best_dist <= dist_th_pixels:
                tp_det_idx.append(det_idx); tp_truth_idx.append(best_truth_idx); matched_truth_indices.add(best_truth_idx)
            else: fp_det_idx.append(det_idx)
    return tp_det_idx, tp_truth_idx, fp_det_idx, [i for i in range(len(truth_peaks)) if i not in matched_truth_indices]

def evaluate_performance(detected_peaks, truth_peaks, cib_maps, sigma_noise_mwf, dist_th_pixels=2.13):
    tp_det_idx, tp_truth_idx, fp_det_idx, fn_truth_idx = cross_match_all_idx(detected_peaks, truth_peaks, dist_th_pixels)
    truth_mass = np.array([t[2] for t in truth_peaks])
    truth_cib = np.array([cib_maps[t[3], t[1], t[0]] for t in truth_peaks])
    truth_noise = np.array([sigma_noise_mwf[t[3]] for t in truth_peaks])
    truth_matched_bool = np.zeros(len(truth_peaks), dtype=bool)
    if len(tp_truth_idx) > 0: truth_matched_bool[tp_truth_idx] = True
    det_snr = np.array([d[2] for d in detected_peaks])
    det_cib = np.array([cib_maps[d[3], d[1], d[0]] for d in detected_peaks]) if len(detected_peaks) > 0 else np.array([])
    det_noise = np.array([sigma_noise_mwf[d[3]] for d in detected_peaks]) if len(detected_peaks) > 0 else np.array([])
    det_matched_bool = np.zeros(len(detected_peaks), dtype=bool)
    if len(tp_det_idx) > 0: det_matched_bool[tp_det_idx] = True
    return {'truth_mass': truth_mass, 'truth_cib': truth_cib, 'truth_noise': truth_noise, 'truth_matched': truth_matched_bool, 'det_snr': det_snr, 'det_cib': det_cib, 'det_noise': det_noise, 'det_matched': det_matched_bool, 'tp_count': len(tp_det_idx), 'fp_count': len(fp_det_idx), 'fn_count': len(fn_truth_idx)}

def compute_binned_stats(values, matches, bins):
    if len(values) == 0: return np.zeros(len(bins)-1), np.zeros(len(bins)-1)
    vals_clipped = np.clip(values, bins[0], bins[-1])
    counts, _ = np.histogram(vals_clipped, bins=bins)
    matched_counts, _ = np.histogram(vals_clipped[matches], bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'): frac = np.where(counts > 0, matched_counts / counts, 0)
    return frac, counts

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '16'
    BASE = '/home/node/data/compsep_data/cut_maps'
    n_patch = 1523
    S_ell_obs, N_ell, ell = np.load('data/S_ell_obs.npy'), np.load('data/N_ell.npy'), np.load('data/ell.npy')
    S_ell_obs, N_ell = np.nan_to_num(S_ell_obs), np.nan_to_num(N_ell)
    sort_idx = np.argsort(ell)
    ell_sorted, S_ell_obs, N_ell = ell[sort_idx], S_ell_obs[:, :, sort_idx], N_ell[:, :, sort_idx]
    freqs = [90, 150, 217, 353, 545, 857]
    f_tSZ = np.array([utils.tsz(f) for f in freqs])
    tsz_maps, cib_353, cib_545, cib_857 = np.load(BASE + '/tsz.npy')[:100], np.load(BASE + '/cib_353.npy')[:100], np.load(BASE + '/cib_545.npy')[:100], np.load(BASE + '/cib_857.npy')[:100]
    P_ell_tSZ = np.mean([utils.powers(tsz_maps[i], tsz_maps[i], ps=5, window_alpha=0.5)[1] for i in range(100)], axis=0)[sort_idx]
    S_ell_tSZ_obs = np.vstack([f_tSZ[i] * P_ell_tSZ for i in range(6)])
    W_ell = np.zeros((6, len(ell_sorted)))
    for k in range(len(ell_sorted)):
        C = S_ell_obs[:, :, k] + N_ell[:, :, k]
        diag_C = np.diag(C)
        if np.all(diag_C == 0): continue
        reg = 1e-4 * np.diag(np.maximum(diag_C, np.max(diag_C)*1e-6))
        try: W_ell[:, k] = np.linalg.solve(C + reg, S_ell_tSZ_obs[:, k])
        except np.linalg.LinAlgError: W_ell[:, k] = np.linalg.pinv(C + reg) @ S_ell_tSZ_obs[:, k]
    W_ell = np.nan_to_num(W_ell)
    nx, ny, fov_deg = 256, 256, 5.0
    dx, dy = fov_deg * np.pi / 180.0 / nx, fov_deg * np.pi / 180.0 / ny
    kx, ky = np.meshgrid(np.fft.fftfreq(nx, d=dx) * 2 * np.pi, np.fft.fftfreq(ny, d=dy) * 2 * np.pi)
    ell_2d = np.sqrt(kx**2 + ky**2)
    true_ps = fov_deg * 60.0 / nx
    ell_corrected = ell_sorted * (5.0 / true_ps)
    W_2D = np.zeros((6, nx, ny))
    for i in range(6): W_2D[i] = np.interp(ell_2d, ell_corrected, W_ell[i], left=W_ell[i, 0], right=0)
    W_2D[:, 0, 0] = 0
    rng = np.random.default_rng(seed=42)
    args_list = [(i, rng.integers(0, 3000), rng.integers(0, 100), W_2D, f_tSZ) for i in range(n_patch)]
    with mp.Pool(processes=16) as pool: results = pool.map(recompute_patch, args_list)
    y_pred_mwf, sigma_noise_mwf = np.zeros((n_patch, 256, 256), dtype=np.float32), np.zeros(n_patch, dtype=np.float32)
    for res in results: y_pred_mwf[res[0]], sigma_noise_mwf[res[0]] = res[1], res[2]
    np.save('data/y_pred_mwf.npy', y_pred_mwf)
    np.save('data/sigma_noise_mwf.npy', sigma_noise_mwf)
    chosen_snr = 4.0
    tsz, cib_857 = np.load(BASE + '/tsz.npy'), np.load(BASE + '/cib_857.npy')
    y_pred_flat, cib_flat, y_true_flat = y_pred_mwf.flatten(), cib_857.flatten(), tsz.flatten()
    sigma_noise_flat = np.repeat(sigma_noise_mwf, 256 * 256)
    y_pred_bins_bc = np.linspace(np.min(y_pred_flat), np.max(y_pred_flat), 501)
    y_pred_bins_bc[0], y_pred_bins_bc[-1] = -np.inf, np.inf
    cib_bins_bc = np.logspace(np.log10(max(1e-2, np.min(cib_flat))), np.log10(np.max(cib_flat)), 21)
    cib_bins_bc[0], cib_bins_bc[-1] = -np.inf, np.inf
    sigma_noise_bins_bc = np.linspace(np.min(sigma_noise_mwf), np.max(sigma_noise_mwf), 21)
    sigma_noise_bins_bc[0], sigma_noise_bins_bc[-1] = -np.inf, np.inf
    sample = np.vstack([y_pred_flat, sigma_noise_flat, cib_flat]).T
    H_sum, _ = np.histogramdd(sample, bins=[y_pred_bins_bc, sigma_noise_bins_bc, cib_bins_bc], weights=y_true_flat)
    H_count, _ = np.histogramdd(sample, bins=[y_pred_bins_bc, sigma_noise_bins_bc, cib_bins_bc])
    np.savez('data/bias_correction_table.npz', lookup_table=np.divide(H_sum, H_count, out=np.zeros_like(H_sum), where=H_count != 0), H_count=H_count, y_pred_bins=y_pred_bins_bc, sigma_noise_bins=sigma_noise_bins_bc, cib_bins=cib_bins_bc)
    with mp.Pool(processes=16) as pool:
        truth_peaks = [item for sublist in pool.map(get_all_truth_peaks_worker, [(tsz[i], i) for i in range(n_patch)]) for item in sublist]
        detected_peaks_std = [item for sublist in pool.map(get_all_detected_peaks_worker, [(y_pred_mwf[i], sigma_noise_mwf[i], chosen_snr, i) for i in range(n_patch)]) for item in sublist]
    x, y = np.meshgrid(np.arange(-nx//2, nx//2) * true_ps, np.arange(-ny//2, ny//2) * true_ps)
    R = np.sqrt(x**2 + y**2)
    profile = np.sum([(1.0 / ((np.maximum(1.177 * (np.sqrt(R**2 + l_val**2) / 2.0), 1e-4))**0.3081 * (1 + np.maximum(1.177 * (np.sqrt(R**2 + l_val**2) / 2.0), 1e-4)**1.051)**((5.4905 - 0.3081)/1.051))) * (20.0 / 200) for l_val in np.linspace(0, 10 * 2.0, 200)], axis=0)
    profile_smoothed = ndimage.gaussian_filter(profile / np.max(profile), sigma=(1.4 / np.sqrt(8 * np.log(2))) / true_ps)
    tau_fft = np.fft.fft2(np.fft.ifftshift(profile_smoothed))
    P_2D = np.mean(np.abs(np.fft.fft2(y_pred_mwf))**2, axis=0)
    P_2D_smoothed = np.fft.ifftshift(ndimage.gaussian_filter(np.fft.fftshift(P_2D), sigma=2.0))
    filt_fft = np.conj(tau_fft) / np.maximum(P_2D_smoothed, np.max(P_2D_smoothed)*1e-10)
    filt_fft /= (np.sum(filt_fft * tau_fft).real / (nx * ny))
    with mp.Pool(processes=16) as pool:
        det_filt_results = pool.map(get_all_detected_peaks_filt_worker, [(y_pred_mwf[i], filt_fft, chosen_snr, i) for i in range(n_patch)])
    detected_peaks_filt, sigma_noise_filt = [], np.zeros(n_patch)
    for res in det_filt_results:
        detected_peaks_filt.extend(res[0])
        sigma_noise_filt[res[2]] = res[1]
    dist_th_pixels = 2.5 / true_ps
    perf_std = evaluate_performance(detected_peaks_std, truth_peaks, cib_857, sigma_noise_mwf, dist_th_pixels)
    perf_filt = evaluate_performance(detected_peaks_filt, truth_peaks, cib_857, sigma_noise_filt, dist_th_pixels)
    cib_bins = np.logspace(np.log10(max(1e-4, np.percentile(cib_857/1e6, 1))), np.log10(np.percentile(cib_857/1e6, 99.9)), 21)
    noise_bins = np.linspace(np.min(sigma_noise_mwf), np.max(sigma_noise_mwf), 21)
    snr_bins, mass_bins = np.linspace(chosen_snr, 15.0, 21), np.logspace(np.log10(1e-6), np.log10(1e-4), 21)
    comp_mass_std, counts_mass = compute_binned_stats(perf_std['truth_mass'], perf_std['truth_matched'], mass_bins)
    comp_mass_filt, _ = compute_binned_stats(perf_filt['truth_mass'], perf_filt['truth_matched'], mass_bins)
    comp_cib_std, counts_cib = compute_binned_stats(perf_std['truth_cib']/1e6, perf_std['truth_matched'], cib_bins)
    comp_cib_filt, _ = compute_binned_stats(perf_filt['truth_cib']/1e6, perf_filt['truth_matched'], cib_bins)
    comp_noise_std, counts_noise = compute_binned_stats(perf_std['truth_noise'], perf_std['truth_matched'], noise_bins)
    comp_noise_filt, _ = compute_binned_stats(perf_filt['truth_noise'], perf_filt['truth_matched'], noise_bins)
    pur_snr_std, counts_snr_std = compute_binned_stats(perf_std['det_snr'], perf_std['det_matched'], snr_bins)
    pur_snr_filt, counts_snr_filt = compute_binned_stats(perf_filt['det_snr'], perf_filt['det_matched'], snr_bins)
    pur_cib_std, counts_det_cib_std = compute_binned_stats(perf_std['det_cib']/1e6, perf_std['det_matched'], cib_bins)
    pur_cib_filt, counts_det_cib_filt = compute_binned_stats(perf_filt['det_cib']/1e6, perf_filt['det_matched'], cib_bins)
    pur_noise_std, counts_det_noise_std = compute_binned_stats(perf_std['det_noise'], perf_std['det_matched'], noise_bins)
    pur_noise_filt, counts_det_noise_filt = compute_binned_stats(perf_filt['det_noise'], perf_filt['det_matched'], noise_bins)
    np.savez('data/detection_performance.npz', mass_bins=mass_bins, cib_bins=cib_bins, noise_bins=noise_bins, snr_bins=snr_bins, comp_mass_std=comp_mass_std, comp_mass_filt=comp_mass_filt, counts_mass=counts_mass, comp_cib_std=comp_cib_std, comp_cib_filt=comp_cib_filt, counts_cib=counts_cib, comp_noise_std=comp_noise_std, comp_noise_filt=comp_noise_filt, counts_noise=counts_noise, pur_snr_std=pur_snr_std, pur_snr_filt=pur_snr_filt, counts_snr_std=counts_snr_std, counts_snr_filt=counts_snr_filt, pur_cib_std=pur_cib_std, pur_cib_filt=pur_cib_filt, counts_det_cib_std=counts_det_cib_std, counts_det_cib_filt=counts_det_cib_filt, pur_noise_std=pur_noise_std, pur_noise_filt=pur_noise_filt, counts_det_noise_std=counts_det_noise_std, counts_det_noise_filt=counts_det_noise_filt)
    np.save('data/detected_peaks_std.npy', np.array(detected_peaks_std) if len(detected_peaks_std) > 0 else np.empty((0, 4)))
    np.save('data/detected_peaks_filt.npy', np.array(detected_peaks_filt) if len(detected_peaks_filt) > 0 else np.empty((0, 4)))
    np.save('data/truth_peaks.npy', np.array(truth_peaks) if len(truth_peaks) > 0 else np.empty((0, 4)))
    np.savez('data/matched_indices_std.npz', det_idx=cross_match_all_idx(detected_peaks_std, truth_peaks, dist_th_pixels)[0], truth_idx=cross_match_all_idx(detected_peaks_std, truth_peaks, dist_th_pixels)[1])
    np.savez('data/matched_indices_filt.npz', det_idx=cross_match_all_idx(detected_peaks_filt, truth_peaks, dist_th_pixels)[0], truth_idx=cross_match_all_idx(detected_peaks_filt, truth_peaks, dist_th_pixels)[1])
    plt.rcParams['text.usetex'] = False
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    mass_centers = np.sqrt(mass_bins[:-1] * mass_bins[1:])
    axs[0, 0].plot(mass_centers, comp_mass_std, label='Standard ILC', marker='o')
    axs[0, 0].plot(mass_centers, comp_mass_filt, label='Filtered MMF', marker='s')
    axs[0, 0].set_xscale('log'); axs[0, 0].set_xlabel('Halo Mass Proxy (Peak Compton-y)'); axs[0, 0].set_ylabel('Completeness'); axs[0, 0].set_title('Completeness vs Mass'); axs[0, 0].legend(); axs[0, 0].grid(True, which='both', ls='--', alpha=0.5)
    cib_centers = np.sqrt(cib_bins[:-1] * cib_bins[1:])
    axs[0, 1].plot(cib_centers, comp_cib_std, label='Standard ILC', marker='o')
    axs[0, 1].plot(cib_centers, comp_cib_filt, label='Filtered MMF', marker='s')
    axs[0, 1].set_xscale('log'); axs[0, 1].set_xlabel('Local CIB Intensity at 857 GHz [MJy/sr]'); axs[0, 1].set_ylabel('Completeness'); axs[0, 1].set_title('Completeness vs CIB Intensity'); axs[0, 1].legend(); axs[0, 1].grid(True, which='both', ls='--', alpha=0.5)
    noise_centers = (noise_bins[:-1] + noise_bins[1:]) / 2
    axs[0, 2].plot(noise_centers, comp_noise_std, label='Standard ILC', marker='o')
    axs[0, 2].plot(noise_centers, comp_noise_filt, label='Filtered MMF', marker='s')
    axs[0, 2].set_xlabel('Local Noise Std Dev [Compton-y]'); axs[0, 2].set_ylabel('Completeness'); axs[0, 2].set_title('Completeness vs Noise Variance'); axs[0, 2].legend(); axs[0, 2].grid(True, which='both', ls='--', alpha=0.5)
    snr_centers = (snr_bins[:-1] + snr_bins[1:]) / 2
    axs[1, 0].plot(snr_centers, pur_snr_std, label='Standard ILC', marker='o')
    axs[1, 0].plot(snr_centers, pur_snr_filt, label='Filtered MMF', marker='s')
    axs[1, 0].set_xlabel('Detection SNR'); axs[1, 0].set_ylabel('Purity'); axs[1, 0].set_title('Purity vs SNR'); axs[1, 0].legend(); axs[1, 0].grid(True, which='both', ls='--', alpha=0.5)
    axs[1, 1].plot(cib_centers, pur_cib_std, label='Standard ILC', marker='o')
    axs[1, 1].plot(cib_centers, pur_cib_filt, label='Filtered MMF', marker='s')
    axs[1, 1].set_xscale('log'); axs[1, 1].set_xlabel('Local CIB Intensity at 857 GHz [MJy/sr]'); axs[1, 1].set_ylabel('Purity'); axs[1, 1].set_title('Purity vs CIB Intensity'); axs[1, 1].legend(); axs[1, 1].grid(True, which='both', ls='--', alpha=0.5)
    axs[1, 2].plot(noise_centers, pur_noise_std, label='Standard ILC', marker='o')
    axs[1, 2].plot(noise_centers, pur_noise_filt, label='Filtered MMF', marker='s')
    axs[1, 2].set_xlabel('Local Noise Std Dev [Compton-y]'); axs[1, 2].set_ylabel('Purity'); axs[1, 2].set_title('Purity vs Noise Variance'); axs[1, 2].legend(); axs[1, 2].grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('data/detection_performance_' + str(int(time.time())) + '.png', dpi=300)
    plt.close()
    print('Performance plot saved to data/detection_performance_' + str(int(time.time())) + '.png')