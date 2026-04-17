# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import multiprocessing as mp
from scipy.ndimage import gaussian_filter
from collections import defaultdict
lookup_table = None
y_pred_bins = None
sigma_noise_bins = None
cib_bins = None
def init_worker(lt, yb, sb, cb):
    global lookup_table, y_pred_bins, sigma_noise_bins, cib_bins
    lookup_table = lt
    y_pred_bins = yb
    sigma_noise_bins = sb
    cib_bins = cb
def apply_lookup_array(y_arr, sigma_arr, cib_arr):
    i_y = np.clip(np.digitize(y_arr, y_pred_bins) - 1, 0, len(y_pred_bins) - 2)
    i_s = np.clip(np.digitize(sigma_arr, sigma_noise_bins) - 1, 0, len(sigma_noise_bins) - 2)
    i_c = np.clip(np.digitize(cib_arr, cib_bins) - 1, 0, len(cib_bins) - 2)
    return lookup_table[i_y, i_s, i_c]
def process_patch(args):
    i_patch, det_list, sigma_noise = args
    BASE = '/home/node/data/compsep_data/cut_maps'
    y_pred = np.load('data/y_pred_mwf.npy', mmap_mode='r')[i_patch]
    cib_857 = np.load(BASE + '/cib_857.npy', mmap_mode='r')[i_patch]
    tsz = np.load(BASE + '/tsz.npy', mmap_mode='r')[i_patch]
    cib_smoothed = gaussian_filter(cib_857, sigma=1.08)
    R_pix = 5.0 / (5.0 * 60 / 256)
    results = []
    Y, X = np.ogrid[:256, :256]
    for det in det_list:
        x_d, y_d, x_t, y_t, mass_proxy = det
        dist_d = np.sqrt((X - x_d)**2 + (Y - y_d)**2)
        mask_d = dist_d <= R_pix
        dist_t = np.sqrt((X - x_t)**2 + (Y - y_t)**2)
        mask_t = dist_t <= R_pix
        true_int_Y = np.sum(tsz[mask_t])
        raw_int_Y = np.sum(y_pred[mask_d])
        y_vals = y_pred[mask_d]
        cib_vals = cib_smoothed[mask_d]
        sigma_vals = np.full_like(y_vals, sigma_noise)
        bc_vals = apply_lookup_array(y_vals, sigma_vals, cib_vals)
        bc_int_Y = np.sum(bc_vals)
        results.append((mass_proxy, true_int_Y, raw_int_Y, bc_int_Y))
    return results
if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '16'
    print('Starting Step 6: Mass Calibration and Performance Quantification')
    bc_data = np.load('data/bias_correction_table.npz')
    lt = bc_data['lookup_table']
    yb = bc_data['y_pred_bins']
    sb = bc_data['sigma_noise_bins']
    cb = bc_data['cib_bins']
    det_std = np.load('data/detected_peaks_std.npy')
    det_filt = np.load('data/detected_peaks_filt.npy')
    truth_peaks = np.load('data/truth_peaks.npy')
    matches_std = np.load('data/matched_indices_std.npz')
    tp_det_idx_std = matches_std['det_idx']
    tp_truth_idx_std = matches_std['truth_idx']
    matches_filt = np.load('data/matched_indices_filt.npz')
    tp_det_idx_filt = matches_filt['det_idx']
    tp_truth_idx_filt = matches_filt['truth_idx']
    sigma_noise_mwf = np.load('data/sigma_noise_mwf.npy')
    def get_patch_args(tp_det_idx, tp_truth_idx, det_peaks, truth_peaks, sigma_noise_all):
        patch_dict = defaultdict(list)
        for d_idx, t_idx in zip(tp_det_idx, tp_truth_idx):
            d = det_peaks[d_idx]
            t = truth_peaks[t_idx]
            i_patch = int(d[3])
            patch_dict[i_patch].append((int(d[0]), int(d[1]), int(t[0]), int(t[1]), t[2]))
        args_list = []
        for i_patch in range(1523):
            if i_patch in patch_dict:
                args_list.append((i_patch, patch_dict[i_patch], sigma_noise_all[i_patch]))
        return args_list
    args_std = get_patch_args(tp_det_idx_std, tp_truth_idx_std, det_std, truth_peaks, sigma_noise_mwf)
    args_filt = get_patch_args(tp_det_idx_filt, tp_truth_idx_filt, det_filt, truth_peaks, sigma_noise_mwf)
    start_time = time.time()
    print('Processing ' + str(len(args_std)) + ' patches for Standard ILC using multiprocessing...')
    with mp.Pool(processes=16, initializer=init_worker, initargs=(lt, yb, sb, cb)) as pool:
        results_std = pool.map(process_patch, args_std)
    print('Processing ' + str(len(args_filt)) + ' patches for Filtered MMF using multiprocessing...')
    with mp.Pool(processes=16, initializer=init_worker, initargs=(lt, yb, sb, cb)) as pool:
        results_filt = pool.map(process_patch, args_filt)
    print('Processing completed in ' + str(round(time.time() - start_time, 2)) + ' seconds.')
    res_std_flat = [item for sublist in results_std for item in sublist]
    res_filt_flat = [item for sublist in results_filt for item in sublist]
    res_std_arr = np.array(res_std_flat) if len(res_std_flat) > 0 else np.empty((0, 4))
    res_filt_arr = np.array(res_filt_flat) if len(res_filt_flat) > 0 else np.empty((0, 4))
    np.save('data/mass_calibration_std.npy', res_std_arr)
    np.save('data/mass_calibration_filt.npy', res_filt_arr)
    def print_bias_stats(name, res_arr):
        if len(res_arr) == 0:
            print('No true positive detections for ' + name + '.')
            return
        mass_proxy = res_arr[:, 0]
        true_Y = res_arr[:, 1]
        raw_Y = res_arr[:, 2]
        bc_Y = res_arr[:, 3]
        valid = (true_Y > 0) & np.isfinite(raw_Y) & np.isfinite(bc_Y)
        if not np.any(valid):
            print('No valid true_Y > 0 for ' + name + '.')
            return
        frac_bias_raw = (raw_Y[valid] - true_Y[valid]) / true_Y[valid]
        frac_bias_bc = (bc_Y[valid] - true_Y[valid]) / true_Y[valid]
        print('\n--- Mass Calibration Stats: ' + name + ' ---')
        print('Number of matched clusters: ' + str(np.sum(valid)))
        print('Mean Fractional Bias (Raw): ' + str(round(np.mean(frac_bias_raw), 4)) + ' +/- ' + str(round(np.std(frac_bias_raw), 4)))
        print('Mean Fractional Bias (Bias-Corrected): ' + str(round(np.mean(frac_bias_bc), 4)) + ' +/- ' + str(round(np.std(frac_bias_bc), 4)))
        print('Median Fractional Bias (Raw): ' + str(round(np.median(frac_bias_raw), 4)))
        print('Median Fractional Bias (Bias-Corrected): ' + str(round(np.median(frac_bias_bc), 4)))
        mass_bins = np.logspace(np.log10(1e-6), np.log10(1e-4), 6)
        print('Bias by Mass Proxy Bin:')
        for i in range(len(mass_bins)-1):
            m_low = mass_bins[i]
            m_high = mass_bins[i+1]
            mask = (mass_proxy[valid] >= m_low) & (mass_proxy[valid] < m_high)
            if np.sum(mask) > 0:
                mb_raw = np.mean(frac_bias_raw[mask])
                mb_bc = np.mean(frac_bias_bc[mask])
                print('  Mass [' + str(m_low) + ', ' + str(m_high) + '): N=' + str(np.sum(mask)) + ', Raw Bias=' + str(round(mb_raw, 4)) + ', BC Bias=' + str(round(mb_bc, 4)))
        print('---------------------------------------')
    print_bias_stats('Standard ILC', res_std_arr)
    print_bias_stats('Filtered MMF', res_filt_arr)
    print('\nStep 6 completed successfully. Output shapes: std ' + str(res_std_arr.shape) + ', filt ' + str(res_filt_arr.shape) + ', saved to data/ directory.')