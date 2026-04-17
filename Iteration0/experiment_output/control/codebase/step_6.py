# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from datetime import datetime

def match_catalogs_1to1(true_cat, cand_cat, dist_th=5.0):
    matched_true_idx = []
    matched_cand_idx = []
    for p in range(1523):
        t_mask = true_cat[:, 0] == p
        c_mask = cand_cat[:, 0] == p
        t_idx = np.where(t_mask)[0]
        c_idx = np.where(c_mask)[0]
        if len(t_idx) == 0 or len(c_idx) == 0:
            continue
        t_coords = true_cat[t_idx, 1:3]
        c_coords = cand_cat[c_idx, 1:3]
        tree = cKDTree(t_coords)
        dists, indices = tree.query(c_coords, distance_upper_bound=dist_th)
        valid = dists <= dist_th
        valid_c = np.where(valid)[0]
        valid_t = indices[valid]
        valid_d = dists[valid]
        sort_idx = np.argsort(valid_d)
        valid_c = valid_c[sort_idx]
        valid_t = valid_t[sort_idx]
        used_t = set()
        used_c = set()
        for c, t in zip(valid_c, valid_t):
            if c not in used_c and t not in used_t:
                matched_true_idx.append(t_idx[t])
                matched_cand_idx.append(c_idx[c])
                used_t.add(t)
                used_c.add(c)
    return np.array(matched_true_idx), np.array(matched_cand_idx)

def get_integrated_y(maps, indices, y_coords, x_coords, radius=3):
    int_y = np.zeros(len(indices))
    for k in range(len(indices)):
        p = indices[k]
        yy = y_coords[k]
        xx = x_coords[k]
        y_min = max(0, yy - radius)
        y_max = min(256, yy + radius + 1)
        x_min = max(0, xx - radius)
        x_max = min(256, xx + radius + 1)
        cutout = maps[p, y_min:y_max, x_min:x_max]
        cy = yy - y_min
        cx = xx - x_min
        y_grid, x_grid = np.ogrid[0:cutout.shape[0], 0:cutout.shape[1]]
        mask = (y_grid - cy)**2 + (x_grid - cx)**2 <= radius**2
        int_y[k] = np.sum(cutout[mask])
    return int_y

def main():
    os.environ['OMP_NUM_THREADS'] = '16'
    data_dir = 'data/'
    base_dir = '/home/node/data/compsep_data/cut_maps'
    true_cat = np.load(os.path.join(data_dir, 'true_catalog.npy'))
    mwf_cat5 = np.load(os.path.join(data_dir, 'mwf_catalog_snr5.npy'))
    ilc_cat5 = np.load(os.path.join(data_dir, 'ilc_catalog_snr5.npy'))
    tsz_maps = np.load(os.path.join(base_dir, 'tsz.npy'), mmap_mode='r')
    cib_857_maps = np.load(os.path.join(base_dir, 'cib_857.npy'), mmap_mode='r')
    mwf_maps = np.load(os.path.join(data_dir, 'mwf_reconstructions.npy'), mmap_mode='r')
    ilc_maps = np.load(os.path.join(data_dir, 'ilc_reconstructions.npy'), mmap_mode='r')
    true_i = true_cat[:, 0].astype(int)
    true_y = true_cat[:, 1].astype(int)
    true_x = true_cat[:, 2].astype(int)
    true_peak_y = true_cat[:, 3]
    true_cib = cib_857_maps[true_i, true_y, true_x]
    mwf_i = mwf_cat5[:, 0].astype(int)
    mwf_y = mwf_cat5[:, 1].astype(int)
    mwf_x = mwf_cat5[:, 2].astype(int)
    mwf_cib = cib_857_maps[mwf_i, mwf_y, mwf_x]
    mwf_rec_y = mwf_cat5[:, 4]
    ilc_i = ilc_cat5[:, 0].astype(int)
    ilc_y = ilc_cat5[:, 1].astype(int)
    ilc_x = ilc_cat5[:, 2].astype(int)
    ilc_cib = cib_857_maps[ilc_i, ilc_y, ilc_x]
    ilc_rec_y = ilc_cat5[:, 4]
    mwf_match_t, mwf_match_c = match_catalogs_1to1(true_cat, mwf_cat5, dist_th=5.0)
    ilc_match_t, ilc_match_c = match_catalogs_1to1(true_cat, ilc_cat5, dist_th=5.0)
    y_max = max(1e-4, np.max(true_peak_y))
    bins_y = np.logspace(-6, np.log10(y_max), 21)
    true_hist_y, _ = np.histogram(true_peak_y, bins=bins_y)
    mwf_match_hist_y, _ = np.histogram(true_peak_y[mwf_match_t], bins=bins_y)
    ilc_match_hist_y, _ = np.histogram(true_peak_y[ilc_match_t], bins=bins_y)
    comp_mwf_y = mwf_match_hist_y / np.maximum(true_hist_y, 1)
    comp_ilc_y = ilc_match_hist_y / np.maximum(true_hist_y, 1)
    rec_y_min = max(1e-7, min(np.min(mwf_rec_y), np.min(ilc_rec_y)))
    rec_y_max = max(np.max(mwf_rec_y), np.max(ilc_rec_y))
    bins_rec_y = np.logspace(np.log10(rec_y_min), np.log10(rec_y_max), 21)
    mwf_hist_rec_y, _ = np.histogram(mwf_rec_y, bins=bins_rec_y)
    mwf_match_hist_rec_y, _ = np.histogram(mwf_rec_y[mwf_match_c], bins=bins_rec_y)
    pur_mwf_y = mwf_match_hist_rec_y / np.maximum(mwf_hist_rec_y, 1)
    ilc_hist_rec_y, _ = np.histogram(ilc_rec_y, bins=bins_rec_y)
    ilc_match_hist_rec_y, _ = np.histogram(ilc_rec_y[ilc_match_c], bins=bins_rec_y)
    pur_ilc_y = ilc_match_hist_rec_y / np.maximum(ilc_hist_rec_y, 1)
    mask_detectable = true_peak_y > 1e-5
    valid_cib = true_cib[mask_detectable]
    valid_cib = valid_cib[valid_cib > 0]
    if len(valid_cib) > 0:
        cib_min = np.percentile(valid_cib, 1)
        cib_max = np.percentile(valid_cib, 99)
    else:
        cib_min, cib_max = 1e4, 1e7
    bins_cib = np.logspace(np.log10(cib_min), np.log10(cib_max), 21)
    true_hist_cib, _ = np.histogram(true_cib[mask_detectable], bins=bins_cib)
    mwf_match_t_det = mwf_match_t[true_peak_y[mwf_match_t] > 1e-5]
    mwf_match_hist_cib, _ = np.histogram(true_cib[mwf_match_t_det], bins=bins_cib)
    ilc_match_t_det = ilc_match_t[true_peak_y[ilc_match_t] > 1e-5]
    ilc_match_hist_cib, _ = np.histogram(true_cib[ilc_match_t_det], bins=bins_cib)
    comp_mwf_cib = mwf_match_hist_cib / np.maximum(true_hist_cib, 1)
    comp_ilc_cib = ilc_match_hist_cib / np.maximum(true_hist_cib, 1)
    valid_mwf_cib = mwf_cib[mwf_cib > 0]
    if len(valid_mwf_cib) > 0:
        cand_cib_min = np.percentile(valid_mwf_cib, 1)
        cand_cib_max = np.percentile(valid_mwf_cib, 99)
    else:
        cand_cib_min, cand_cib_max = 1e4, 1e7
    bins_cand_cib = np.logspace(np.log10(cand_cib_min), np.log10(cand_cib_max), 21)
    mwf_hist_cib, _ = np.histogram(mwf_cib, bins=bins_cand_cib)
    mwf_match_hist_cand_cib, _ = np.histogram(mwf_cib[mwf_match_c], bins=bins_cand_cib)
    pur_mwf_cib = mwf_match_hist_cand_cib / np.maximum(mwf_hist_cib, 1)
    ilc_hist_cib, _ = np.histogram(ilc_cib, bins=bins_cand_cib)
    ilc_match_hist_cand_cib, _ = np.histogram(ilc_cib[ilc_match_c], bins=bins_cand_cib)
    pur_ilc_cib = ilc_match_hist_cand_cib / np.maximum(ilc_hist_cib, 1)
    true_int_y_mwf = get_integrated_y(tsz_maps, true_i[mwf_match_t], true_y[mwf_match_t], true_x[mwf_match_t], radius=3)
    rec_int_y_mwf = get_integrated_y(mwf_maps, mwf_i[mwf_match_c], mwf_y[mwf_match_c], mwf_x[mwf_match_c], radius=3)
    true_int_y_ilc = get_integrated_y(tsz_maps, true_i[ilc_match_t], true_y[ilc_match_t], true_x[ilc_match_t], radius=3)
    rec_int_y_ilc = get_integrated_y(ilc_maps, ilc_i[ilc_match_c], ilc_y[ilc_match_c], ilc_x[ilc_match_c], radius=3)
    bias_mwf = (rec_int_y_mwf - true_int_y_mwf) / np.maximum(true_int_y_mwf, 1e-12)
    bias_ilc = (rec_int_y_ilc - true_int_y_ilc) / np.maximum(true_int_y_ilc, 1e-12)
    true_peak_y_mwf = true_peak_y[mwf_match_t]
    true_peak_y_ilc = true_peak_y[ilc_match_t]
    mask_bias_mwf = true_peak_y_mwf > 1e-5
    mean_bias_mwf = np.nanmean(bias_mwf[mask_bias_mwf])
    std_bias_mwf = np.nanstd(bias_mwf[mask_bias_mwf])
    mask_bias_ilc = true_peak_y_ilc > 1e-5
    mean_bias_ilc = np.nanmean(bias_ilc[mask_bias_ilc])
    std_bias_ilc = np.nanstd(bias_ilc[mask_bias_ilc])
    out_filepath = os.path.join(data_dir, 'performance_metrics.npz')
    np.savez(out_filepath, bins_y=bins_y, comp_mwf_y=comp_mwf_y, comp_ilc_y=comp_ilc_y, bins_rec_y=bins_rec_y, pur_mwf_y=pur_mwf_y, pur_ilc_y=pur_ilc_y, bins_cib=bins_cib, comp_mwf_cib=comp_mwf_cib, comp_ilc_cib=comp_ilc_cib, bins_cand_cib=bins_cand_cib, pur_mwf_cib=pur_mwf_cib, pur_ilc_cib=pur_ilc_cib, true_int_y_mwf=true_int_y_mwf, rec_int_y_mwf=rec_int_y_mwf, bias_mwf=bias_mwf, true_peak_y_mwf=true_peak_y_mwf, true_int_y_ilc=true_int_y_ilc, rec_int_y_ilc=rec_int_y_ilc, bias_ilc=bias_ilc, true_peak_y_ilc=true_peak_y_ilc)
    plt.rcParams['text.usetex'] = False
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs[0, 0].plot(bins_y[:-1], comp_mwf_y, drawstyle='steps-post', label='MWF', color='blue', lw=2)
    axs[0, 0].plot(bins_y[:-1], comp_ilc_y, drawstyle='steps-post', label='ILC', color='red', lw=2, linestyle='--')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xlabel('True Peak Compton-y')
    axs[0, 0].set_ylabel('Completeness')
    axs[0, 0].set_title('Completeness vs True Peak Y')
    axs[0, 0].legend()
    axs[0, 0].grid(True, ls='--', alpha=0.5)
    axs[0, 1].plot(bins_rec_y[:-1], pur_mwf_y, drawstyle='steps-post', label='MWF', color='blue', lw=2)
    axs[0, 1].plot(bins_rec_y[:-1], pur_ilc_y, drawstyle='steps-post', label='ILC', color='red', lw=2, linestyle='--')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_xlabel('Candidate Recovered Y')
    axs[0, 1].set_ylabel('Purity')
    axs[0, 1].set_title('Purity vs Candidate Recovered Y')
    axs[0, 1].legend()
    axs[0, 1].grid(True, ls='--', alpha=0.5)
    axs[1, 0].plot(bins_cib[:-1], comp_mwf_cib, drawstyle='steps-post', label='MWF', color='blue', lw=2)
    axs[1, 0].plot(bins_cib[:-1], comp_ilc_cib, drawstyle='steps-post', label='ILC', color='red', lw=2, linestyle='--')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_xlabel('Local CIB 857 GHz [Jy/sr]')
    axs[1, 0].set_ylabel('Completeness (True Y > 1e-5)')
    axs[1, 0].set_title('Completeness vs Local CIB')
    axs[1, 0].legend()
    axs[1, 0].grid(True, ls='--', alpha=0.5)
    axs[1, 1].plot(bins_cand_cib[:-1], pur_mwf_cib, drawstyle='steps-post', label='MWF', color='blue', lw=2)
    axs[1, 1].plot(bins_cand_cib[:-1], pur_ilc_cib, drawstyle='steps-post', label='ILC', color='red', lw=2, linestyle='--')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_xlabel('Local CIB 857 GHz [Jy/sr]')
    axs[1, 1].set_ylabel('Purity')
    axs[1, 1].set_title('Purity vs Local CIB')
    axs[1, 1].legend()
    axs[1, 1].grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename1 = os.path.join(data_dir, 'completeness_purity_6_' + timestamp + '.png')
    fig.savefig(plot_filename1, dpi=300)
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 6))
    valid_mwf = (true_int_y_mwf > 0) & (rec_int_y_mwf > 0)
    valid_ilc = (true_int_y_ilc > 0) & (rec_int_y_ilc > 0)
    axs2[0].scatter(true_int_y_ilc[valid_ilc], rec_int_y_ilc[valid_ilc], alpha=0.3, s=10, label='ILC', color='red')
    axs2[0].scatter(true_int_y_mwf[valid_mwf], rec_int_y_mwf[valid_mwf], alpha=0.3, s=10, label='MWF', color='blue')
    if np.sum(valid_mwf) > 0 and np.sum(valid_ilc) > 0:
        min_val = min(np.nanmin(true_int_y_mwf[valid_mwf]), np.nanmin(rec_int_y_mwf[valid_mwf]), np.nanmin(true_int_y_ilc[valid_ilc]), np.nanmin(rec_int_y_ilc[valid_ilc]))
        max_val = max(np.nanmax(true_int_y_mwf[valid_mwf]), np.nanmax(rec_int_y_mwf[valid_mwf]), np.nanmax(true_int_y_ilc[valid_ilc]), np.nanmax(rec_int_y_ilc[valid_ilc]))
    else:
        min_val, max_val = 1e-7, 1e-4
    if min_val <= 0: min_val = 1e-7
    axs2[0].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    axs2[0].set_xscale('log')
    axs2[0].set_yscale('log')
    axs2[0].set_xlabel('True Integrated Y')
    axs2[0].set_ylabel('Recovered Integrated Y')
    axs2[0].set_title('Recovered vs True Integrated Y')
    axs2[0].legend()
    axs2[0].grid(True, ls='--', alpha=0.5)
    axs2[1].scatter(true_peak_y_ilc, bias_ilc, alpha=0.3, s=10, label='ILC', color='red')
    axs2[1].scatter(true_peak_y_mwf, bias_mwf, alpha=0.3, s=10, label='MWF', color='blue')
    axs2[1].axhline(0, color='k', linestyle='--', lw=2)
    axs2[1].set_xscale('log')
    axs2[1].set_ylim(-2, 2)
    axs2[1].set_xlabel('True Peak Compton-y')
    axs2[1].set_ylabel('Mass Bias (Rec - True) / True')
    axs2[1].set_title('Mass Bias vs True Peak Y')
    axs2[1].legend()
    axs2[1].grid(True, ls='--', alpha=0.5)
    fig2.tight_layout()
    plot_filename2 = os.path.join(data_dir, 'scatter_bias_6_' + timestamp + '.png')
    fig2.savefig(plot_filename2, dpi=300)
    print('Step 6 completed successfully.')

if __name__ == '__main__':
    main()