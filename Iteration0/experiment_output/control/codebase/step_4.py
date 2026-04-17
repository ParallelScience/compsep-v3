# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from datetime import datetime
import utils

def init_worker(base_dir_in):
    global s_90, s_150, s_217, s_353, s_545, s_857
    global n_90, n_150, n_217
    global base_dir
    global planck_cache
    base_dir = base_dir_in
    planck_cache = {}
    s_90 = np.load(os.path.join(base_dir, 'stacked_90.npy'), mmap_mode='r')
    s_150 = np.load(os.path.join(base_dir, 'stacked_150.npy'), mmap_mode='r')
    s_217 = np.load(os.path.join(base_dir, 'stacked_217.npy'), mmap_mode='r')
    s_353 = np.load(os.path.join(base_dir, 'stacked_353.npy'), mmap_mode='r')
    s_545 = np.load(os.path.join(base_dir, 'stacked_545.npy'), mmap_mode='r')
    s_857 = np.load(os.path.join(base_dir, 'stacked_857.npy'), mmap_mode='r')
    n_90 = np.load(os.path.join(base_dir, 'so_noise/90.npy'), mmap_mode='r')
    n_150 = np.load(os.path.join(base_dir, 'so_noise/150.npy'), mmap_mode='r')
    n_217 = np.load(os.path.join(base_dir, 'so_noise/217.npy'), mmap_mode='r')

def get_planck_noise(freq, i_planck, i_patch):
    key = str(freq) + '_' + str(i_planck)
    if key not in planck_cache:
        filepath = os.path.join(base_dir, 'planck_noise/planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy')
        planck_cache[key] = np.load(filepath, mmap_mode='r')
    return planck_cache[key][i_patch]

def process_patch(args):
    i_patch, i_so, i_planck, W_2d, responses = args
    obs = np.zeros((6, 256, 256), dtype=np.float64)
    obs[0] = s_90[i_patch] + n_90[i_so]
    obs[1] = s_150[i_patch] + n_150[i_so]
    obs[2] = s_217[i_patch] + n_217[i_so]
    pn_353 = get_planck_noise(353, i_planck, i_patch) * 1e6
    pn_545 = get_planck_noise(545, i_planck, i_patch) * 1e6 * utils.jysr2uk(545)
    pn_857 = get_planck_noise(857, i_planck, i_patch) * 1e6 * utils.jysr2uk(857)
    obs[3] = s_353[i_patch] + pn_353
    obs[4] = s_545[i_patch] + pn_545
    obs[5] = s_857[i_patch] + pn_857
    pad_w = 128
    obs_padded = np.pad(obs, ((0,0), (pad_w, pad_w), (pad_w, pad_w)), mode='reflect')
    obs_fft = np.fft.fft2(obs_padded, axes=(1, 2))
    mwf_fft = np.sum(W_2d * obs_fft, axis=0)
    mwf_map_padded = np.real(np.fft.ifft2(mwf_fft))
    mwf_map = mwf_map_padded[pad_w:-pad_w, pad_w:-pad_w]
    obs_y = obs / responses[:, None, None]
    var_y = np.var(obs_y, axis=(1, 2))
    w = 1.0 / var_y
    w /= np.sum(w)
    ilc_map = np.sum(obs_y * w[:, None, None], axis=0)
    return i_patch, mwf_map, ilc_map, w

def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    base_dir = '/home/node/data/compsep_data/cut_maps'
    data_dir = 'data/'
    mwf_data = np.load(os.path.join(data_dir, 'mwf_weights.npz'))
    W = mwf_data['W']
    ell = mwf_data['ell']
    ps_arcmin = 300.0 / 256.0
    ps_rad = ps_arcmin * np.pi / (180.0 * 60.0)
    N_pad = 512
    kx = np.fft.fftfreq(N_pad, d=ps_rad)
    ky = np.fft.fftfreq(N_pad, d=ps_rad)
    KX, KY = np.meshgrid(kx, ky)
    ell_2d = 2 * np.pi * np.sqrt(KX**2 + KY**2)
    W_2d = np.zeros((6, N_pad, N_pad))
    ell_2d_flat = ell_2d.ravel()
    for i in range(6):
        W_2d[i] = np.interp(ell_2d_flat, ell, W[i], left=W[i,0], right=W[i,-1]).reshape(N_pad, N_pad)
    freqs = [90, 150, 217, 353, 545, 857]
    responses = np.array([utils.tsz(f) for f in freqs])
    n_patch = 1523
    rng = np.random.default_rng(seed=42)
    i_so_indices = rng.integers(0, 3000, size=n_patch)
    i_planck_indices = rng.integers(0, 100, size=n_patch)
    args_list = [(p, i_so_indices[p], i_planck_indices[p], W_2d, responses) for p in range(n_patch)]
    mwf_reconstructions = np.zeros((n_patch, 256, 256), dtype=np.float64)
    ilc_reconstructions = np.zeros((n_patch, 256, 256), dtype=np.float64)
    avg_ilc_weights = np.zeros(6)
    with mp.Pool(processes=16, initializer=init_worker, initargs=(base_dir,)) as pool:
        results = pool.map(process_patch, args_list)
    for res in results:
        i_patch, mwf_map, ilc_map, w = res
        mwf_reconstructions[i_patch] = mwf_map
        ilc_reconstructions[i_patch] = ilc_map
        avg_ilc_weights += w
    avg_ilc_weights /= n_patch
    mwf_out = os.path.join(data_dir, 'mwf_reconstructions.npy')
    ilc_out = os.path.join(data_dir, 'ilc_reconstructions.npy')
    np.save(mwf_out, mwf_reconstructions)
    np.save(ilc_out, ilc_reconstructions)
    tsz_map = np.load(os.path.join(base_dir, 'tsz.npy'))
    n_eval = 500
    eval_indices = rng.choice(n_patch, size=n_eval, replace=False)
    P_true_auto, P_mwf_auto, P_mwf_cross, P_ilc_auto, P_ilc_cross = [], [], [], [], []
    for idx in eval_indices:
        true_map = tsz_map[idx]
        mwf_map = mwf_reconstructions[idx]
        ilc_map = ilc_reconstructions[idx]
        cl_true, ell_eval = utils.powers(true_map, true_map, ps=ps_arcmin, window_alpha=0.5)
        cl_mwf, _ = utils.powers(mwf_map, mwf_map, ps=ps_arcmin, window_alpha=0.5)
        cl_mwf_cross, _ = utils.powers(mwf_map, true_map, ps=ps_arcmin, window_alpha=0.5)
        cl_ilc, _ = utils.powers(ilc_map, ilc_map, ps=ps_arcmin, window_alpha=0.5)
        cl_ilc_cross, _ = utils.powers(ilc_map, true_map, ps=ps_arcmin, window_alpha=0.5)
        P_true_auto.append(cl_true)
        P_mwf_auto.append(cl_mwf)
        P_mwf_cross.append(cl_mwf_cross)
        P_ilc_auto.append(cl_ilc)
        P_ilc_cross.append(cl_ilc_cross)
    P_true_auto = np.mean(P_true_auto, axis=0)
    P_mwf_auto = np.mean(P_mwf_auto, axis=0)
    P_mwf_cross = np.mean(P_mwf_cross, axis=0)
    P_ilc_auto = np.mean(P_ilc_auto, axis=0)
    P_ilc_cross = np.mean(P_ilc_cross, axis=0)
    sort_idx = np.argsort(ell_eval)
    ell_eval = ell_eval[sort_idx]
    P_true_auto = P_true_auto[sort_idx]
    P_mwf_auto = P_mwf_auto[sort_idx]
    P_mwf_cross = P_mwf_cross[sort_idx]
    P_ilc_auto = P_ilc_auto[sort_idx]
    P_ilc_cross = P_ilc_cross[sort_idx]
    T_mwf = P_mwf_cross / np.maximum(P_true_auto, 1e-30)
    r_mwf = P_mwf_cross / np.sqrt(np.maximum(P_mwf_auto * P_true_auto, 1e-30))
    T_ilc = P_ilc_cross / np.maximum(P_true_auto, 1e-30)
    r_ilc = P_ilc_cross / np.sqrt(np.maximum(P_ilc_auto * P_true_auto, 1e-30))
    plt.rcParams['text.usetex'] = False
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].plot(ell_eval, T_mwf, label='MWF', color='blue', lw=2)
    axs[0].plot(ell_eval, T_ilc, label='ILC Baseline', color='red', lw=2, linestyle='--')
    axs[0].axhline(1.0, color='k', linestyle=':', alpha=0.5)
    axs[0].set_xscale('log')
    axs[0].set_xlabel('Multipole ell')
    axs[0].set_ylabel('Transfer Function T(ell)')
    axs[0].legend()
    axs[1].plot(ell_eval, r_mwf, label='MWF', color='blue', lw=2)
    axs[1].plot(ell_eval, r_ilc, label='ILC Baseline', color='red', lw=2, linestyle='--')
    axs[1].axhline(1.0, color='k', linestyle=':', alpha=0.5)
    axs[1].axhline(0.0, color='k', linestyle='-', alpha=0.5)
    axs[1].set_xscale('log')
    axs[1].set_xlabel('Multipole ell')
    axs[1].set_ylabel('Cross-Correlation Coefficient r_ell')
    axs[1].legend()
    fig.tight_layout()
    plot_filename = os.path.join(data_dir, 'reconstruction_eval_4.png')
    fig.savefig(plot_filename, dpi=300)
    print('Step 4 completed successfully.')

if __name__ == '__main__':
    main()