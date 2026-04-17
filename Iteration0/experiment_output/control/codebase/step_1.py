# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def get_planck_noise(freq, i_planck, i_patch, base_path, cache):
    key = str(freq) + '_' + str(i_planck)
    if key not in cache:
        filepath = base_path + '/planck_noise/planck_noise_' + str(freq) + '_' + str(i_planck) + '.npy'
        cache[key] = np.load(filepath, mmap_mode='r')
    return cache[key][i_patch]

def main():
    os.environ['OMP_NUM_THREADS'] = '16'
    sys.path.insert(0, '/home/node/data/compsep_data')
    import utils
    base_dir = '/home/node/data/compsep_data/cut_maps'
    n_patch = 1523
    n_planck = 100
    n_so = 3000
    rng = np.random.default_rng(seed=42)
    i_so_indices = rng.integers(0, n_so, size=n_patch)
    i_planck_indices = rng.integers(0, n_planck, size=n_patch)
    print('Loading individual component maps to confirm dynamic range disparity...')
    cmb = np.load(base_dir + '/lensed_cmb.npy')
    cib_857 = np.load(base_dir + '/cib_857.npy')
    tsz_map = np.load(base_dir + '/tsz.npy')
    ksz = np.load(base_dir + '/ksz.npy')
    var_cmb = np.var(cmb)
    var_cib_857 = np.var(cib_857)
    var_tsz = np.var(tsz_map)
    var_ksz = np.var(ksz)
    print('--------------------------------------------------')
    print('Component Variance Statistics:')
    print('Lensed CMB:     ' + np.format_float_scientific(var_cmb, precision=2) + ' uK_CMB^2')
    print('CIB at 857 GHz: ' + np.format_float_scientific(var_cib_857, precision=2) + ' (Jy/sr)^2')
    print('tSZ Compton-y:  ' + np.format_float_scientific(var_tsz, precision=2))
    print('kSZ Doppler-b:  ' + np.format_float_scientific(var_ksz, precision=2))
    print('--------------------------------------------------')
    del cmb, cib_857, ksz
    print('Loading stacked signal maps...')
    signal_90 = np.load(base_dir + '/stacked_90.npy')
    signal_150 = np.load(base_dir + '/stacked_150.npy')
    signal_217 = np.load(base_dir + '/stacked_217.npy')
    signal_353 = np.load(base_dir + '/stacked_353.npy')
    signal_545 = np.load(base_dir + '/stacked_545.npy')
    signal_857 = np.load(base_dir + '/stacked_857.npy')
    channels = ['90 GHz', '150 GHz', '217 GHz', '353 GHz', '545 GHz', '857 GHz']
    sig_vars = [np.var(signal_90), np.var(signal_150), np.var(signal_217), np.var(signal_353), np.var(signal_545), np.var(signal_857)]
    print('Stacked Signal Map Variances (uK_CMB^2):')
    for ch, v in zip(channels, sig_vars):
        print(ch + ': ' + np.format_float_scientific(v, precision=2))
    print('--------------------------------------------------')
    print('Computing tSZ normalization parameters...')
    tsz_mean = np.mean(tsz_map)
    tsz_std = np.std(tsz_map)
    print('tSZ Mean: ' + np.format_float_scientific(tsz_mean, precision=6))
    print('tSZ Std:  ' + np.format_float_scientific(tsz_std, precision=6))
    np.savez('data/tsz_norm_params.npz', mean=tsz_mean, std=tsz_std)
    print('Saved tSZ normalization parameters to data/tsz_norm_params.npz')
    print('Loading all patches using the specified loading pattern...')
    noise_90_so = np.load(base_dir + '/so_noise/90.npy', mmap_mode='r')
    noise_150_so = np.load(base_dir + '/so_noise/150.npy', mmap_mode='r')
    noise_217_so = np.load(base_dir + '/so_noise/217.npy', mmap_mode='r')
    planck_cache = {}
    obs_90_mean = 0.0
    for i_patch in range(n_patch):
        i_so = i_so_indices[i_patch]
        i_planck = i_planck_indices[i_patch]
        s_90 = signal_90[i_patch]
        n_90 = noise_90_so[i_so]
        pn_353_raw = get_planck_noise(353, i_planck, i_patch, base_dir, planck_cache)
        pn_545_raw = get_planck_noise(545, i_planck, i_patch, base_dir, planck_cache)
        pn_857_raw = get_planck_noise(857, i_planck, i_patch, base_dir, planck_cache)
        obs_90 = s_90 + n_90
        obs_90_mean += np.mean(obs_90)
    print('Successfully loaded all ' + str(n_patch) + ' patches. Mean of obs_90: ' + str(round(obs_90_mean / n_patch, 4)))
    print('Generating diagnostic plots...')
    plt.rcParams['text.usetex'] = False
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].bar(channels, sig_vars, color='skyblue', edgecolor='black')
    axs[0].set_yscale('log')
    axs[0].set_title('Per-Channel Stacked Signal Variance')
    axs[0].set_ylabel('Variance [uK_CMB^2]')
    axs[0].set_xlabel('Frequency Channel')
    axs[0].grid(True, which='both', ls='--', alpha=0.5)
    axs[0].tick_params(axis='x', rotation=45)
    counts, bins = np.histogram(tsz_map, bins=100)
    axs[1].stairs(counts, bins, fill=True, color='coral')
    axs[1].set_yscale('log')
    axs[1].set_title('tSZ Compton-y Histogram (Raw)')
    axs[1].set_ylabel('Pixel Count')
    axs[1].set_xlabel('Compton-y')
    axs[1].grid(True, ls='--', alpha=0.5)
    bins_std = (bins - tsz_mean) / tsz_std
    axs[2].stairs(counts, bins_std, fill=True, color='mediumseagreen')
    axs[2].set_yscale('log')
    axs[2].set_title('tSZ Histogram (Standardized)')
    axs[2].set_ylabel('Pixel Count')
    axs[2].set_xlabel('Standardized Value')
    axs[2].grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = 'data/diagnostic_plot_1_' + timestamp + '.png'
    fig.savefig(plot_filename, dpi=300)
    print('Diagnostic plot saved to ' + plot_filename)
    print('Step 1 completed successfully. Output shape: ' + str(tsz_map.shape) + ', saved to data/tsz_norm_params.npz')

if __name__ == '__main__':
    main()