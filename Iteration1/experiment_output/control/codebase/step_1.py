# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, '/home/node/data/compsep_data/')
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
def compute_spectra_for_patch(args):
    sig_maps, noise_maps = args
    n_sig = len(sig_maps)
    n_noise = len(noise_maps)
    ell, _ = utils.powers(sig_maps[0], sig_maps[0], ps=5, window_alpha=0.5)
    n_ell = len(ell)
    S_mat = np.zeros((n_sig, n_sig, n_ell))
    N_mat = np.zeros((n_noise, n_noise, n_ell))
    for i in range(n_sig):
        for j in range(i, n_sig):
            _, cl = utils.powers(sig_maps[i], sig_maps[j], ps=5, window_alpha=0.5)
            S_mat[i, j] = cl
            if i != j:
                S_mat[j, i] = cl
    for i in range(n_noise):
        for j in range(i, n_noise):
            _, cl = utils.powers(noise_maps[i], noise_maps[j], ps=5, window_alpha=0.5)
            N_mat[i, j] = cl
            if i != j:
                N_mat[j, i] = cl
    return S_mat, N_mat
if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '16'
    BASE = '/home/node/data/compsep_data/cut_maps'
    n_patch = 1523
    n_planck = 100
    n_so = 3000
    n_use = 400
    rng = np.random.default_rng(seed=42)
    patch_indices = np.arange(n_use)
    i_so_indices = rng.integers(0, n_so, size=n_use)
    i_planck_indices = rng.integers(0, n_planck, size=n_use)
    sig_90 = np.load(BASE + '/stacked_90.npy')[:n_use]
    sig_150 = np.load(BASE + '/stacked_150.npy')[:n_use]
    sig_217 = np.load(BASE + '/stacked_217.npy')[:n_use]
    sig_353 = np.load(BASE + '/stacked_353.npy')[:n_use]
    sig_545 = np.load(BASE + '/stacked_545.npy')[:n_use]
    sig_857 = np.load(BASE + '/stacked_857.npy')[:n_use]
    tsz = np.load(BASE + '/tsz.npy')[:n_use]
    cib_353 = np.load(BASE + '/cib_353.npy')[:n_use]
    cib_545 = np.load(BASE + '/cib_545.npy')[:n_use]
    cib_857 = np.load(BASE + '/cib_857.npy')[:n_use]
    so_90_full = np.load(BASE + '/so_noise/90.npy')
    so_150_full = np.load(BASE + '/so_noise/150.npy')
    so_217_full = np.load(BASE + '/so_noise/217.npy')
    noise_90 = so_90_full[i_so_indices]
    noise_150 = so_150_full[i_so_indices]
    noise_217 = so_217_full[i_so_indices]
    del so_90_full, so_150_full, so_217_full
    noise_353 = np.zeros((n_use, 256, 256), dtype=np.float64)
    noise_545 = np.zeros((n_use, 256, 256), dtype=np.float64)
    noise_857 = np.zeros((n_use, 256, 256), dtype=np.float64)
    for i_p in np.unique(i_planck_indices):
        mask = (i_planck_indices == i_p)
        patches_for_ip = patch_indices[mask]
        pn_353 = np.load(BASE + '/planck_noise/planck_noise_353_' + str(i_p) + '.npy')
        pn_545 = np.load(BASE + '/planck_noise/planck_noise_545_' + str(i_p) + '.npy')
        pn_857 = np.load(BASE + '/planck_noise/planck_noise_857_' + str(i_p) + '.npy')
        noise_353[mask] = pn_353[patches_for_ip] * 1e6
        noise_545[mask] = pn_545[patches_for_ip] * 1e6 * utils.jysr2uk(545)
        noise_857[mask] = pn_857[patches_for_ip] * 1e6 * utils.jysr2uk(857)
    args_list = []
    for i in range(n_use):
        s_maps = np.stack([sig_90[i], sig_150[i], sig_217[i], sig_353[i], sig_545[i], sig_857[i], tsz[i], cib_353[i], cib_545[i], cib_857[i]])
        n_maps = np.stack([noise_90[i], noise_150[i], noise_217[i], noise_353[i], noise_545[i], noise_857[i]])
        args_list.append((s_maps, n_maps))
    with mp.Pool(processes=16) as pool:
        results = pool.map(compute_spectra_for_patch, args_list)
    S_mat_avg = np.mean([r[0] for r in results], axis=0)
    N_mat_avg = np.mean([r[1] for r in results], axis=0)
    ell, _ = utils.powers(sig_90[0], sig_90[0], ps=5, window_alpha=0.5)
    S_ell_obs = S_mat_avg[0:6, 0:6, :]
    S_ell_tSZ_obs = S_mat_avg[6, 0:6, :]
    P_ell_tSZ_CIB = S_mat_avg[6, 7:10, :]
    N_ell = N_mat_avg
    np.save('data/S_ell_obs.npy', S_ell_obs)
    np.save('data/S_ell_tSZ_obs.npy', S_ell_tSZ_obs)
    np.save('data/P_ell_tSZ_CIB.npy', P_ell_tSZ_CIB)
    np.save('data/N_ell.npy', N_ell)
    np.save('data/ell.npy', ell)
    np.save('data/S_mat_full.npy', S_mat_avg)
    np.save('data/N_mat_full.npy', N_mat_avg)
    sigma = 10.0
    cl_vals = []
    for _ in range(100):
        wn = rng.normal(0, sigma, size=(256, 256))
        ell_val, cl = utils.powers(wn, wn, ps=5, window_alpha=None)
        cl_vals.append(cl)
    cl_val_mean = np.mean(cl_vals, axis=0)
    pixel_area = (5.0 * np.pi / 180.0 / 256.0)**2
    theory_cl = sigma**2 * pixel_area
    plt.figure(figsize=(8, 6))
    plt.plot(ell_val[1:], cl_val_mean[1:], label='Empirical C_ell (White Noise)', color='blue')
    plt.axhline(theory_cl, color='red', linestyle='--', label='Theoretical C_ell')
    plt.xlabel('Multipole ell')
    plt.ylabel('Power Spectrum C_ell')
    plt.title('Estimator Validation: Gaussian White Noise')
    plt.xscale('log')
    plt.xlim(100, 20000)
    plt.ylim(theory_cl * 0.5, theory_cl * 1.5)
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    timestamp = int(time.time())
    val_plot_path = 'data/estimator_validation_' + str(timestamp) + '.png'
    plt.savefig(val_plot_path, dpi=300)
    plt.close()
    print('Step 1 completed successfully. Output shapes: S_ell_obs ' + str(S_ell_obs.shape) + ', N_ell ' + str(N_ell.shape) + ', saved to data/ directory.')