# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('Starting Step 7: Summary Visualization')
    tf_data = np.load('data/transfer_functions.npz')
    ell = tf_data['ell']
    T_ell_constrained = tf_data['T_ell_constrained']
    T_ell_std = tf_data['T_ell_std']
    perf_data = np.load('data/detection_performance.npz')
    mass_bins = perf_data['mass_bins']
    comp_mass_std = perf_data['comp_mass_std']
    comp_mass_filt = perf_data['comp_mass_filt']
    snr_bins = perf_data['snr_bins']
    pur_snr_std = perf_data['pur_snr_std']
    pur_snr_filt = perf_data['pur_snr_filt']
    mass_cal_std = np.load('data/mass_calibration_std.npy')
    mass_cal_filt = np.load('data/mass_calibration_filt.npy')
    plt.rcParams['text.usetex'] = False
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs[0, 0].plot(ell[1:], T_ell_constrained[1:], label='Constrained MWF', color='red', linewidth=2)
    axs[0, 0].plot(ell[1:], T_ell_std[1:], label='Standard MWF', color='blue', linewidth=2, linestyle='--')
    axs[0, 0].axhline(1.0, color='black', linestyle=':', alpha=0.7)
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xlim(100, 10000)
    axs[0, 0].set_ylim(-0.1, 1.2)
    axs[0, 0].set_xlabel('Multipole ell')
    axs[0, 0].set_ylabel('Transfer Function T(ell)')
    axs[0, 0].set_title('Spatial Fidelity: Transfer Function')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which='both', ls='--', alpha=0.5)
    mass_centers = np.sqrt(mass_bins[:-1] * mass_bins[1:])
    axs[0, 1].plot(mass_centers, comp_mass_std, label='Standard ILC', marker='o')
    axs[0, 1].plot(mass_centers, comp_mass_filt, label='Filtered MMF', marker='s')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_xlabel('Halo Mass Proxy (Peak Compton-y)')
    axs[0, 1].set_ylabel('Completeness')
    axs[0, 1].set_title('Completeness vs Mass Proxy')
    axs[0, 1].legend()
    axs[0, 1].grid(True, which='both', ls='--', alpha=0.5)
    snr_centers = (snr_bins[:-1] + snr_bins[1:]) / 2
    axs[1, 0].plot(snr_centers, pur_snr_std, label='Standard ILC', marker='o')
    axs[1, 0].plot(snr_centers, pur_snr_filt, label='Filtered MMF', marker='s')
    axs[1, 0].set_xlabel('Detection SNR')
    axs[1, 0].set_ylabel('Purity')
    axs[1, 0].set_title('Purity vs SNR')
    axs[1, 0].legend()
    axs[1, 0].grid(True, which='both', ls='--', alpha=0.5)
    valid_std = None
    valid_filt = None
    has_data = False
    if len(mass_cal_std) > 0:
        true_Y_std = mass_cal_std[:, 1]
        bc_Y_std = mass_cal_std[:, 3]
        valid_std = (true_Y_std > 0) & np.isfinite(bc_Y_std)
        if np.any(valid_std):
            frac_bias_std = (bc_Y_std[valid_std] - true_Y_std[valid_std]) / true_Y_std[valid_std]
            axs[1, 1].hist(frac_bias_std, bins=20, alpha=0.5, label='Standard ILC (BC)', density=True)
            has_data = True
    if len(mass_cal_filt) > 0:
        true_Y_filt = mass_cal_filt[:, 1]
        bc_Y_filt = mass_cal_filt[:, 3]
        valid_filt = (true_Y_filt > 0) & np.isfinite(bc_Y_filt)
        if np.any(valid_filt):
            frac_bias_filt = (bc_Y_filt[valid_filt] - true_Y_filt[valid_filt]) / true_Y_filt[valid_filt]
            axs[1, 1].hist(frac_bias_filt, bins=20, alpha=0.5, label='Filtered MMF (BC)', density=True)
            has_data = True
    if not has_data:
        axs[1, 1].text(0.5, 0.5, 'No Detections Found', horizontalalignment='center', verticalalignment='center', transform=axs[1, 1].transAxes, fontsize=14)
        axs[1, 1].set_xlim(-1, 1)
    axs[1, 1].set_xlabel('Fractional Mass Bias (Recovered - True) / True')
    axs[1, 1].set_ylabel('Density')
    axs[1, 1].set_title('Mass Bias Distribution')
    if has_data:
        axs[1, 1].legend()
    axs[1, 1].grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_path = 'data/summary_visualization_' + str(timestamp) + '.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print('Summary visualization saved to ' + plot_path)
    print('\n--- Summary Statistics ---')
    def get_90_comp_mass(comp, mass_centers):
        idx = np.where(comp >= 0.9)[0]
        if len(idx) > 0:
            return mass_centers[idx[0]]
        else:
            return None
    comp90_std = get_90_comp_mass(comp_mass_std, mass_centers)
    comp90_filt = get_90_comp_mass(comp_mass_filt, mass_centers)
    if comp90_std is not None:
        print('Standard ILC 90% Completeness Mass Threshold: ' + str(round(comp90_std, 7)))
    else:
        print('Standard ILC 90% Completeness Mass Threshold: Not reached (Max: ' + str(round(np.nanmax(comp_mass_std), 4)) + ')')
    if comp90_filt is not None:
        print('Filtered MMF 90% Completeness Mass Threshold: ' + str(round(comp90_filt, 7)))
    else:
        print('Filtered MMF 90% Completeness Mass Threshold: Not reached (Max: ' + str(round(np.nanmax(comp_mass_filt), 4)) + ')')
    pur_std_val = pur_snr_std[0]
    if np.isnan(pur_std_val):
        pur_std_str = 'N/A'
    else:
        pur_std_str = str(round(pur_std_val, 4))
    pur_filt_val = pur_snr_filt[0]
    if np.isnan(pur_filt_val):
        pur_filt_str = 'N/A'
    else:
        pur_filt_str = str(round(pur_filt_val, 4))
    print('Standard ILC Purity at lowest SNR bin (' + str(round(snr_centers[0], 2)) + '): ' + pur_std_str)
    print('Filtered MMF Purity at lowest SNR bin (' + str(round(snr_centers[0], 2)) + '): ' + pur_filt_str)
    if len(mass_cal_std) > 0 and valid_std is not None and np.any(valid_std):
        print('Standard ILC Mean Fractional Mass Bias: ' + str(round(np.mean(frac_bias_std), 4)) + ' +/- ' + str(round(np.std(frac_bias_std), 4)))
    else:
        print('Standard ILC Mean Fractional Mass Bias: N/A (No detections)')
    if len(mass_cal_filt) > 0 and valid_filt is not None and np.any(valid_filt):
        print('Filtered MMF Mean Fractional Mass Bias: ' + str(round(np.mean(frac_bias_filt), 4)) + ' +/- ' + str(round(np.std(frac_bias_filt), 4)))
    else:
        print('Filtered MMF Mean Fractional Mass Bias: N/A (No detections)')
    print('--------------------------')
    print('Step 7 completed successfully. Output saved to ' + plot_path)