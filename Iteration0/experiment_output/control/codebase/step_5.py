# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import scipy.ndimage as ndimage

def get_footprint(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return x**2 + y**2 <= radius**2

def apply_matched_filter_and_find_peaks(maps, tau, footprint, margin=15):
    tau_padded = np.zeros((256, 256))
    tau_padded[128-15:128+16, 128-15:128+16] = tau
    tau_k = np.fft.fft2(np.fft.ifftshift(tau_padded))
    P_k = np.zeros((256, 256))
    for i in range(maps.shape[0]):
        P_k += np.abs(np.fft.fft2(maps[i]))**2
    P_k /= maps.shape[0]
    P_k = np.maximum(P_k, 1e-30)
    F_k = np.conj(tau_k) / P_k
    norm = np.sum(tau_k * F_k).real / (256 * 256)
    F_k /= norm
    catalog_snr4 = []
    catalog_snr5 = []
    for i in range(maps.shape[0]):
        map_k = np.fft.fft2(maps[i])
        filtered_k = map_k * F_k
        filtered_map = np.real(np.fft.ifft2(filtered_k))
        med = np.median(filtered_map)
        mad = np.median(np.abs(filtered_map - med))
        local_noise = mad / 0.6744897501960817 if mad > 0 else np.std(filtered_map)
        if local_noise == 0:
            continue
        snr_map = filtered_map / local_noise
        local_max = ndimage.maximum_filter(snr_map, footprint=footprint) == snr_map
        peaks_4 = (snr_map > 4) & local_max
        y4, x4 = np.where(peaks_4)
        for yy, xx in zip(y4, x4):
            if yy >= margin and yy < 256 - margin and xx >= margin and xx < 256 - margin:
                catalog_snr4.append((i, yy, xx, snr_map[yy, xx], filtered_map[yy, xx], local_noise))
        peaks_5 = (snr_map > 5) & local_max
        y5, x5 = np.where(peaks_5)
        for yy, xx in zip(y5, x5):
            if yy >= margin and yy < 256 - margin and xx >= margin and xx < 256 - margin:
                catalog_snr5.append((i, yy, xx, snr_map[yy, xx], filtered_map[yy, xx], local_noise))
    cat4 = np.array(catalog_snr4) if len(catalog_snr4) > 0 else np.empty((0, 6))
    cat5 = np.array(catalog_snr5) if len(catalog_snr5) > 0 else np.empty((0, 6))
    return cat4, cat5

def get_true_catalog(tsz_maps, footprint, margin=15):
    catalog = []
    for i in range(tsz_maps.shape[0]):
        tsz = tsz_maps[i]
        local_max = ndimage.maximum_filter(tsz, footprint=footprint) == tsz
        peaks = (tsz > 1e-6) & local_max
        y, x = np.where(peaks)
        for yy, xx in zip(y, x):
            if yy >= margin and yy < 256 - margin and xx >= margin and xx < 256 - margin:
                catalog.append((i, yy, xx, tsz[yy, xx]))
    return np.array(catalog) if len(catalog) > 0 else np.empty((0, 4))

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '16'
    data_dir = 'data/'
    base_dir = '/home/node/data/compsep_data/cut_maps'
    print('Loading reconstructed maps and profiles...')
    mwf_maps = np.load(os.path.join(data_dir, 'mwf_reconstructions.npy'))
    ilc_maps = np.load(os.path.join(data_dir, 'ilc_reconstructions.npy'))
    tsz_maps = np.load(os.path.join(base_dir, 'tsz.npy'))
    profiles = np.load(os.path.join(data_dir, 'cluster_profiles.npz'))
    tau = profiles['tSZ_1arcmin']
    footprint = get_footprint(5)
    margin = 15
    print('Applying matched filter to MWF reconstructions...')
    mwf_cat4, mwf_cat5 = apply_matched_filter_and_find_peaks(mwf_maps, tau, footprint, margin)
    print('MWF candidates (SNR > 4): ' + str(len(mwf_cat4)))
    print('MWF candidates (SNR > 5): ' + str(len(mwf_cat5)))
    print('Applying matched filter to ILC reconstructions...')
    ilc_cat4, ilc_cat5 = apply_matched_filter_and_find_peaks(ilc_maps, tau, footprint, margin)
    print('ILC candidates (SNR > 4): ' + str(len(ilc_cat4)))
    print('ILC candidates (SNR > 5): ' + str(len(ilc_cat5)))
    print('Constructing ground-truth catalog from tsz.npy...')
    true_cat = get_true_catalog(tsz_maps, footprint, margin)
    print('True clusters (y > 1e-6): ' + str(len(true_cat)))
    print('Saving catalogs to disk...')
    np.save(os.path.join(data_dir, 'mwf_catalog_snr4.npy'), mwf_cat4)
    np.save(os.path.join(data_dir, 'mwf_catalog_snr5.npy'), mwf_cat5)
    np.save(os.path.join(data_dir, 'ilc_catalog_snr4.npy'), ilc_cat4)
    np.save(os.path.join(data_dir, 'ilc_catalog_snr5.npy'), ilc_cat5)
    np.save(os.path.join(data_dir, 'true_catalog.npy'), true_cat)
    print('Step 5 completed successfully. Output shape: ' + str(mwf_cat5.shape) + ', saved to data/mwf_catalog_snr5.npy')