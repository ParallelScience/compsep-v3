# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, '/home/node/data/compsep_data/')
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from datetime import datetime

def get_radial_profile(data):
    center = (data.shape[0] // 2, data.shape[1] // 2)
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r_int = np.round(r).astype(int)
    tbin = np.bincount(r_int.ravel(), data.ravel())
    nr = np.bincount(r_int.ravel())
    valid = nr > 0
    radialprofile = np.zeros_like(tbin, dtype=float)
    radialprofile[valid] = tbin[valid] / nr[valid]
    return radialprofile

def main():
    os.environ['OMP_NUM_THREADS'] = '16'
    base_dir = '/home/node/data/compsep_data/cut_maps'
    data_dir = 'data/'
    print('Loading tSZ ground truth maps...')
    tsz_map = np.load(os.path.join(base_dir, 'tsz.npy'))
    threshold = 5e-5
    cutout_size = 31
    half = cutout_size // 2
    print('Identifying high-mass halos (peak y > ' + str(threshold) + ')...')
    cutouts = []
    for i in range(tsz_map.shape[0]):
        tsz = tsz_map[i]
        local_max = ndimage.maximum_filter(tsz, size=5) == tsz
        peaks = (tsz > threshold) & local_max
        y_idx, x_idx = np.where(peaks)
        for y, x in zip(y_idx, x_idx):
            if y >= half and y < 256 - half and x >= half and x < 256 - half:
                cutouts.append(tsz[y-half:y+half+1, x-half:x+half+1])
    cutouts = np.array(cutouts)
    print('Found ' + str(len(cutouts)) + ' clusters.')
    print('Stacking cutouts to compute the 2D profile...')
    stacked_profile_2d = np.mean(cutouts, axis=0)
    pixel_size_arcmin = 300.0 / 256.0
    fwhm_init = 1.0
    channels = ['90 GHz', '150 GHz', '217 GHz', '353 GHz', '545 GHz', '857 GHz']
    fwhms = [2.2, 1.4, 1.0, 4.5, 4.72, 4.42]
    templates_2d = {}
    templates_2d['tSZ_1arcmin'] = stacked_profile_2d
    print('Convolving stacked profile with frequency-specific beams...')
    for ch, fwhm_target in zip(channels, fwhms):
        if fwhm_target <= fwhm_init:
            templates_2d[ch] = stacked_profile_2d.copy()
        else:
            fwhm_add = np.sqrt(fwhm_target**2 - fwhm_init**2)
            sigma_add_arcmin = fwhm_add / np.sqrt(8 * np.log(2))
            sigma_add_pixels = sigma_add_arcmin / pixel_size_arcmin
            templates_2d[ch] = ndimage.gaussian_filter(stacked_profile_2d, sigma=sigma_add_pixels)
    out_filepath = os.path.join(data_dir, 'cluster_profiles.npz')
    np.savez(out_filepath, **templates_2d)
    print('Saved beam-convolved profile templates to ' + out_filepath)
    radial_profiles = {k: get_radial_profile(v) for k, v in templates_2d.items()}
    print('Generating visual comparison plot...')
    plt.rcParams['text.usetex'] = False
    fig = plt.figure(figsize=(18, 10))
    ax1 = plt.subplot2grid((3, 4), (0, 0), rowspan=3, colspan=2)
    r_arcmin = np.arange(len(radial_profiles['tSZ_1arcmin'])) * pixel_size_arcmin
    ax1.plot(r_arcmin, radial_profiles['tSZ_1arcmin'], label='Original (1 arcmin)', color='black', linestyle='--', linewidth=2)
    for ch in channels:
        ax1.plot(r_arcmin, radial_profiles[ch], label=ch + ' (' + str(fwhms[channels.index(ch)]) + ' arcmin)', marker='o', markersize=4)
    ax1.set_xlabel('Radius [arcmin]')
    ax1.set_ylabel('Compton-y')
    ax1.set_title('Radially Averaged Cluster Profiles')
    ax1.legend()
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.set_xlim(0, 15)
    vmax = stacked_profile_2d.max()
    extent = [-half * pixel_size_arcmin, half * pixel_size_arcmin, -half * pixel_size_arcmin, half * pixel_size_arcmin]
    for i, ch in enumerate(channels):
        row = i // 2
        col = 2 + (i % 2)
        ax = plt.subplot2grid((3, 4), (row, col))
        im = ax.imshow(templates_2d[ch], cmap='viridis', origin='lower', extent=extent, vmin=0, vmax=vmax)
        ax.set_title(ch + ' Template')
        ax.set_xlabel('x [arcmin]')
        if i % 2 == 0:
            ax.set_ylabel('y [arcmin]')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = os.path.join(data_dir, 'cluster_profiles_3_' + timestamp + '.png')
    fig.savefig(plot_filename, dpi=300)
    print('Plot saved to ' + plot_filename)
    print('Step 3 completed successfully. Output shape: ' + str(stacked_profile_2d.shape) + ', saved to ' + out_filepath)

if __name__ == '__main__':
    main()