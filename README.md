# compsep-v3

**Scientist:** denario-6
**Date:** 2026-04-16


# Dataset Description: FLAMINGO Lensed Component-Separation Maps

This dataset consists of flat-sky cut maps derived from full-sky maps of the cosmic microwave background (CMB), thermal and kinetic Sunyaev-Zel'dovich effects (tSZ/kSZ), and the cosmic infrared background (CIB). The sky signal maps originate from the FLAMINGO L1_m9 HYDRO_FIDUCIAL simulation (lightcone0, Jeger rotation) — a large-volume (1 Gpc comoving box) hydrodynamical simulation with self-consistent baryonic physics (radiative cooling, star formation, AGN feedback, diffuse gas flows).

The CMB is a Gaussian realization lensed by the FLAMINGO convergence map (κ) via lenspyx, using Planck-like cosmology (h=0.6736, n_s=0.965, A_s=2.1e-9, seed=42).

---

## File Inventory (absolute paths, verified)

All files are under `/home/node/data/compsep_data/cut_maps/`.

### Individual component maps — ground truth, 1 arcmin beam
Shape: **(1523, 256, 256)**, dtype **float64**

- `/home/node/data/compsep_data/cut_maps/lensed_cmb.npy` — Lensed CMB temperature, µK_CMB
- `/home/node/data/compsep_data/cut_maps/tsz.npy` — tSZ Compton-y parameter, dimensionless
- `/home/node/data/compsep_data/cut_maps/ksz.npy` — kSZ Doppler-b parameter, dimensionless
- `/home/node/data/compsep_data/cut_maps/cib_353.npy` — CIB at 353 GHz, Jy/sr
- `/home/node/data/compsep_data/cut_maps/cib_545.npy` — CIB at 545 GHz, Jy/sr
- `/home/node/data/compsep_data/cut_maps/cib_857.npy` — CIB at 857 GHz, Jy/sr

### Stacked (observed) signal maps — frequency-specific beam, no noise
Shape: **(1523, 256, 256)**, dtype **float64**, units **µK_CMB**

- `/home/node/data/compsep_data/cut_maps/stacked_90.npy` — 90 GHz total signal, beam FWHM 2.2 arcmin (SO LAT)
- `/home/node/data/compsep_data/cut_maps/stacked_150.npy` — 150 GHz total signal, beam FWHM 1.4 arcmin (SO LAT)
- `/home/node/data/compsep_data/cut_maps/stacked_217.npy` — 217 GHz total signal, beam FWHM 1.0 arcmin (SO LAT)
- `/home/node/data/compsep_data/cut_maps/stacked_353.npy` — 353 GHz total signal, beam FWHM 4.5 arcmin (Planck HFI)
- `/home/node/data/compsep_data/cut_maps/stacked_545.npy` — 545 GHz total signal, beam FWHM 4.72 arcmin (Planck HFI)
- `/home/node/data/compsep_data/cut_maps/stacked_857.npy` — 857 GHz total signal, beam FWHM 4.42 arcmin (Planck HFI)

The stacked signal at each frequency is: signal(freq) = CIB(freq)*jysr2uk(freq) + tSZ*f_tSZ(freq) + kSZ*f_kSZ + lensed_CMB, then smoothed with the frequency-specific beam. Individual component maps have a 1 arcmin beam.

### Noise maps

**Simons Observatory (SO) LAT noise** — 3000 independent Gaussian realizations
Shape: **(3000, 256, 256)**, dtype **float64**, units **µK_CMB**

- `/home/node/data/compsep_data/cut_maps/so_noise/90.npy`
- `/home/node/data/compsep_data/cut_maps/so_noise/150.npy`
- `/home/node/data/compsep_data/cut_maps/so_noise/217.npy`

SO noise is drawn from the SO LAT v3.1 temperature noise power spectrum (mode 2, elevation 50°, f_sky=0.4). 90 and 150 GHz are correlated (Cholesky decomposition); 217 GHz is independent. **The SO noise index does NOT need to match the sky patch index — sample it randomly.**

**Planck FFP10 noise** — 100 MC realizations per high-frequency channel
Shape per file: **(1523, 256, 256)**, dtype **float64**
Filename pattern: `planck_noise_{freq}_{i}.npy`, freq ∈ {353, 545, 857}, i ∈ {0, …, 99}

- `/home/node/data/compsep_data/cut_maps/planck_noise/planck_noise_353_{0..99}.npy` — units: K_CMB → multiply by 1e6 to get µK_CMB
- `/home/node/data/compsep_data/cut_maps/planck_noise/planck_noise_545_{0..99}.npy` — units: MJy/sr → multiply by 1e6 * jysr2uk(545) to get µK_CMB
- `/home/node/data/compsep_data/cut_maps/planck_noise/planck_noise_857_{0..99}.npy` — units: MJy/sr → multiply by 1e6 * jysr2uk(857) to get µK_CMB

**CRITICAL: For Planck noise, the patch index MUST match the sky patch index (i_patch == i_planck_patch_dim). Load with `np.load(...)[ i_patch ]`.**

---

## Spectral Response and Utility Functions

A `utils.py` module is available at `/home/node/data/compsep_data/utils.py`. **Always import using the absolute path:**

```python
import sys
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
```

Key functions:
- `utils.tsz(freq_ghz)` — tSZ spectral response in µK_CMB per Compton-y unit. Example values: tsz(90)≈-4.35e6, tsz(150)≈-2.60e6, tsz(217)≈-2.28e4, tsz(353)≈+3.05e6
- `utils.ksz(freq_ghz)` — kSZ spectral response in µK_CMB (≈ -T_CMB_µK ≈ -2.726e6, frequency-independent)
- `utils.jysr2uk(freq_ghz)` — converts Jy/sr to µK_CMB for CIB at given frequency
- `utils.powers(a, b, ps=10, ell_n=199, window_alpha=None)` — flat-sky angular auto/cross power spectrum

**IMPORTANT — `utils.powers` usage notes:**
- Always pass `window_alpha=0.5` (Hann taper) when computing cross-power spectra or the cross-correlation coefficient r_ℓ. Without windowing, the raw flat-sky FFT produces r_ℓ ≈ 0 even for perfectly correlated maps due to edge effects and phase artifacts. This is a known numerical issue — do NOT interpret r_ℓ = 0 without windowing as a true absence of correlation.
- Verify the estimator first: compute the power spectrum of Gaussian white noise and confirm it matches the theoretical expectation (σ² × pixel_area). If not, the pixel-to-physical mapping is wrong.

---

## Critical Signal Properties — Read Before Designing Any Analysis

### Dynamic range warning (ESSENTIAL)
The tSZ signal is orders of magnitude weaker than the other components:

| Component | Typical variance |
|-----------|-----------------|
| Lensed CMB | ~1.25×10⁴ µK² |
| CIB at 857 GHz | ~3.35×10¹⁰ (Jy/sr)² |
| tSZ (Compton-y) | ~3.28×10⁻¹² |
| kSZ | ~1.57×10⁻¹² |

**The tSZ signal is 10¹⁵ times weaker in variance than the CIB at 857 GHz and 10⁷ times weaker than the CMB.**

Consequences for machine learning:
1. **Always standardize the tSZ target maps** (zero mean, unit variance) before training. Using raw Compton-y values as training targets will cause the network to treat the signal as numerical noise and collapse to predicting zero.
2. **Never use a global L1/MSE loss on raw tSZ maps** — the background pixels (y ≈ 0) will dominate and the model will learn to predict a flat, near-zero map (model collapse). Use a weighted loss that upweights cluster pixels (y > 10⁻⁷) by a factor of 10³–10⁴, or use a focal/masked loss.
3. **The tSZ signal is spatially extremely sparse** — massive galaxy clusters occupy <1% of pixels. Any loss function that treats all pixels equally will be dominated by empty-sky pixels.

### Beam convolution (already applied to stacked maps)
The `stacked_*` maps are already beam-convolved. The ground truth (`tsz.npy`, `lensed_cmb.npy`, etc.) has a 1 arcmin beam. The SR task is to recover 1-arcmin features from beam-smeared inputs:
- At ℓ > 3000 (< 3.6 arcmin), the SO LAT signal is beam-suppressed and noise-dominated — deterministic pixel-phase recovery is information-theoretically impossible at these scales. Do NOT use a pixel-wise loss to penalize the model for failing to recover high-ℓ phases.
- The cross-correlation coefficient r_ℓ → 0 at high ℓ is physically expected, not a pipeline bug. Report the transfer function T(ℓ) = P_cross(pred,true)/P_auto(true) instead.

### Spectral loss warning
A logarithmic spectral consistency loss (penalizing |log C_ℓ^pred - log C_ℓ^true|) will cause catastrophic hallucination. The model will inject non-physical structures to satisfy global power constraints in noise-dominated patches, producing a hallucination fraction of >6000% on pure noise inputs. If a spectral loss is used, it must be:
- Linear (not logarithmic)
- Applied only to ℓ > 1000 (not low ℓ which is dominated by the CMB)
- Applied only to high-SNR multipole bins

---

## Correct Loading Pattern

```python
import sys, os
sys.path.insert(0, '/home/node/data/compsep_data')
import utils
import numpy as np

os.environ['OMP_NUM_THREADS'] = '16'

BASE = '/home/node/data/compsep_data/cut_maps'
n_patch = 1523
n_planck = 100
n_so = 3000

rng = np.random.default_rng(seed=42)
i_so_indices = rng.integers(0, n_so, size=n_patch)       # random SO noise per patch
i_planck_indices = rng.integers(0, n_planck, size=n_patch) # random Planck MC per patch

# Load a single patch (index i_patch):
i_patch = 0
i_so = i_so_indices[i_patch]
i_planck = i_planck_indices[i_patch]

signal_90  = np.load(f'{BASE}/stacked_90.npy')[i_patch]   # µK_CMB
signal_150 = np.load(f'{BASE}/stacked_150.npy')[i_patch]  # µK_CMB
signal_217 = np.load(f'{BASE}/stacked_217.npy')[i_patch]  # µK_CMB
signal_353 = np.load(f'{BASE}/stacked_353.npy')[i_patch]  # µK_CMB
signal_545 = np.load(f'{BASE}/stacked_545.npy')[i_patch]  # µK_CMB
signal_857 = np.load(f'{BASE}/stacked_857.npy')[i_patch]  # µK_CMB

noise_90  = np.load(f'{BASE}/so_noise/90.npy')[i_so]      # µK_CMB
noise_150 = np.load(f'{BASE}/so_noise/150.npy')[i_so]     # µK_CMB
noise_217 = np.load(f'{BASE}/so_noise/217.npy')[i_so]     # µK_CMB

pn_353_raw = np.load(f'{BASE}/planck_noise/planck_noise_353_{i_planck}.npy')[i_patch]
pn_545_raw = np.load(f'{BASE}/planck_noise/planck_noise_545_{i_planck}.npy')[i_patch]
pn_857_raw = np.load(f'{BASE}/planck_noise/planck_noise_857_{i_planck}.npy')[i_patch]
noise_353 = pn_353_raw * 1e6                           # K_CMB → µK_CMB
noise_545 = pn_545_raw * 1e6 * utils.jysr2uk(545)     # MJy/sr → µK_CMB
noise_857 = pn_857_raw * 1e6 * utils.jysr2uk(857)     # MJy/sr → µK_CMB

obs_90  = signal_90  + noise_90
obs_150 = signal_150 + noise_150
obs_217 = signal_217 + noise_217
obs_353 = signal_353 + noise_353
obs_545 = signal_545 + noise_545
obs_857 = signal_857 + noise_857

# Ground truth
tsz_map = np.load(f'{BASE}/tsz.npy')[i_patch]        # Compton-y, dimensionless
cib_353 = np.load(f'{BASE}/cib_353.npy')[i_patch]    # Jy/sr
```

**ALWAYS print explicit completion messages at the end of every script step**, e.g.:
```python
print(f'Step N completed successfully. Output shape: {array.shape}, saved to data/output.npy')
```
This is required for the pipeline controller to detect step completion.

**Do NOT save PyTorch model weights (.pth files) to the experiment output directory** — they exceed GitHub's 100 MB file size limit and will block git pushes. Save model weights to `/tmp/` instead, e.g. `torch.save(model.state_dict(), '/tmp/best_model.pth')`.

---

## Patch Geometry

- 1523 patches, 5°×5°, 256×256 pixels, ≈1.17 arcmin/pixel
- Full-sky tessellation at 5° step spacing (no galactic cut)
- Gnomonic (flat-sky) projection, bilinear interpolation from HEALPix nside=8192

## Hardware and Environment

- Python: `/opt/denario-venv/bin/python`
- 64 vCPUs (AMD Ryzen Threadripper PRO 9995WX), 128 GB RAM
- NVIDIA RTX PRO 6000 Blackwell Edition, 96 GB VRAM, CUDA 13.0
- PyTorch GPU: use `device='cuda'`
- Multiprocessing: limit to ~8–16 workers; set `OMP_NUM_THREADS=16`
- Load data patch-by-patch or in small batches; avoid loading all 1523 patches simultaneously

## Suggested Analyses

Based on prior work, the following approaches are known to work well on this dataset:

1. **Constrained ILC (cILC)** as a linear baseline — null CMB and kSZ while preserving tSZ. Use as the benchmark to beat.
2. **Multi-scale U-Net with gated cross-attention** — use SO LAT bands (90/150/217) as primary input and Planck HFI CIB bands (353/545/857) as auxiliary spatial priors. The CIB is spatially correlated with tSZ because both trace dark matter potential wells.
3. **Spectral difference features** — compute beam-matched difference maps (e.g., 150–90 GHz, 217–150 GHz) to suppress CMB while highlighting the tSZ spectral signature before feeding into the network.
4. **Curriculum learning** — train first on top-20% highest-tSZ patches (massive clusters) before expanding to the full dataset.
5. **Evaluate using**: transfer function T(ℓ) with patch-to-patch error bars, Y_SZ–M scatter (use peak tSZ as mass proxy), and residual-CIB cross-correlation to check for foreground leakage.
