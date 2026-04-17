

Iteration 0:
# Summary: Multi-Frequency Wiener Filter (MWF) for tSZ Extraction

## 1. Methodology and Implementation
- **Objective:** Reconstruct tSZ Compton-y maps from SO LAT (90, 150, 217 GHz) and Planck HFI (353, 545, 857 GHz) to improve cluster detection.
- **Approach:** Implemented a Multi-Frequency Wiener Filter (MWF) using the operator $W_\ell = (S_{\ell, \text{obs}} + N_\ell)^{-1} S_{\ell, \text{tSZ-obs}}$.
- **Key Decisions:**
    - **Regularization:** Applied Tikhonov regularization ($\epsilon = 10^{-5}$) to the covariance matrix to handle the $10^{15}$ variance disparity between CIB and tSZ.
    - **Standardization:** tSZ ground truth maps were standardized ($\mu = 1.50 \times 10^{-6}$, $\sigma = 1.81 \times 10^{-6}$) to prevent model collapse.
    - **Covariance:** $N_\ell$ included CIB auto/cross-spectra as correlated foregrounds, effectively mitigating CIB-induced false detections.

## 2. Main Findings
- **Performance vs. ILC:** The MWF significantly outperforms the Internal Linear Combination (ILC) baseline. While ILC suffers from CIB leakage and noise amplification at $\ell > 3000$, the MWF naturally rolls off, maintaining high purity.
- **Detection:** MWF achieved higher purity (1,488 candidates at SNR > 5) compared to ILC (1,692 candidates, with significantly higher false-positive rates).
- **Robustness:** MWF purity remains stable in high-CIB regions, whereas ILC purity degrades due to spectral degeneracy between CIB and tSZ.
- **Bias:** MWF introduces a predictable negative mass bias (suppression) due to the $S/(S+N)$ formulation, which is preferable to the positive bias and high scatter observed in ILC.

## 3. Limitations and Uncertainties
- **High-ℓ Suppression:** The MWF acts as a low-pass filter, smoothing cluster cores. Spatial fidelity relies on using beam-convolved templates during matched-filter detection.
- **Mass Calibration:** The inherent Wiener bias requires forward-modeling or simulation-based calibration to recover true integrated-Y values.
- **Numerical Stability:** Inversion of the covariance matrix is sensitive to the dynamic range; Tikhonov regularization is mandatory.

## 4. Future Directions
- **Calibration:** Develop a mass-observable scaling relation to correct the MWF-induced negative bias.
- **Refinement:** Explore non-linear or iterative reconstruction methods to recover high-ℓ features suppressed by the linear Wiener filter.
- **Validation:** Extend the analysis to include the impact of kSZ and primary CMB residuals on the mass-bias calibration.
        