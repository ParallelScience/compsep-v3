1. **Data Preprocessing and Normalization**:
   - Implement a data loader that performs on-the-fly noise injection: randomly sample SO noise indices and match Planck noise indices to the patch index. Apply unit conversions (K_CMB to µK_CMB) and frequency-dependent scaling.
   - Standardize the `tsz.npy` ground truth maps (zero mean, unit variance) and store the transformation parameters (mean/std) to allow for de-normalization of the final reconstructed y-map back to physical Compton-y units.
   - Verify the dynamic range of the input maps to ensure the 10¹⁵ variance disparity is handled using `float64` precision throughout the pipeline.

2. **Power Spectrum and Covariance Estimation**:
   - Compute auto- and cross-power spectra for all 6 frequency channels and the tSZ ground truth using `utils.powers` with `window_alpha=0.5` for all terms.
   - Perform this estimation on a representative subset of 500 patches to ensure statistical stability and computational efficiency.
   - Construct the total observed signal covariance matrix $S_{\ell, \text{obs}}$ (including CMB, tSZ, kSZ, and CIB) and the tSZ-specific cross-correlation vector $S_{\ell, \text{tSZ-obs}}$.
   - Validate the estimator by computing the power spectrum of Gaussian white noise to confirm it matches the theoretical expectation ($\sigma^2 \times \text{pixel\_area}$).

3. **Multi-Frequency Wiener Filter (MWF) Construction**:
   - Define the MWF operator $W_\ell = (S_{\ell, \text{obs}} + N_\ell)^{-1} S_{\ell, \text{tSZ-obs}}$, where $N_\ell$ includes the CIB auto- and cross-spectra as correlated foreground noise.
   - Apply Tikhonov regularization (adding a small diagonal term) to the covariance matrix $(S_{\ell, \text{obs}} + N_\ell)$ before inversion to ensure numerical stability against the 10¹⁵ dynamic range disparity.
   - Parallelize the construction of these matrices across available CPU cores.

4. **Cluster Profile Modeling**:
   - Derive a frequency-specific, beam-convolved cluster profile by stacking `tsz.npy` maps centered on high-mass halos.
   - Ensure the profile accounts for the specific beam FWHM of each frequency channel (SO LAT vs. Planck HFI) to maintain spatial fidelity.

5. **Reconstruction and Validation**:
   - Apply the MWF operator in the harmonic domain to the multi-frequency observation vector to produce the reconstructed Compton-y map.
   - Compute the empirical transfer function $T(\ell) = P_{\text{cross}}(\text{pred}, \text{true}) / P_{\text{auto}}(\text{true})$ and compare it against the theoretical expectation, masking noise-dominated regimes ($\ell > 3000$) to avoid artifacts.
   - Evaluate reconstruction quality using a linear spectral loss for $\ell > 1000$ and compare performance against a standard Internal Linear Combination (ILC) baseline.

6. **Cluster Detection Pipeline**:
   - Perform peak-finding on the reconstructed y-map using a thresholding approach (e.g., SNR > 5).
   - Use the beam-convolved cluster profile as a template to refine candidate detections and handle overlapping profiles.
   - Account for the "look-elsewhere effect" when defining the detection threshold.

7. **Performance Quantification**:
   - Calculate detection completeness and purity by comparing the candidate catalog against the ground-truth catalog derived from `tsz.npy`.
   - Evaluate performance as a function of halo mass (using peak Compton-y as a proxy) and local CIB intensity.
   - Measure the mass bias in the recovered integrated-Y for detected clusters.

8. **Final Reporting**:
   - Compile the results into a summary of the MWF performance, including the transfer function, completeness/purity curves, and the impact of CIB contamination.
   - Ensure all intermediate outputs are saved to the designated directory and completion messages are printed for each step.