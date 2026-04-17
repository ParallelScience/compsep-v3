1. **Covariance Modeling and Estimator Validation**:
    - Compute auto- and cross-power spectra for all 6 frequency channels, the tSZ ground truth, and the CIB maps using `utils.powers` with `window_alpha=0.5`.
    - Construct the signal covariance matrix $S_\ell$ by explicitly including the cross-power spectrum $P_{\ell, \text{tSZ-CIB}}$ as a signal component.
    - Define the noise covariance matrix $N_\ell$ using SO and Planck noise realizations, ensuring the SO 90/150 GHz correlation is captured.
    - Validate the estimator by confirming that the power spectrum of Gaussian white noise matches the theoretical expectation ($\sigma^2 \times \text{pixel\_area}$).

2. **Bias-Corrected MWF Construction**:
    - Define the MWF operator $W_\ell = (S_{\ell, \text{obs}} + N_\ell)^{-1} S_{\ell, \text{tSZ-obs}}$ with Tikhonov regularization for numerical stability.
    - Develop a multi-dimensional bias-correction lookup table $f(Y_{obs}, \sigma_{noise}, I_{CIB})$ by comparing MWF-reconstructed Compton-y values against ground-truth `tsz.npy`.
    - Ensure the table is binned by local noise variance and local CIB intensity to account for heterogeneous survey conditions.

3. **Constrained MWF for Spatial Fidelity Benchmark**:
    - Implement a "constrained" MWF variant that utilizes ground-truth cluster locations to recover core features.
    - Use this as a validation-only benchmark to establish the theoretical upper limit of the filter's spatial fidelity, distinct from the blind detection pipeline.

4. **Empirical False Detection Rate (FDR) Calibration**:
    - Run the detection pipeline on 3000 pure noise realizations (SO+Planck noise only) using the same sampling logic for noise indices as the signal patches.
    - Calculate the FDR as a function of SNR to establish a statistically rigorous detection threshold that accounts for the "look-elsewhere effect" across the 1523 patches.

5. **Cluster Detection and Peak-Finding**:
    - Perform peak-finding on the standard (unconstrained) MWF output map using the threshold determined in Step 4.
    - Evaluate the detection statistic against a secondary spatial matched filter to determine if profile-specific filtering is required for improved sensitivity.

6. **Mass Calibration and Performance Quantification**:
    - Apply the multi-dimensional bias-correction lookup table to the detected cluster candidates to recover the integrated Compton-Y.
    - Quantify detection completeness and purity as a function of halo mass, local CIB intensity, and local noise variance.
    - Measure the mass bias of the recovered integrated-Y to assess the accuracy of the MWF-based mass estimation.

7. **Computational Optimization**:
    - Vectorize the covariance matrix inversion and the application of the MWF across the 1523 patches to leverage the available 64 vCPUs and GPU memory.
    - Ensure all intermediate outputs are saved and completion messages are printed for each step.

8. **Final Reporting**:
    - Compile results, focusing on the impact of including $P_{\ell, \text{tSZ-CIB}}$ on signal separation and the effectiveness of the multi-dimensional bias-correction table.
    - Document the transfer function $T(\ell)$ and the final completeness/purity curves.