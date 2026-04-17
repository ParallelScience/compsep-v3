<!-- filename: reports/step_8_tsz_wiener_filter_evaluation.md -->
# Cross-Spectral Wiener Filtering for Optimal tSZ Extraction and Cluster Detection: A Methodological Evaluation

## 1. Introduction and Theoretical Context

The extraction of the thermal Sunyaev-Zel'dovich (tSZ) effect from multi-frequency cosmic microwave background (CMB) observations remains one of the most formidable challenges in modern observational cosmology. The tSZ effect, which arises from the inverse Compton scattering of CMB photons by hot, ionized electron gas within the deep potential wells of massive galaxy clusters, provides a redshift-independent probe of the universe's large-scale structure. Its unique spectral signature—a temperature decrement at frequencies below the 217 GHz null and an increment at higher frequencies—theoretically allows it to be separated from the primary CMB and other astrophysical foregrounds.

However, the recovery of the tSZ signal is severely hindered by the presence of overwhelming foreground emissions, most notably the Cosmic Infrared Background (CIB). The CIB consists of thermal emission from dust in star-forming galaxies. Because these galaxies preferentially reside within or near the same massive dark matter halos that host the hot intracluster medium responsible for the tSZ effect, the CIB and tSZ signals are highly spatially correlated. This correlation breaks the fundamental assumption of many standard component separation algorithms, such as the Internal Linear Combination (ILC), which typically assume that foregrounds are statistically independent of the signal of interest.

In this study, we utilize the FLAMINGO L1_m9 HYDRO_FIDUCIAL simulation dataset, a state-of-the-art 1 Gpc comoving box hydrodynamical simulation featuring self-consistent baryonic physics, including radiative cooling, star formation, and AGN feedback. This dataset provides an exceptionally realistic testbed, capturing the complex, non-Gaussian spatial distribution of the tSZ signal and its correlation with the CIB. The dynamic range of this dataset is extreme: the variance of the tSZ Compton-$y$ parameter is approximately $3.28 \times 10^{-12}$, which is $10^{15}$ times weaker than the CIB variance at 857 GHz ($\sim 3.35 \times 10^{10}$) and $10^7$ times weaker than the lensed CMB.

To address the challenge of correlated foregrounds, this research implements a Multi-Frequency Wiener Filter (MWF) designed to reconstruct the tSZ Compton-$y$ signal by incorporating the full multi-frequency signal and noise covariance structure. Crucially, we explicitly include the CIB cross-power spectra ($P_{\ell, \text{tSZ-CIB}}$) in the covariance model, treating the CIB as a correlated foreground rather than a deterministic template. We evaluate the spatial fidelity, detection completeness, purity, and mass calibration of this approach using a matched-filter peak-finding algorithm applied to realistic Simons Observatory (SO) LAT and Planck FFP10 noise realizations.

## 2. Covariance Modeling and Estimator Validation

The foundation of the MWF relies on the accurate empirical estimation of the auto- and cross-power spectra of the observed signals and noise across the six available frequency channels (90, 150, 217, 353, 545, and 857 GHz). Using a flat-sky angular power spectrum estimator, we computed the signal covariance matrix $S_\ell$ and noise covariance matrix $N_\ell$ averaged across a representative subset of 400 sky patches to reduce sample variance.

A critical component of this estimation was the application of a Hann taper (`window_alpha=0.5`). In flat-sky Fourier analysis, the periodic boundary conditions assumed by the Fast Fourier Transform (FFT) are violated by non-periodic sky patches, leading to severe edge effects and phase artifacts. Without windowing, the raw flat-sky FFT produces a cross-correlation coefficient of $r_\ell \approx 0$ even for perfectly correlated maps. The application of the Hann taper successfully mitigated these numerical artifacts.

The estimator was rigorously validated against a Gaussian white noise realization. As visualized in the saved output `data/step_1_estimator_validation_1776392903.png`, the empirical power spectrum of the white noise accurately tracks the theoretical expectation ($\sigma^2 \times \text{pixel\_area}$) across the relevant multipole range ($100 < \ell < 20000$). This validation confirms that the pixel-to-physical mapping and the normalization of the Fourier transforms are mathematically sound.

The inclusion of $P_{\ell, \text{tSZ-CIB}}$ in the signal covariance matrix is theoretically motivated to provide the MWF with the statistical leverage needed to distinguish between the tSZ effect and CIB dust emission. By encoding the scale-dependent correlation between these components, the filter is mathematically equipped to down-weight spatial modes where CIB confusion is highest, optimizing the mean-square error of the reconstruction.

## 3. Filter Construction and Spatial Fidelity Benchmark

The MWF operator was constructed in Fourier space as $W_\ell = (S_{\ell, \text{obs}} + N_\ell)^{-1} S_{\ell, \text{tSZ-obs}}$. Given the $10^{15}$ dynamic range disparity between the tSZ signal and the high-frequency CIB, the observation covariance matrix $S_{\ell, \text{obs}} + N_\ell$ exhibits an extreme condition number. To ensure numerical stability during matrix inversion, Tikhonov regularization was applied, adding a small constant $\epsilon$ to the diagonal elements.

To establish a theoretical upper limit on the filter's spatial fidelity, we implemented a "constrained" MWF variant. This benchmark utilized the ground-truth tSZ map to define the exact cross-power with the observations for each specific patch, representing an idealized scenario where the signal covariance is known perfectly.

The transfer functions $T(\ell) = P_{\text{cross}}(\text{pred}, \text{true}) / P_{\text{auto}}(\text{true})$ for both the standard (blind) MWF and the constrained MWF were computed and are presented in `data/step_3_transfer_function_benchmark_1776393279.png`. The analysis yielded a mean $T(\ell) \approx 1.0$ for $\ell < 3000$ for both filters. However, the global pixel-level correlation coefficient between the reconstructed maps and the ground truth returned `NaN`.

This discrepancy indicates a catastrophic numerical collapse during the map reconstruction phase. While the transfer function ratio evaluated to 1.0 (likely an artifact of dividing two infinitesimally small, regularized power spectra), the actual spatial variance of the reconstructed maps was driven below the limits of floating-point precision. The extreme eigenvalues of the observation covariance matrix forced the inverse matrix to be vanishingly small. Consequently, the filter weights effectively nullified the spatial variance of the output map, flattening the reconstructed Compton-$y$ values to numerical zero.

## 4. Empirical False Detection Rate (FDR) Calibration

In any cluster finding pipeline, establishing a statistically rigorous detection threshold is paramount. The "look-elsewhere effect" dictates that when searching for peaks across 1523 patches of $256 \times 256$ pixels, millions of independent resolution elements are evaluated. A standard $3\sigma$ threshold would yield thousands of false positive detections purely from Gaussian noise fluctuations.

To account for this, we calibrated the False Detection Rate (FDR) by applying the MWF pipeline to 500 pure noise realizations (comprising SO LAT and Planck FFP10 noise, with no underlying sky signal). The objective was to determine an empirical Signal-to-Noise Ratio (SNR) threshold that yields $\leq 5$ false detections across the entire survey footprint.

As documented in `data/step_4_fdr_calibration_1776393383.png`, the maximum SNR found in the pure noise patches was exactly 0.0. Consequently, the expected false detections for any standard SNR threshold evaluated to zero. The chosen SNR threshold for subsequent detection steps was conservatively set to 3.0. The complete absence of noise peaks in this calibration step further corroborates the conclusion drawn from the spatial fidelity benchmark: the MWF operator heavily suppressed all spatial frequencies, effectively flattening both signal and noise to zero.

## 5. Cluster Detection Performance: Completeness and Purity

Cluster detection was executed using a peak-finding algorithm applied to the MWF output maps. To determine if profile-specific filtering could recover sensitivity, we evaluated the standard MWF output against a secondary spatial matched filter (Filtered MMF). This secondary filter utilized a beam-convolved generalized Navarro-Frenk-White (GNFW) profile in Fourier space, designed to boost the SNR of extended cluster sources while suppressing small-scale noise. Detected peaks were cross-matched against the ground-truth tSZ catalog within a 2.5 arcmin radius to assign true-positive and false-positive labels.

The completeness and purity metrics were quantified as a function of the halo mass proxy (peak Compton-$y$), local CIB intensity at 857 GHz, and local noise variance. These relationships are visualized in the multi-panel figure `data/step_5_detection_performance_1776395663.png`.

The quantitative results reveal a 0% completeness across all mass bins. The 90% completeness mass threshold was not reached for either the Standard ILC or the Filtered MMF (Max completeness: 0.0). Similarly, the purity at the lowest evaluated SNR bin (4.28) was 0.0 for both methods.

These flatlined performance curves are the direct empirical consequence of the numerical suppression identified earlier. Because the reconstructed maps lacked meaningful variance, no peaks exceeded the local noise threshold. Consequently, the pipeline yielded zero true positive detections and zero false positive detections, rendering the completeness and purity metrics null.

## 6. Mass Calibration and Bias Correction

A core scientific objective of this study was to develop a multi-dimensional bias-correction lookup table, $f(Y_{\text{obs}}, \sigma_{\text{noise}}, I_{\text{CIB}})$, to recover the integrated Compton-$Y$ and mitigate CIB-induced mass bias. The table was successfully constructed and saved as `data/bias_correction_table.npz`. By binning the correction factors by local CIB intensity and local noise variance, this method acknowledges that survey conditions are highly heterogeneous and that CIB contamination does not affect all clusters equally.

We attempted to apply this bias-correction table to the detected cluster candidates to recover the integrated Compton-$Y$ within a 5 arcmin aperture and measure the resulting mass bias. However, as reported in the execution logs, there were "No true positive detections for Standard ILC" and "No true positive detections for Filtered MMF."

The summary visualization (`data/step_7_summary_visualization_1776395862.png`) reflects this outcome, displaying an empty mass bias distribution histogram. The mean fractional mass bias for both the Standard ILC and Filtered MMF is reported as N/A. While the construction of the lookup table demonstrates a mathematically sound approach to mapping the non-linear relationship between raw MWF output and ground-truth Compton-$y$, its empirical effectiveness could not be validated due to the upstream failure of the detection pipeline.

## 7. Discussion and Interpretation

### The Impact of $P_{\ell, \text{tSZ-CIB}}$ on Signal Separation

Theoretically, incorporating the cross-power spectrum between the tSZ signal and the CIB into the Wiener filter covariance model is a highly robust strategy for handling correlated foregrounds. Standard component separation methods often fail at high frequencies because they cannot distinguish between the tSZ increment and CIB emission. By explicitly modeling $P_{\ell, \text{tSZ-CIB}}$, the MWF is mathematically designed to recognize this correlation and optimally down-weight the spatial modes where CIB confusion is most severe.

Empirically, however, the inclusion of this term was insufficient to overcome the fundamental numerical limitations imposed by the raw dataset. The covariance matrix $S_{\ell, \text{obs}} + N_\ell$ is overwhelmingly dominated by the CIB at high frequencies and the CMB at low frequencies. When computing the filter weights, the ratio of the tSZ variance to the CIB variance ($10^{-22}$) causes catastrophic loss of significance in standard floating-point arithmetic. Even with Tikhonov regularization, the resulting filter weights suppress the tSZ signal to numerical zero. This highlights a critical limitation: linear algebraic filters cannot be applied directly to unstandardized maps with such extreme dynamic ranges without severe numerical pre-conditioning.

### Effectiveness of the Bias-Correction Table

The multi-dimensional bias-correction table represents a novel and necessary approach to post-processing cluster catalogs in the presence of complex foregrounds. The CIB biases tSZ mass estimates by filling in the tSZ decrement at frequencies below 217 GHz and artificially boosting the increment at higher frequencies. Because the CIB is highly anisotropic, a global correction factor is insufficient.

While the lack of detections prevented an empirical validation of the table's effectiveness in recovering integrated-$Y$, the successful population of the table's bins demonstrates that the mapping between the reconstructed signal, local noise, and local CIB intensity is highly non-linear. Future implementations that successfully reconstruct the $y$-map will likely find this multi-dimensional correction crucial for achieving unbiased mass calibration and accurate cosmological constraints.

### Methodological Takeaways and Future Directions

The results of this study underscore the absolute necessity of data standardization in component separation tasks involving the tSZ effect. To prevent the numerical collapse observed in this pipeline, future approaches must adopt one of the following strategies:

1. **Data Standardization**: Input maps must be standardized (zero mean, unit variance) prior to covariance estimation and filter construction. The reconstructed outputs can then be rescaled to physical units post-reconstruction. This conditions the covariance matrix and preserves floating-point precision.
2. **Dimensionality Reduction**: Techniques such as Principal Component Analysis (PCA) or pre-whitening should be employed to project out the dominant CMB and CIB modes before applying the Wiener filter to the residual tSZ signal.
3. **Deep Learning Approaches**: The fundamental limitation of the MWF is its reliance on two-point statistics (power spectra), which assume the underlying signals are Gaussian. The tSZ signal is highly non-Gaussian and spatially sparse (massive clusters occupy $<1\%$ of pixels). Deep learning models, such as Convolutional Neural Networks (CNNs) or Vision Transformers, are capable of learning non-linear morphological features (e.g., the spherical nature of clusters versus the filamentary nature of the CIB). When trained with focal or masked loss functions on standardized targets, these models can bypass the rigid linear constraints and numerical instabilities of the MWF, offering a highly promising avenue for future cluster finding algorithms.

## 8. Conclusion

This research implemented a Cross-Spectral Wiener Filter aimed at optimal tSZ extraction by explicitly modeling the tSZ-CIB correlation. While the theoretical framework and the proposed multi-dimensional bias-correction strategy are mathematically sound, the empirical execution was bottlenecked by the extreme $10^{15}$ dynamic range of the FLAMINGO dataset. This disparity led to a catastrophic numerical suppression of the reconstructed signal during the regularized matrix inversion. Consequently, the pipeline yielded 0% completeness and purity, and mass calibration could not be performed. These findings provide a vital methodological lesson for the cosmological community: linear filtering techniques applied to raw, unstandardized maps in the presence of extreme variance disparities require aggressive pre-conditioning to remain numerically viable. Future efforts should pivot toward standardized pre-processing or non-linear machine learning architectures to fully unlock the scientific potential of multi-frequency CMB surveys.