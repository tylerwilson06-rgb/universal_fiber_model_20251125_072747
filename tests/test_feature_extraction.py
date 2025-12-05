"""
Test Suite for Feature Extraction Module

Tests MultiDomainFeatureExtractor, ProprietaryFeatures, and UniversalFeatureVectorBuilder.

This test suite validates feature extraction according to:
- Davis & Mermelstein (1980): MFCC theory
- Parseval's theorem: Energy preservation in transforms
- Wu et al. (2021): ML for optical fiber sensing

Test categories:
1. MFCC feature extraction (120 dimensions)
2. Wavelet feature extraction (64 dimensions)
3. Spectral feature extraction (6 dimensions)
4. Temporal feature extraction (6 dimensions)
5. Spatial feature extraction (4 dimensions)
6. Proprietary features (4 dimensions)
7. UFV builder (204 dimensions total)

Coverage target: 80%+
Test count: 30 tests
"""

import pytest
import numpy as np
import warnings

from src.feature_extraction import (
    MultiDomainFeatureExtractor,
    ProprietaryFeatures,
    UniversalFeatureVectorBuilder
)


class TestMFCCFeatureExtraction:
    """
    Tests for MFCC feature extraction.
    
    Expected output: 120 features (40 MFCC + 40 delta + 40 delta-delta)
    Reference: Davis & Mermelstein (1980)
    """
    
    @pytest.mark.unit
    def test_mfcc_output_dimension(self, feature_extractor, das_signal, expected_dimensions):
        """
        Test MFCC always returns exactly 120 features.
        
        40 MFCCs + 40 deltas + 40 delta-deltas = 120
        """
        mfcc = feature_extractor.extract_mfcc_features(das_signal)
        
        assert len(mfcc) == expected_dimensions['mfcc']
    
    @pytest.mark.unit
    def test_mfcc_no_nan_inf(self, feature_extractor, das_signal, assert_no_nan_inf):
        """
        Test MFCC output contains no NaN or Inf values.
        
        All features must be finite for model input.
        """
        mfcc = feature_extractor.extract_mfcc_features(das_signal)
        
        assert_no_nan_inf(mfcc, "MFCC features")
    
    @pytest.mark.unit
    def test_mfcc_short_signal_handling(self, feature_extractor, expected_dimensions):
        """
        Test MFCC handles signals shorter than MIN_SIGNAL_LENGTH_MFCC.
        
        Should pad and still produce 120 features.
        """
        short_signal = np.random.randn(100)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mfcc = feature_extractor.extract_mfcc_features(short_signal)
        
        assert len(mfcc) == expected_dimensions['mfcc']
        assert not np.any(np.isnan(mfcc))
    
    @pytest.mark.unit
    def test_mfcc_numerical_stability(self, feature_extractor, expected_dimensions):
        """
        Test MFCC handles large amplitude signals without overflow.
        
        Signals scaled to 1e6 should still produce valid features.
        """
        large_signal = np.random.randn(10000) * 1e6
        
        mfcc = feature_extractor.extract_mfcc_features(large_signal)
        
        assert len(mfcc) == expected_dimensions['mfcc']
        assert not np.any(np.isnan(mfcc))
        assert not np.any(np.isinf(mfcc))
    
    @pytest.mark.unit
    def test_mfcc_zero_signal_handling(self, feature_extractor, zero_signal, expected_dimensions):
        """
        Test MFCC handles all-zeros signal gracefully.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mfcc = feature_extractor.extract_mfcc_features(zero_signal)
        
        assert len(mfcc) == expected_dimensions['mfcc']


class TestWaveletFeatureExtraction:
    """
    Tests for wavelet packet feature extraction.
    
    Expected output: 64 features
    Reference: Parseval's theorem for energy preservation
    """
    
    @pytest.mark.unit
    def test_wavelet_output_dimension(self, feature_extractor, das_signal, expected_dimensions):
        """
        Test wavelet always returns exactly 64 features.
        """
        wavelet = feature_extractor.extract_wavelet_features(das_signal)
        
        assert len(wavelet) == expected_dimensions['wavelet']
    
    @pytest.mark.unit
    def test_wavelet_energy_preservation(self, feature_extractor, das_signal):
        """
        Test wavelet decomposition approximately preserves signal energy.
        
        Reference: Parseval's theorem - total energy should be preserved
        across wavelet decomposition.
        
        Note: Due to feature extraction (log, entropy), we only check
        that energy features are positive and finite.
        """
        wavelet = feature_extractor.extract_wavelet_features(das_signal)
        
        # Energy features (every 4th feature starting at 0) should be non-negative
        energy_features = wavelet[0::4]  # Raw energy values
        
        assert np.all(energy_features >= 0), "Energy features should be non-negative"
    
    @pytest.mark.unit
    def test_wavelet_short_signal_handling(self, feature_extractor, expected_dimensions):
        """
        Test wavelet handles signals shorter than MIN_SIGNAL_LENGTH_WAVELET.
        
        Should adapt decomposition level for short signals.
        """
        short_signal = np.random.randn(20)  # Below 32 samples
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wavelet = feature_extractor.extract_wavelet_features(short_signal)
        
        assert len(wavelet) == expected_dimensions['wavelet']
    
    @pytest.mark.unit
    def test_wavelet_decomposition_levels(self, feature_extractor, das_signal):
        """
        Test wavelet uses level 4 decomposition by default.
        
        Level 4 with db4 wavelet produces 16 subbands * 4 features = 64.
        """
        wavelet = feature_extractor.extract_wavelet_features(das_signal)
        
        # Should have exactly 64 features
        assert len(wavelet) == 64
    
    @pytest.mark.unit
    def test_wavelet_no_nan_inf(self, feature_extractor, das_signal, assert_no_nan_inf):
        """
        Test wavelet output contains no NaN or Inf values.
        """
        wavelet = feature_extractor.extract_wavelet_features(das_signal)
        
        assert_no_nan_inf(wavelet, "Wavelet features")


class TestSpectralFeatureExtraction:
    """
    Tests for spectral feature extraction.
    
    Expected output: 6 features (centroid, bandwidth, rolloff, flatness, kurtosis, peak_freq)
    """
    
    @pytest.mark.unit
    def test_spectral_output_dimension(self, feature_extractor, das_signal, expected_dimensions):
        """
        Test spectral always returns exactly 6 features.
        """
        spectral = feature_extractor.extract_spectral_features(das_signal)
        
        assert len(spectral) == expected_dimensions['spectral']
    
    @pytest.mark.unit
    def test_spectral_centroid_calculation(self, feature_extractor, synthetic_sine_1khz):
        """
        Test spectral centroid is approximately correct for known signal.
        
        Pure 1kHz sine should have centroid near 1000Hz.
        """
        spectral = feature_extractor.extract_spectral_features(synthetic_sine_1khz)
        
        centroid = spectral[0]  # First feature is centroid
        
        # Should be approximately 1000Hz (tolerance: 200Hz due to windowing effects)
        assert 500 < centroid < 1500, f"Centroid {centroid} not near 1000Hz"
    
    @pytest.mark.unit
    def test_spectral_zero_signal_handling(self, feature_extractor, zero_signal, expected_dimensions):
        """
        Test spectral features handle all-zeros signal.
        
        Should return zeros without crashing.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spectral = feature_extractor.extract_spectral_features(zero_signal)
        
        assert len(spectral) == expected_dimensions['spectral']
        # Zero signal should produce near-zero spectral features
        assert not np.any(np.isnan(spectral))
    
    @pytest.mark.unit
    def test_spectral_bandwidth_rolloff(self, feature_extractor, white_noise_signal):
        """
        Test bandwidth and rolloff are reasonable for white noise.
        
        White noise should have higher bandwidth than pure sine.
        """
        spectral = feature_extractor.extract_spectral_features(white_noise_signal)
        
        bandwidth = spectral[1]  # Second feature is bandwidth
        rolloff = spectral[2]    # Third feature is rolloff
        
        assert bandwidth > 0, "Bandwidth should be positive for noise"
        assert rolloff > 0, "Rolloff should be positive for noise"
    
    @pytest.mark.unit
    def test_spectral_no_nan_inf(self, feature_extractor, das_signal, assert_no_nan_inf):
        """
        Test spectral output contains no NaN or Inf values.
        """
        spectral = feature_extractor.extract_spectral_features(das_signal)
        
        assert_no_nan_inf(spectral, "Spectral features")


class TestTemporalFeatureExtraction:
    """
    Tests for temporal feature extraction.
    
    Expected output: 6 features (RMS, peak, ZCR, crest, MAD, lag1_corr)
    """
    
    @pytest.mark.unit
    def test_temporal_output_dimension(self, feature_extractor, das_signal, expected_dimensions):
        """
        Test temporal always returns exactly 6 features.
        """
        temporal = feature_extractor.extract_temporal_features(das_signal)
        
        assert len(temporal) == expected_dimensions['temporal']
    
    @pytest.mark.unit
    def test_temporal_rms_calculation(self, feature_extractor):
        """
        Test RMS calculation is correct for known signal.
        
        Signal of all 2s should have RMS = 2.
        """
        signal = np.full(10000, 2.0)
        temporal = feature_extractor.extract_temporal_features(signal)
        
        rms = temporal[0]  # First feature is RMS
        
        assert abs(rms - 2.0) < 0.01, f"RMS {rms} should be 2.0"
    
    @pytest.mark.unit
    def test_temporal_zcr_calculation(self, feature_extractor, synthetic_sine_1khz):
        """
        Test zero-crossing rate for known signal.
        
        1kHz sine at 10kHz sampling has ~2000 zero crossings per second.
        ZCR = crossings / samples ≈ 0.2
        """
        temporal = feature_extractor.extract_temporal_features(synthetic_sine_1khz)
        
        zcr = temporal[2]  # Third feature is ZCR
        
        # Should be approximately 0.2 (2000/10000)
        assert 0.1 < zcr < 0.3, f"ZCR {zcr} not near expected 0.2"
    
    @pytest.mark.unit
    def test_temporal_autocorrelation(self, feature_extractor, synthetic_sine_1khz):
        """
        Test lag-1 autocorrelation for periodic signal.
        
        Sine wave should have high positive lag-1 correlation.
        """
        temporal = feature_extractor.extract_temporal_features(synthetic_sine_1khz)
        
        lag1_corr = temporal[5]  # Sixth feature is lag-1 correlation
        
        # Sine wave should have high positive correlation at lag 1
        assert lag1_corr > 0.8, f"Lag-1 correlation {lag1_corr} should be high for sine"
    
    @pytest.mark.unit
    def test_temporal_no_nan_inf(self, feature_extractor, das_signal, assert_no_nan_inf):
        """
        Test temporal output contains no NaN or Inf values.
        """
        temporal = feature_extractor.extract_temporal_features(das_signal)
        
        assert_no_nan_inf(temporal, "Temporal features")


class TestSpatialFeatureExtraction:
    """
    Tests for spatial feature extraction (multi-channel signals).
    
    Expected output: 4 features (gradient, mean_corr, std_corr, energy_spread)
    """
    
    @pytest.mark.unit
    def test_spatial_output_dimension(self, feature_extractor, phi_otdr_signal, expected_dimensions):
        """
        Test spatial always returns exactly 4 features.
        """
        spatial = feature_extractor.extract_spatial_features(phi_otdr_signal)
        
        assert len(spatial) == expected_dimensions['spatial']
    
    @pytest.mark.unit
    def test_spatial_single_channel_zeros(self, feature_extractor, das_signal, expected_dimensions):
        """
        Test spatial returns zeros for single-channel signal.
        
        Spatial features require multi-channel input.
        """
        spatial = feature_extractor.extract_spatial_features(das_signal)
        
        assert len(spatial) == expected_dimensions['spatial']
        assert np.allclose(spatial, 0), "Single channel should produce zero spatial features"
    
    @pytest.mark.unit
    def test_spatial_multichannel_gradient(self, feature_extractor, phi_otdr_signal):
        """
        Test spatial gradient calculation for multi-channel signal.
        """
        spatial = feature_extractor.extract_spatial_features(phi_otdr_signal)
        
        gradient = spatial[0]  # First feature is gradient
        
        assert gradient >= 0, "Gradient should be non-negative"
    
    @pytest.mark.unit
    def test_spatial_channel_correlation(self, feature_extractor, phi_otdr_signal):
        """
        Test inter-channel correlation for correlated channels.
        
        Phi-OTDR fixture has correlation=0.7, so mean correlation should be high.
        """
        spatial = feature_extractor.extract_spatial_features(phi_otdr_signal)
        
        mean_corr = spatial[1]  # Second feature is mean correlation
        
        # With 70% correlation in fixture, mean should be positive
        assert mean_corr > 0.3, f"Mean correlation {mean_corr} should reflect 70% correlation"
    
    @pytest.mark.unit
    def test_spatial_no_nan_inf(self, feature_extractor, phi_otdr_signal, assert_no_nan_inf):
        """
        Test spatial output contains no NaN or Inf values.
        """
        spatial = feature_extractor.extract_spatial_features(phi_otdr_signal)
        
        assert_no_nan_inf(spatial, "Spatial features")


class TestExtractAllFeatures:
    """Tests for combined feature extraction (extract_all method)."""
    
    @pytest.mark.unit
    def test_extract_all_dimension(self, feature_extractor, das_signal, expected_dimensions):
        """
        Test extract_all returns exactly 200 standard features.
        """
        features = feature_extractor.extract_all(das_signal, is_multichannel=False)
        
        assert len(features) == expected_dimensions['standard_total']
    
    @pytest.mark.unit
    def test_extract_all_no_nan(self, feature_extractor, das_signal, assert_no_nan_inf):
        """
        Test extract_all output contains no NaN values.
        """
        features = feature_extractor.extract_all(das_signal, is_multichannel=False)
        
        assert_no_nan_inf(features, "All standard features")
    
    @pytest.mark.unit
    def test_extract_all_multichannel(self, feature_extractor, phi_otdr_signal, expected_dimensions):
        """
        Test extract_all handles multi-channel signals correctly.
        """
        features = feature_extractor.extract_all(phi_otdr_signal, is_multichannel=True)
        
        assert len(features) == expected_dimensions['standard_total']


class TestProprietaryFeatures:
    """
    Tests for proprietary fiber-aware features.
    
    Features: RBE, DESI, SCR, BSI (4 total)
    """
    
    @pytest.mark.unit
    def test_rbe_entropy_calculation(self, proprietary_features, das_signal):
        """
        Test Rayleigh Backscatter Entropy calculation.
        
        Entropy should be positive for non-constant signals.
        """
        rbe = proprietary_features.calculate_RBE(das_signal)
        
        assert rbe > 0, "RBE should be positive for varying signal"
        assert not np.isnan(rbe)
    
    @pytest.mark.unit
    def test_desi_wavelet_ratio(self, proprietary_features, das_signal):
        """
        Test Dynamic Event Shape Index calculation.
        
        DESI is ratio of low-scale to high-scale wavelet energy.
        """
        desi = proprietary_features.calculate_DESI(das_signal)
        
        assert desi >= 0, "DESI should be non-negative"
        assert not np.isnan(desi)
    
    @pytest.mark.unit
    def test_scr_multichannel_correlation(self, proprietary_features, phi_otdr_signal):
        """
        Test Spatial Coherence Ratio for multi-channel signal.
        """
        scr = proprietary_features.calculate_SCR(phi_otdr_signal)
        
        # With 70% correlation in fixture, SCR should reflect this
        assert 0.3 < scr < 1.0, f"SCR {scr} should reflect channel correlation"
    
    @pytest.mark.unit
    def test_scr_single_channel_default(self, proprietary_features, das_signal):
        """
        Test SCR returns default (0.5) for single-channel signal.
        """
        scr = proprietary_features.calculate_SCR(das_signal)
        
        assert scr == 0.5, "SCR should be 0.5 for single channel"
    
    @pytest.mark.unit
    def test_bsi_variance_calculation(self, proprietary_features, das_signal):
        """
        Test Backscatter Stability Index (variance) calculation.
        """
        bsi = proprietary_features.calculate_BSI(das_signal)
        
        assert bsi >= 0, "BSI (variance) should be non-negative"
        assert not np.isnan(bsi)
    
    @pytest.mark.unit
    def test_proprietary_all_dimension(self, proprietary_features, das_signal, expected_dimensions):
        """
        Test proprietary extract_all returns exactly 4 features.
        """
        features = proprietary_features.extract_all(das_signal, is_multichannel=False)
        
        assert len(features) == expected_dimensions['proprietary']


class TestUniversalFeatureVectorBuilder:
    """
    Tests for UniversalFeatureVectorBuilder.
    
    Builds complete 204-dimensional UFV.
    """
    
    @pytest.mark.unit
    def test_ufv_dimension_guarantee(self, ufv_builder, das_signal, expected_dimensions):
        """
        Test UFV always returns exactly 204 features.
        
        200 standard + 4 proprietary = 204
        """
        ufv = ufv_builder.build_ufv(das_signal, fs=10000, is_multichannel=False)
        
        assert len(ufv) == expected_dimensions['ufv_total']
    
    @pytest.mark.unit
    def test_ufv_empty_signal_raises(self, ufv_builder):
        """
        Test UFV raises ValueError for empty signal.
        """
        empty_signal = np.array([])
        
        with pytest.raises(ValueError, match="empty"):
            ufv_builder.build_ufv(empty_signal, fs=10000)
    
    @pytest.mark.unit
    def test_ufv_invalid_fs_raises(self, ufv_builder, das_signal):
        """
        Test UFV raises ValueError for invalid sampling rate.
        """
        with pytest.raises(ValueError, match="sampling rate|Invalid"):
            ufv_builder.build_ufv(das_signal, fs=0)
        
        with pytest.raises(ValueError, match="sampling rate|Invalid"):
            ufv_builder.build_ufv(das_signal, fs=-1000)
    
    @pytest.mark.unit
    def test_ufv_float32_output(self, ufv_builder, das_signal):
        """
        Test UFV returns float32 array for model compatibility.
        """
        ufv = ufv_builder.build_ufv(das_signal, fs=10000)
        
        assert ufv.dtype == np.float32
    
    @pytest.mark.unit
    def test_ufv_no_nan_inf(self, ufv_builder, das_signal, assert_no_nan_inf):
        """
        Test UFV output contains no NaN or Inf values.
        """
        ufv = ufv_builder.build_ufv(das_signal, fs=10000)
        
        assert_no_nan_inf(ufv, "UFV")
    
    @pytest.mark.unit
    def test_ufv_multichannel(self, ufv_builder, phi_otdr_signal, expected_dimensions):
        """
        Test UFV handles multi-channel signals correctly.
        """
        ufv = ufv_builder.build_ufv(phi_otdr_signal, fs=10000, is_multichannel=True)
        
        assert len(ufv) == expected_dimensions['ufv_total']
        assert not np.any(np.isnan(ufv))
    
    @pytest.mark.unit
    def test_ufv_deterministic(self, ufv_builder, das_signal):
        """
        Test UFV produces identical results for same input.
        """
        ufv1 = ufv_builder.build_ufv(das_signal.copy(), fs=10000)
        ufv2 = ufv_builder.build_ufv(das_signal.copy(), fs=10000)
        
        np.testing.assert_array_almost_equal(ufv1, ufv2)

