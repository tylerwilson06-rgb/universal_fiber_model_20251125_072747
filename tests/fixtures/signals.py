"""
Signal Generator Functions for Testing

Provides synthetic and controlled test signals for validating the
Universal Fiber Sensor Model preprocessing and feature extraction.

Based on:
- Jousset et al. (2018) - DAS synthetic signal generation
- IEEE Std 1057-2017 - Waveform digitizing standards
- Hartog (2017) - Distributed optical fibre sensors
"""

import numpy as np
from typing import Dict, Tuple, Optional


def generate_synthetic_signal(
    freq: float = 1000.0,
    duration: float = 1.0,
    fs: int = 10000,
    noise_level: float = 0.1,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate a synthetic sine wave signal with optional noise.
    
    Used for testing frequency preservation in resampling and
    spectral feature extraction accuracy.
    
    Args:
        freq: Frequency of the sine wave in Hz
        duration: Duration in seconds
        fs: Sampling rate in Hz
        noise_level: Standard deviation of Gaussian noise (0 = pure sine)
        seed: Random seed for reproducibility
    
    Returns:
        1D numpy array of the signal
    
    Reference:
        IEEE Std 1057-2017, Section 4.2 - Test signal generation
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = int(fs * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    
    if noise_level > 0:
        noise = np.random.randn(num_samples) * noise_level
        signal = signal + noise
    
    return signal.astype(np.float64)


def generate_multi_frequency_signal(
    frequencies: Tuple[float, ...] = (500.0, 1000.0, 2000.0),
    amplitudes: Optional[Tuple[float, ...]] = None,
    duration: float = 1.0,
    fs: int = 10000,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate a signal with multiple frequency components.
    
    Used for testing anti-aliasing filter effectiveness and
    frequency content preservation during resampling.
    
    Args:
        frequencies: Tuple of frequencies in Hz
        amplitudes: Tuple of amplitudes (default: all 1.0)
        duration: Duration in seconds
        fs: Sampling rate in Hz
        seed: Random seed for reproducibility
    
    Returns:
        1D numpy array of the signal
    
    Reference:
        Shannon-Nyquist sampling theorem validation
    """
    if seed is not None:
        np.random.seed(seed)
    
    if amplitudes is None:
        amplitudes = tuple(1.0 for _ in frequencies)
    
    num_samples = int(fs * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    signal = np.zeros(num_samples)
    for freq, amp in zip(frequencies, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)
    
    return signal.astype(np.float64)


def generate_das_like_signal(
    length: int = 10000,
    fs: int = 10000,
    event_amplitude: float = 2.0,
    background_noise: float = 0.3,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate a DAS (Distributed Acoustic Sensing) like signal.
    
    Simulates fiber optic acoustic sensing with background noise
    and potential event signatures.
    
    Args:
        length: Number of samples
        fs: Sampling rate in Hz
        event_amplitude: Amplitude of simulated events
        background_noise: Standard deviation of background noise
        seed: Random seed for reproducibility
    
    Returns:
        1D numpy array simulating DAS signal
    
    Reference:
        Jousset et al. (2018) - DAS for seismic applications
        Hartog (2017) - Distributed optical fibre sensors
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Background noise (simulates Rayleigh backscatter variations)
    signal = np.random.randn(length) * background_noise
    
    # Add some low-frequency drift (common in DAS)
    t = np.linspace(0, length / fs, length)
    drift = 0.1 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz drift
    signal += drift
    
    # Add simulated event (burst of energy)
    event_start = length // 3
    event_duration = length // 10
    event_signal = event_amplitude * np.sin(2 * np.pi * 50 * t[event_start:event_start + event_duration])
    event_signal *= np.hanning(event_duration)  # Window the event
    signal[event_start:event_start + event_duration] += event_signal
    
    return signal.astype(np.float64)


def generate_multichannel_signal(
    length: int = 10000,
    channels: int = 12,
    fs: int = 10000,
    correlation: float = 0.7,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate a multi-channel signal (like Phi-OTDR).
    
    Simulates spatially distributed fiber sensing with
    inter-channel correlation.
    
    Args:
        length: Number of samples per channel
        channels: Number of channels
        fs: Sampling rate in Hz
        correlation: Inter-channel correlation coefficient (0-1)
        seed: Random seed for reproducibility
    
    Returns:
        2D numpy array of shape (length, channels)
    
    Reference:
        Phi-OTDR multi-channel sensing characteristics
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate correlated channels
    # Start with common component
    common_signal = np.random.randn(length)
    
    signals = []
    for ch in range(channels):
        # Mix common signal with channel-specific noise
        independent = np.random.randn(length)
        channel_signal = (
            np.sqrt(correlation) * common_signal + 
            np.sqrt(1 - correlation) * independent
        )
        signals.append(channel_signal)
    
    return np.column_stack(signals).astype(np.float64)


def generate_edge_case_signals(seed: Optional[int] = 42) -> Dict[str, np.ndarray]:
    """
    Generate a dictionary of edge case signals for robustness testing.
    
    Based on Google ML Testing Framework recommendations for
    input validation and error handling.
    
    Args:
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping edge case names to signal arrays
    
    Reference:
        Breck et al. (2020) - Testing and Debugging in ML
    """
    if seed is not None:
        np.random.seed(seed)
    
    edge_cases = {
        # Empty and minimal signals
        'empty': np.array([], dtype=np.float64),
        'single_sample': np.array([1.0], dtype=np.float64),
        'two_samples': np.array([1.0, -1.0], dtype=np.float64),
        'ten_samples': np.random.randn(10).astype(np.float64),
        'hundred_samples': np.random.randn(100).astype(np.float64),
        
        # Extreme lengths
        'very_short': np.random.randn(5).astype(np.float64),
        'very_long': np.random.randn(100000).astype(np.float64),
        
        # Special values
        'all_zeros': np.zeros(1000, dtype=np.float64),
        'all_ones': np.ones(1000, dtype=np.float64),
        'constant': np.full(1000, 3.14159, dtype=np.float64),
        
        # NaN and Inf (for error handling tests)
        'contains_nan': np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float64),
        'contains_inf': np.array([1.0, 2.0, np.inf, 4.0, 5.0], dtype=np.float64),
        'contains_neg_inf': np.array([1.0, 2.0, -np.inf, 4.0, 5.0], dtype=np.float64),
        'all_nan': np.full(100, np.nan, dtype=np.float64),
        
        # Extreme values
        'very_large': np.random.randn(1000).astype(np.float64) * 1e10,
        'very_small': np.random.randn(1000).astype(np.float64) * 1e-10,
        
        # Special patterns
        'impulse': np.concatenate([np.zeros(500), np.array([100.0]), np.zeros(499)]),
        'step': np.concatenate([np.zeros(500), np.ones(500)]),
        'ramp': np.linspace(-1, 1, 1000, dtype=np.float64),
        
        # Standard test signals
        'standard_10k': np.random.randn(10000).astype(np.float64),
        'standard_5k': np.random.randn(5000).astype(np.float64),
        'standard_20k': np.random.randn(20000).astype(np.float64),
    }
    
    return edge_cases


def generate_known_frequency_signal(
    freq: float,
    fs: int,
    duration: float = 1.0
) -> Tuple[np.ndarray, float]:
    """
    Generate a pure sine wave with known frequency for validation.
    
    Returns both the signal and the expected peak frequency for
    spectral feature validation.
    
    Args:
        freq: Frequency in Hz
        fs: Sampling rate in Hz
        duration: Duration in seconds
    
    Returns:
        Tuple of (signal array, expected peak frequency)
    
    Reference:
        Spectral analysis validation tests
    """
    num_samples = int(fs * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    signal = np.sin(2 * np.pi * freq * t).astype(np.float64)
    
    return signal, freq


def generate_white_noise(
    length: int = 10000,
    std: float = 1.0,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate white Gaussian noise with known statistical properties.
    
    Used for testing temporal and spectral feature extraction
    on signals with known statistics.
    
    Args:
        length: Number of samples
        std: Standard deviation
        seed: Random seed for reproducibility
    
    Returns:
        1D numpy array of white noise
    """
    if seed is not None:
        np.random.seed(seed)
    
    return (np.random.randn(length) * std).astype(np.float64)

