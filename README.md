# Universal Fiber Sensor Model

**Publication-Ready Implementation**

A universal deep learning model for analyzing fiber optic sensor signals from Distributed Acoustic Sensing (DAS), Phase-sensitive Optical Time Domain Reflectometry (Phi-OTDR), and Optical Time Domain Reflectometry (OTDR) sensors. The model performs simultaneous event classification, risk assessment, damage detection, and sensor type identification.

## Performance

| Dataset   | Task                | Accuracy | Classes |
|-----------|---------------------|----------|---------|
| DAS       | Event Classification| 80.57%   | 9       |
| Phi-OTDR  | Event Classification| 94.71%   | 6       |
| OTDR      | Damage Detection    | 100.00%  | 4       |

**Risk Regression MSE:** 0.0006

## Key Features

- **Universal Compatibility**: Works with DAS, Phi-OTDR, and OTDR sensors
- **Universal Preprocessing Mode**: Automatically handles ANY sampling rate and signal length - truly universal
- **Multi-Task Learning**: Simultaneous event classification, risk assessment, damage detection, and sensor identification
- **Robust Feature Extraction**: 204-dimensional Universal Feature Vector (UFV) with proprietary fiber-aware features
- **Production-Ready**: Comprehensive error handling, input validation, and batch inference support
- **GPU Support**: Automatic GPU detection and utilization when available

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full list

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Standard Mode (Requires Proper Input Format)

For signals that already match the training format (10kHz sampling rate, 512+ samples):

```python
from src.inference import FiberSensorInference
import numpy as np

# Initialize model (auto-detects GPU if available)
model = FiberSensorInference('models/trained_model.pth')

# Prepare your sensor signal (numpy array)
raw_signal = np.random.randn(10000)  # 1 second at 10kHz

# Make prediction
prediction = model.predict(raw_signal, sampling_rate=10000)

print(f"Event: {prediction['event_type']}")
print(f"Confidence: {prediction['event_confidence']:.2%}")
print(f"Risk Score: {prediction['risk_score']:.2%}")
print(f"Damage: {prediction['damage_type']}")
print(f"Sensor Type: {prediction['sensor_type']}")
```

### Universal Mode (Handles Any Input)

**This is the key feature that makes the model truly universal.** Works with ANY sampling rate and ANY signal length:

```python
from src.inference import FiberSensorInference
import numpy as np

# Initialize model
model = FiberSensorInference('models/trained_model.pth')

# Example 1: Different sampling rate (5kHz instead of 10kHz)
signal_5khz = np.random.randn(5000)  # 5kHz, 1 second
prediction = model.predict_universal(signal_5khz, original_sampling_rate=5000)

# Example 2: Very high sampling rate (50kHz)
signal_50khz = np.random.randn(50000)  # 50kHz, 1 second
prediction = model.predict_universal(signal_50khz, original_sampling_rate=50000)

# Example 3: Very short signal (automatically padded)
signal_short = np.random.randn(100)  # Very short
prediction = model.predict_universal(signal_short, original_sampling_rate=1000)

# Example 4: Very long signal (automatically windowed and averaged)
signal_long = np.random.randn(100000)  # Very long
prediction = model.predict_universal(signal_long, original_sampling_rate=10000)

# All of these work automatically!
```

### Multi-Channel Signals

```python
# For multi-channel signals (e.g., Phi-OTDR)
phi_signal = np.random.randn(10000, 12)  # 12 channels
prediction = model.predict(phi_signal, sampling_rate=10000, is_multichannel=True)
```

### Batch Inference

```python
# Process multiple signals efficiently (standard mode)
batch_signals = [np.random.randn(10000) for _ in range(10)]
predictions = model.predict_batch(batch_signals, sampling_rate=10000)

# Universal batch mode (handles different rates/lengths)
signals = [
    np.random.randn(5000),   # 5kHz
    np.random.randn(20000),  # 20kHz
    np.random.randn(10000),  # 10kHz
]
rates = [5000, 20000, 10000]
predictions = model.predict_batch_universal(signals, original_sampling_rates=rates)
```

## Project Structure

```
universal_fiber_model/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── feature_extraction.py    # UFV builder with robust error handling
│   ├── model_architecture.py    # Model definition (PyTorch)
│   ├── signal_preprocessing.py  # Universal preprocessing (NEW)
│   └── inference.py             # Production-ready inference interface
├── models/
│   └── trained_model.pth        # Pre-trained model weights
├── training/
│   └── Optics.ipynb             # Training notebook (Google Colab)
├── examples/
│   └── usage_example.py         # Comprehensive usage examples
├── docs/
│   └── (documentation)
├── requirements.txt
└── README.md
```

## Model Architecture

### Input: Universal Feature Vector (UFV)

The model expects a 204-dimensional feature vector extracted from raw sensor signals:

- **Standard Features (200)**:
  - MFCC features: 120 (40 coefficients + deltas + delta-deltas)
  - Wavelet features: 64 (wavelet packet decomposition)
  - Spectral features: 6 (centroid, bandwidth, rolloff, etc.)
  - Temporal features: 6 (RMS, peak, zero-crossing rate, etc.)
  - Spatial features: 4 (for multi-channel data)

- **Proprietary Features (4)**:
1. **RBE** - Rayleigh Backscatter Entropy: Measures signal disorder
2. **DESI** - Dynamic Event Shape Index: Characterizes transient shapes
3. **SCR** - Spatial Coherence Ratio: Multi-channel correlation
4. **BSI** - Backscatter Stability Index: Signal variance

### Neural Network Architecture

- **Fusion Layer**: 204 → 256 → 256 → Attention → 128
  - Two fully connected layers with layer normalization
  - Multi-head attention mechanism for feature fusion
  - Dropout regularization (0.3)

- **Multi-Head Classifier**:
  - **Event Head**: 128 → 64 → 15 classes
  - **Risk Head**: 128 → 32 → 1 (sigmoid, 0-1)
  - **Damage Head**: 128 → 32 → 4 classes
  - **Sensor Type Head**: 128 → 32 → 3 classes

- **Total Parameters**: 437,239 (~1.75 MB)

## Universal Preprocessing System

The universal preprocessing system is a key innovation that enables the model to work with signals from any sensor configuration:

### Features

1. **Automatic Resampling**: Intelligently resamples signals to the target rate (10kHz) using high-quality methods
   - Uses librosa's kaiser-windowed sinc interpolation (best quality)
   - Falls back to scipy FFT-based resampling if librosa unavailable
   - Includes anti-aliasing to preserve signal characteristics

2. **Adaptive Length Handling**:
   - **Short signals**: Automatically pads with zeros
   - **Long signals**: Uses multi-window averaging for robust feature extraction
   - **Optimal length**: Uses signal as-is

3. **Quality Preservation**: 
   - Validates signal quality
   - Issues warnings for edge cases
   - Maintains signal characteristics through advanced processing

### Technical Details

The preprocessing system implements signal processing best practices:

- **Resampling**: FFT-based resampling with anti-aliasing filters
- **Multi-Window Averaging**: For long signals, extracts features from multiple overlapping windows and aggregates them
- **Quality Monitoring**: Tracks preprocessing operations and provides metadata

### Usage

```python
# Universal mode automatically handles everything
prediction = model.predict_universal(
    raw_signal, 
    original_sampling_rate=your_rate,  # Any rate!
    is_multichannel=False
)

# Get preprocessing information
prediction, info = model.predict_universal(
    raw_signal,
    original_sampling_rate=your_rate,
    return_preprocessing_info=True
)
print(f"Resample ratio: {info['resample_ratio']}")
print(f"Length ratio: {info['length_ratio']}")
print(f"Warnings: {info['warnings']}")
```

## Proprietary Features

The model includes four proprietary features designed specifically for fiber optic sensing:

1. **RBE (Rayleigh Backscatter Entropy)**: Quantifies signal disorder using histogram entropy
2. **DESI (Dynamic Event Shape Index)**: Characterizes transient event shapes via wavelet energy ratios
3. **SCR (Spatial Coherence Ratio)**: Measures spatial correlation across multiple channels
4. **BSI (Backscatter Stability Index)**: Quantifies signal variance and stability

## Datasets

The model was trained on three datasets:

- **DAS**: 6,456 samples, 9 event classes
- **Phi-OTDR**: 15,418 samples, 6 event classes  
- **OTDR**: 180 samples, 4 damage classes

**Total Training Samples**: 22,054

## Training

The model was trained using Google Colab. The complete training notebook is available in `training/Optics.ipynb`. This notebook includes:

- Model architecture definition
- Data loading and preprocessing pipeline
- Training hyperparameters and configuration
- Training loop implementation
- Model evaluation and accuracy calculations
- Model checkpoint saving

To reproduce the training:
1. Open `training/Optics.ipynb` in Google Colab
2. Upload your training datasets
3. Run all cells to train the model
4. The trained model will be saved as `trained_model.pth`

## API Reference

### `FiberSensorInference`

Main inference class for making predictions.

#### Methods

- `__init__(model_path, device=None)`: Initialize model
  - `model_path`: Path to trained model checkpoint
  - `device`: Device to use ('cpu', 'cuda', or None for auto-detection)

- `predict(raw_signal, sampling_rate=10000, is_multichannel=False)`: Standard prediction
  - Requires signals at 10kHz with 512+ samples
  - Returns: Dictionary with event_type, event_confidence, risk_score, damage_type, damage_confidence, sensor_type, sensor_confidence

- `predict_universal(raw_signal, original_sampling_rate=None, is_multichannel=False, auto_preprocess=True, return_preprocessing_info=False)`: Universal prediction
  - **Handles ANY sampling rate and signal length automatically**
  - Automatically resamples and normalizes length
  - Returns: Dictionary with predictions (and optionally preprocessing info)

- `predict_batch(raw_signals, sampling_rate=10000, is_multichannel=False)`: Batch predictions (standard mode)
  - Returns: List of prediction dictionaries

- `predict_batch_universal(raw_signals, original_sampling_rates=None, is_multichannel=False, auto_preprocess=True)`: Universal batch predictions
  - Handles signals with different sampling rates/lengths
  - Returns: List of prediction dictionaries

## Error Handling

The implementation includes comprehensive error handling:

- **Input Validation**: Checks for empty signals, NaN/Inf values, invalid sampling rates
- **Edge Case Handling**: Handles very short signals, single-channel fallbacks
- **Robust Feature Extraction**: Graceful degradation with warnings for edge cases
- **Model Loading**: Handles different checkpoint formats and missing keys

## Important Notes

### Standard Mode

1. **Sampling Rate**: Default is 10kHz. Adjust according to your sensor specifications.
2. **Signal Length**: Minimum recommended length is 512 samples. Shorter signals will be padded with zeros.
3. **Multi-Channel**: Automatically detected for 2D arrays with shape (samples, channels).
4. **GPU Usage**: Automatically uses GPU if available. Set `device='cpu'` to force CPU usage.
5. **Normalization**: The model includes embedded normalization statistics from training. The inference engine automatically uses these for optimal accuracy.

### Universal Mode

1. **Sampling Rate**: Can be ANY rate (100 Hz to 1 MHz). Automatically resampled to 10kHz.
2. **Signal Length**: Can be ANY length. Automatically handled (padding, truncation, or windowing).
3. **Best Results**: Provide actual sampling rate for optimal accuracy.
4. **Performance**: Slightly slower due to preprocessing, but enables true universality.

## Citation

If you use this model in your research, please cite:

```bibtex
@article{yourname2025universal,
  title={Universal Fiber Sensor Model with Proprietary Features},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## References

### Signal Processing and Feature Extraction

1. Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357-366. (MFCC features)

2. Mallat, S. (2008). *A Wavelet Tour of Signal Processing: The Sparse Way*. Academic Press. (Wavelet transforms)

3. McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis based on a sinusoidal representation. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 34(4), 744-754. (Spectral features)

4. Smith, J. O. (2011). *Spectral Audio Signal Processing*. W3K Publishing. (Spectral analysis)

### Fiber Optic Sensing

5. Masoudi, A., & Newson, T. P. (2016). Contributed Review: Distributed optical fibre dynamic strain sensing. *Review of Scientific Instruments*, 87(1), 011501. (DAS principles)

6. Lu, Y., et al. (2010). Distributed vibration sensor based on coherent detection of phase-OTDR. *Journal of Lightwave Technology*, 28(22), 3243-3249. (Phi-OTDR)

7. Healey, P. (1984). Fading in heterodyne OTDR. *Electronics Letters*, 20(1), 30-32. (OTDR fundamentals)

### Deep Learning and Multi-Task Learning

8. Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75. (Multi-task learning)

9. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32. (PyTorch)

### Signal Resampling and Preprocessing

10. Smith, J. O. (2002). *Digital Audio Resampling Home Page*. https://ccrma.stanford.edu/~jos/resample/ (Resampling theory)

11. McLeod, A. I., & Wyvill, G. (1985). A fast algorithm for resampling discrete signals. *Computer Graphics*, 19(3), 133-139. (FFT-based resampling)

12. Proakis, J. G., & Manolakis, D. G. (2006). *Digital Signal Processing: Principles, Algorithms, and Applications*. Prentice Hall. (Digital signal processing fundamentals)

### Software Libraries

13. McFee, B., et al. (2015). librosa: Audio and music analysis in Python. *Proceedings of the 14th Python in Science Conference*, 18-25.

14. Lee, G. R., et al. (2019). PyWavelets: A Python package for wavelet analysis. *Journal of Open Source Software*, 4(36), 1237.

15. Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261-272.

## Contact

- Author: Tyler Wilson
- Email: justtywilson@gmail.com
- Author: Samuel Wilson
- Email: justsamwilson@gmail.com

## License

N/A

## Acknowledgments

- Trained on Google Colab with T4 GPU
- Built with PyTorch, librosa, and PyWavelets
- Signal preprocessing based on established signal processing best practices

## Troubleshooting

### Common Issues

**Model file not found:**
- Ensure `trained_model.pth` is in the `models/` directory
- Check the file path is correct

**Dimension mismatch errors:**
- Use `predict_universal()` for signals with non-standard lengths/rates
- Ensure signals have sufficient length (minimum 512 samples recommended for standard mode)

**GPU out of memory:**
- Use CPU mode: `FiberSensorInference(model_path, device='cpu')`
- Reduce batch size for batch inference

**Low accuracy on new data:**
- For standard mode: Verify sampling rate matches training data (default: 10kHz)
- For universal mode: Ensure you provide the correct original sampling rate
- Check that signal preprocessing matches training pipeline

**Resampling warnings:**
- Very high or very low sampling rates may produce warnings
- Results are still valid, but accuracy may be slightly reduced
- For best results, use signals sampled between 1kHz and 100kHz
