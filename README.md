# Universal Fiber Sensor Model

**Publication-Ready Implementation**

A universal deep learning model for analyzing fiber optic sensor signals from Distributed Acoustic Sensing (DAS), Phase-sensitive Optical Time Domain Reflectometry (Phi-OTDR), and Optical Time Domain Reflectometry (OTDR) sensors. The model performs simultaneous event classification, risk assessment, damage detection, and sensor type identification.

## 🎯 Performance

| Dataset   | Task                | Accuracy | Classes |
|-----------|---------------------|----------|---------|
| DAS       | Event Classification| 80.57%   | 9       |
| Phi-OTDR  | Event Classification| 94.71%   | 6       |
| OTDR      | Damage Detection    | 100.00%  | 4       |

**Risk Regression MSE:** 0.0006

## ✨ Key Features

- **Universal Compatibility**: Works with DAS, Phi-OTDR, and OTDR sensors
- **Multi-Task Learning**: Simultaneous event classification, risk assessment, damage detection, and sensor identification
- **Robust Feature Extraction**: 204-dimensional Universal Feature Vector (UFV) with proprietary fiber-aware features
- **Production-Ready**: Comprehensive error handling, input validation, and batch inference support
- **GPU Support**: Automatic GPU detection and utilization when available

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full list

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```python
from src.inference import FiberSensorInference
import numpy as np

# Initialize model (auto-detects GPU if available)
model = FiberSensorInference('models/trained_model.pth')

# Prepare your sensor signal (numpy array)
raw_signal = np.random.randn(10000)  # Example: 1 second at 10kHz

# Make prediction
prediction = model.predict(raw_signal, sampling_rate=10000)

print(f"Event: {prediction['event_type']}")
print(f"Confidence: {prediction['event_confidence']:.2%}")
print(f"Risk Score: {prediction['risk_score']:.2%}")
print(f"Damage: {prediction['damage_type']}")
print(f"Sensor Type: {prediction['sensor_type']}")
```

### Multi-Channel Signals

```python
# For multi-channel signals (e.g., Phi-OTDR)
phi_signal = np.random.randn(10000, 12)  # 12 channels
prediction = model.predict(phi_signal, sampling_rate=10000, is_multichannel=True)
```

### Batch Inference

```python
# Process multiple signals efficiently
batch_signals = [np.random.randn(10000) for _ in range(10)]
predictions = model.predict_batch(batch_signals, sampling_rate=10000)
```

### Real-Time Monitoring

```python
# Continuous monitoring example
while True:
    signal = read_sensor_data()  # Your data acquisition function
    prediction = model.predict(signal, sampling_rate=10000)
    
    if prediction['risk_score'] > 0.7:
        alert(f"High risk detected: {prediction['event_type']}")
```

## 📁 Project Structure

```
universal_fiber_model/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── feature_extraction.py    # UFV builder with robust error handling
│   ├── model_architecture.py    # Model definition (PyTorch)
│   └── inference.py             # Production-ready inference interface
├── models/
│   └── trained_model.pth        # Pre-trained model weights
├── examples/
│   └── usage_example.py         # Comprehensive usage examples
├── docs/
│   └── (documentation)
├── requirements.txt
└── README.md
```

## 🧠 Model Architecture

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

## 🔬 Proprietary Features

The model includes four proprietary features designed specifically for fiber optic sensing:

1. **RBE (Rayleigh Backscatter Entropy)**: Quantifies signal disorder using histogram entropy
2. **DESI (Dynamic Event Shape Index)**: Characterizes transient event shapes via wavelet energy ratios
3. **SCR (Spatial Coherence Ratio)**: Measures spatial correlation across multiple channels
4. **BSI (Backscatter Stability Index)**: Quantifies signal variance and stability

## 📊 Datasets

The model was trained on three datasets:

- **DAS**: 6,456 samples, 9 event classes
- **Phi-OTDR**: 15,418 samples, 6 event classes  
- **OTDR**: 180 samples, 4 damage classes

**Total Training Samples**: 22,054

## 🛠️ API Reference

### `FiberSensorInference`

Main inference class for making predictions.

#### Methods

- `__init__(model_path, device=None)`: Initialize model
  - `model_path`: Path to trained model checkpoint
  - `device`: Device to use ('cpu', 'cuda', or None for auto-detection)

- `predict(raw_signal, sampling_rate=10000, is_multichannel=False)`: Single prediction
  - Returns: Dictionary with event_type, event_confidence, risk_score, damage_type, damage_confidence, sensor_type, sensor_confidence

- `predict_batch(raw_signals, sampling_rate=10000, is_multichannel=False)`: Batch predictions
  - Returns: List of prediction dictionaries

## 🔧 Error Handling

The implementation includes comprehensive error handling:

- **Input Validation**: Checks for empty signals, NaN/Inf values, invalid sampling rates
- **Edge Case Handling**: Handles very short signals, single-channel fallbacks
- **Robust Feature Extraction**: Graceful degradation with warnings for edge cases
- **Model Loading**: Handles different checkpoint formats and missing keys

## ⚠️ Important Notes

1. **Sampling Rate**: Default is 10kHz. Adjust according to your sensor specifications.
2. **Signal Length**: Minimum recommended length is 512 samples. Shorter signals will be padded with zeros.
3. **Multi-Channel**: Automatically detected for 2D arrays with shape (samples, channels).
4. **GPU Usage**: Automatically uses GPU if available. Set `device='cpu'` to force CPU usage.

## 🎓 Citation

If you use this model in your research, please cite:

```bibtex
@article{yourname2025universal,
  title={Universal Fiber Sensor Model with Proprietary Features},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## 📧 Contact

- Author: [Your Name]
- Email: [Your Email]
- GitHub: [Your GitHub]

## 📄 License

[Your License]

## 🙏 Acknowledgments

- Trained on Google Colab with T4 GPU
- Built with PyTorch, librosa, and PyWavelets

## 🔍 Troubleshooting

### Common Issues

**Model file not found:**
- Ensure `trained_model.pth` is in the `models/` directory
- Check the file path is correct

**Dimension mismatch errors:**
- Ensure signals have sufficient length (minimum 512 samples recommended)
- Check that multi-channel signals have correct shape (samples, channels)

**GPU out of memory:**
- Use CPU mode: `FiberSensorInference(model_path, device='cpu')`
- Reduce batch size for batch inference

**Low accuracy on new data:**
- Verify sampling rate matches training data (default: 10kHz)
- Ensure signal preprocessing matches training pipeline
