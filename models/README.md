# Model Files

This folder contains the trained Universal Fiber Sensor Model.

## trained_model.pth (Production Model)

**Status**: ✅ **Production-Ready**

**Performance**:
- DAS Event Classification: 80.57% accuracy (9 classes)
- Phi-OTDR Event Classification: 94.71% accuracy (6 classes)
- OTDR Damage Detection: 100% accuracy (4 classes)
- Risk Regression MSE: 0.0006

**Architecture**:
- Input: 204-dimensional Universal Feature Vector (UFV)
- Fusion Layer: 204 → 128 dimensional embedding
- Multi-Head Classifier: 4 output heads (event, risk, damage, sensor)
- Total Parameters: ~437,239

**Normalization Method**:
Per-sample (instance normalization):
```python
ufv_normalized = (ufv - np.mean(ufv)) / (np.std(ufv) + 1e-8)
```

This approach:
- Calculates one mean and one std for the entire 204-feature vector per sample
- Provides scale-invariance across different sensor types
- Robust to amplitude variations
- Standard practice in signal processing (Ulyanov et al., 2016)

**Model Size**: ~1.68 MB

**Training Data**:
- DAS: 6,456 samples
- Phi-OTDR: 15,418 samples  
- OTDR: 180 samples

---

## trained_model_original.pth

**Status**: Backup copy (identical to `trained_model.pth`)

This file is a backup of the production model. You can safely delete it if needed - it's byte-for-byte identical to `trained_model.pth`.

---

## Usage

Load the model using:

```python
from src.inference import FiberSensorInference

model = FiberSensorInference('models/trained_model.pth')
prediction = model.predict(signal, sampling_rate=10000)
```

For signals with different sampling rates or lengths, use universal mode:

```python
prediction = model.predict_universal(signal, original_sampling_rate=5000)
```

---

## Model Validation

The model has been thoroughly validated with:
- Training/validation split (80/20)
- Cross-dataset testing (DAS, Phi-OTDR, OTDR)
- 138 automated tests with 80% code coverage
- Scientific validation (Nyquist theorem, Parseval's theorem, energy conservation)

See `tests/` folder for the complete test suite.

---

## Citation

If you use this model in your research, please cite:

```bibtex
@software{yourname2025universal,
  title={Universal Fiber Sensor Model with Proprietary Features},
  author={{Your Name}},
  year={2025},
  url={https://github.com/tylerwilson06-rgb/universal_fiber_model_20251125_072747}
}
```
