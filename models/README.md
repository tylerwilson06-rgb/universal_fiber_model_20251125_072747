# Model Versions

This folder contains two versions of the trained model for comparison purposes.

## 🔵 trained_model_original.pth (RECOMMENDED)

**Status**: ✅ **Currently in use by the project**

**Details**:
- Size: ~1.68 MB
- Normalization: Per-sample (z-score across all 204 features)
- Trained on: DAS (6,456), Phi-OTDR (15,418), OTDR (180) samples
- Performance: 80.57% / 94.71% / 100% accuracy across datasets
- Compatible with: `src/inference.py` (lines 146-149)

**How it normalizes**:
```python
ufv = (ufv - np.mean(ufv)) / (np.std(ufv) + 1e-8)
```
Calculates one mean and one std for the entire 204-feature vector.

---

## 🟡 trained_model_with_stats.pth (EXPERIMENTAL)

**Status**: ⚠️ **For comparison only - not yet validated**

**Details**:
- Size: ~1.70 MB
- Normalization: Per-feature (204 separate mean/std values)
- Statistics calculated from: 6,456 DAS samples using Welford's algorithm
- Stored stats: `normalization_stats['mean']` and `normalization_stats['std']`
- Compatible with: `src/inference.py` (lines 137-144)

**How it would normalize**:
```python
ufv = (ufv - mean_vector) / (std_vector + 1e-8)
```
Applies 204 different means/stds (one per feature dimension).

**⚠️ Known Issues**:
- Wavelet features have extreme values (mean_max = 1.07e+09)
- Normalization method differs from original training
- Not yet tested for accuracy/performance

---

## 📊 Key Differences

| Aspect | Original | With Stats |
|--------|----------|------------|
| **Normalization** | Per-sample (global) | Per-feature (dimension-wise) |
| **Mean/Std** | 1 mean, 1 std | 204 means, 204 stds |
| **Feature Ranges** | All scaled together | Each feature scaled independently |
| **Compatibility** | ✅ Matches training | ❓ Unknown |
| **Validated** | ✅ Yes (80-100% accuracy) | ❌ No |

---

## 🎯 Which Should You Use?

**For production/deployment**: Use `trained_model_original.pth`

**For experimentation**: You can test `trained_model_with_stats.pth` if you:
1. Want to compare per-feature vs per-sample normalization
2. Have validation data to test accuracy
3. Are willing to accept potential performance degradation

---

## 🗑️ How to Remove the Experimental Model

If you decide you don't need the comparison, simply delete:

```powershell
# Windows PowerShell
Remove-Item models\trained_model_with_stats.pth

# Or manually delete the file in File Explorer
```

To restore the original filename:

```powershell
Rename-Item models\trained_model_original.pth -NewName trained_model.pth
```

---

## 📖 Documentation

For full details on the normalization experiment, see:
- `COLAB_EXPERIMENT_SUMMARY.md` (root folder)
- `colab_backup/README.md` (archived data)
