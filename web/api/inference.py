"""
API wrapper for model inference
Handles file parsing and model predictions for the web application
"""

import numpy as np
import sys
import os
import io
from pathlib import Path

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference import FiberSensorInference

class ModelAPI:
    """API wrapper for model inference"""
    
    def __init__(self, model_path=None):
        """
        Initialize model API
        
        Args:
            model_path: Path to model file (default: models/trained_model.pth)
        """
        if model_path is None:
            # Try relative path first, then absolute
            base_dir = Path(__file__).parent.parent.parent
            model_path = base_dir / 'models' / 'trained_model.pth'
            
            if not model_path.exists():
                # Try current directory
                model_path = Path('models/trained_model.pth')
        
        self.model_path = str(model_path)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the model (lazy loading)"""
        try:
            self.model = FiberSensorInference(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def parse_file(self, file_content, filename):
        """
        Parse uploaded file into numpy array
        
        Args:
            file_content: File content (bytes or string)
            filename: Original filename
        
        Returns:
            numpy array
        """
        try:
            # Determine file type
            ext = Path(filename).suffix.lower()
            
            if ext == '.npy':
                # NumPy array file
                signal = np.load(io.BytesIO(file_content))
            
            elif ext in ['.csv', '.txt']:
                # Text file - try to parse
                if isinstance(file_content, bytes):
                    content = file_content.decode('utf-8')
                else:
                    content = file_content
                
                # Try CSV first (comma-separated)
                try:
                    signal = np.loadtxt(io.StringIO(content), delimiter=',')
                except:
                    # Try space-separated
                    try:
                        signal = np.loadtxt(io.StringIO(content))
                    except:
                        # Try one value per line
                        lines = content.strip().split('\n')
                        signal = np.array([float(line.strip()) for line in lines if line.strip()])
            
            else:
                # Try to parse as text
                if isinstance(file_content, bytes):
                    content = file_content.decode('utf-8')
                else:
                    content = file_content
                
                # Try various delimiters
                for delimiter in [',', ' ', '\t', ';']:
                    try:
                        signal = np.loadtxt(io.StringIO(content), delimiter=delimiter)
                        break
                    except:
                        continue
                else:
                    # Last resort: one value per line
                    lines = content.strip().split('\n')
                    signal = np.array([float(line.strip()) for line in lines if line.strip()])
            
            # Ensure it's a numpy array
            signal = np.asarray(signal, dtype=np.float64)
            
            # Flatten if 2D with single column
            if signal.ndim == 2 and signal.shape[1] == 1:
                signal = signal.flatten()
            
            return signal
            
        except Exception as e:
            raise ValueError(f"Failed to parse file '{filename}': {e}")
    
    def predict_standard(self, signal, sampling_rate=10000, is_multichannel=False):
        """
        Standard mode prediction
        
        Args:
            signal: numpy array
            sampling_rate: sampling rate in Hz
            is_multichannel: whether signal is multi-channel
        
        Returns:
            dict with predictions
        """
        try:
            prediction = self.model.predict(signal, sampling_rate, is_multichannel)
            return {
                'success': True,
                'prediction': prediction,
                'mode': 'standard'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'mode': 'standard'
            }
    
    def predict_universal(self, signal, original_sampling_rate=None, is_multichannel=False):
        """
        Universal mode prediction
        
        Args:
            signal: numpy array
            original_sampling_rate: original sampling rate in Hz
            is_multichannel: whether signal is multi-channel
        
        Returns:
            dict with predictions and preprocessing info
        """
        try:
            prediction, preprocessing_info = self.model.predict_universal(
                signal,
                original_sampling_rate=original_sampling_rate,
                is_multichannel=is_multichannel,
                return_preprocessing_info=True
            )
            
            return {
                'success': True,
                'prediction': prediction,
                'preprocessing_info': preprocessing_info,
                'mode': 'universal'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'mode': 'universal'
            }

