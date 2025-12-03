"""
Universal Fiber Sensor Model

A deep learning model for analyzing fiber optic sensor signals from DAS, 
Phi-OTDR, and OTDR sensors. Performs event classification, risk assessment, 
damage detection, and sensor type identification.
"""

__version__ = '1.0.0'
__author__ = 'Universal Fiber Sensor Model Team'

from src.feature_extraction import (
    MultiDomainFeatureExtractor,
    ProprietaryFeatures,
    UniversalFeatureVectorBuilder
)
from src.model_architecture import (
    FusionLayer,
    MultiHeadClassifier,
    UniversalFiberSensorModel
)
from src.signal_preprocessing import (
    UniversalSignalPreprocessor,
    AdaptiveFeatureAggregator
)
from src.inference import FiberSensorInference

__all__ = [
    'MultiDomainFeatureExtractor',
    'ProprietaryFeatures',
    'UniversalFeatureVectorBuilder',
    'FusionLayer',
    'MultiHeadClassifier',
    'UniversalFiberSensorModel',
    'UniversalSignalPreprocessor',
    'AdaptiveFeatureAggregator',
    'FiberSensorInference',
]
