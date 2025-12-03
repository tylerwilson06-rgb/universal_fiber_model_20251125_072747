"""
Example Usage Script
Demonstrates how to use the Universal Fiber Sensor Model
"""

import numpy as np
import sys
import os

# Add parent directory to path to import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import FiberSensorInference

def main():
    """Main example function"""
    
    # Initialize model (auto-detects GPU if available)
    model_path = os.path.join('..', 'models', 'trained_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure the trained model is in the models/ directory")
        return
    
    try:
        print("Loading model...")
        model = FiberSensorInference(model_path)
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Example 1: DAS-like signal (single channel)
    print("=" * 60)
    print("Example 1: DAS Signal (Single Channel)")
    print("=" * 60)
    try:
        das_signal = np.random.randn(10000)  # 1 second at 10kHz
        prediction = model.predict(das_signal, sampling_rate=10000)
        
        print(f"  Event Type: {prediction['event_type']}")
        print(f"  Event Confidence: {prediction['event_confidence']:.2%}")
        print(f"  Risk Score: {prediction['risk_score']:.2%}")
        print(f"  Damage Type: {prediction['damage_type']}")
        print(f"  Damage Confidence: {prediction['damage_confidence']:.2%}")
        print(f"  Sensor Type: {prediction['sensor_type']}")
        print(f"  Sensor Confidence: {prediction['sensor_confidence']:.2%}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    # Example 2: Multi-channel signal (Phi-OTDR-like)
    print("=" * 60)
    print("Example 2: Multi-channel Signal (Phi-OTDR-like)")
    print("=" * 60)
    try:
        phi_signal = np.random.randn(10000, 12)  # 12 channels
        prediction = model.predict(phi_signal, sampling_rate=10000, is_multichannel=True)
        
        print(f"  Event Type: {prediction['event_type']}")
        print(f"  Event Confidence: {prediction['event_confidence']:.2%}")
        print(f"  Risk Score: {prediction['risk_score']:.2%}")
        print(f"  Damage Type: {prediction['damage_type']}")
        print(f"  Sensor Type: {prediction['sensor_type']}")
        print(f"  Sensor Confidence: {prediction['sensor_confidence']:.2%}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    # Example 3: Real-time monitoring loop
    print("=" * 60)
    print("Example 3: Real-time Monitoring Loop")
    print("=" * 60)
    try:
        for i in range(5):
            signal = np.random.randn(10000)
            prediction = model.predict(signal, sampling_rate=10000)
            
            if prediction['risk_score'] > 0.7:
                print(f"  ⚠️  HIGH RISK: {prediction['event_type']} "
                      f"(Risk: {prediction['risk_score']:.2%})")
            else:
                print(f"  ✅ Normal: {prediction['event_type']} "
                      f"(Risk: {prediction['risk_score']:.2%})")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    # Example 4: Batch inference
    print("=" * 60)
    print("Example 4: Batch Inference (Efficient)")
    print("=" * 60)
    try:
        batch_signals = [np.random.randn(10000) for _ in range(3)]
        predictions = model.predict_batch(batch_signals, sampling_rate=10000)
        
        for i, pred in enumerate(predictions):
            print(f"  Signal {i+1}: {pred['event_type']} "
                  f"(Confidence: {pred['event_confidence']:.2%}, "
                  f"Risk: {pred['risk_score']:.2%})")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    # Example 5: Universal Mode - Different Sampling Rate
    print("=" * 60)
    print("Example 5: Universal Mode - Different Sampling Rate (5kHz)")
    print("=" * 60)
    try:
        signal_5khz = np.random.randn(5000)  # 5kHz, 1 second
        prediction = model.predict_universal(signal_5khz, original_sampling_rate=5000)
        
        print(f"  Event Type: {prediction['event_type']}")
        print(f"  Event Confidence: {prediction['event_confidence']:.2%}")
        print(f"  Risk Score: {prediction['risk_score']:.2%}")
        print(f"  Note: Signal automatically resampled from 5kHz to 10kHz")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    # Example 6: Universal Mode - Very Short Signal
    print("=" * 60)
    print("Example 6: Universal Mode - Very Short Signal")
    print("=" * 60)
    try:
        short_signal = np.random.randn(100)  # Very short
        prediction, info = model.predict_universal(
            short_signal, 
            original_sampling_rate=1000,
            return_preprocessing_info=True
        )
        
        print(f"  Event Type: {prediction['event_type']}")
        print(f"  Risk Score: {prediction['risk_score']:.2%}")
        print(f"  Preprocessing Info:")
        print(f"    - Original length: {info['original_shape'][0]} samples")
        print(f"    - Processed length: {info['processed_shape'][0]} samples")
        print(f"    - Length ratio: {info['length_ratio']:.2f}")
        if info['warnings']:
            print(f"    - Warnings: {', '.join(info['warnings'])}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    # Example 7: Universal Mode - Very Long Signal
    print("=" * 60)
    print("Example 7: Universal Mode - Very Long Signal")
    print("=" * 60)
    try:
        long_signal = np.random.randn(50000)  # Very long (5 seconds at 10kHz)
        prediction, info = model.predict_universal(
            long_signal,
            original_sampling_rate=10000,
            return_preprocessing_info=True
        )
        
        print(f"  Event Type: {prediction['event_type']}")
        print(f"  Risk Score: {prediction['risk_score']:.2%}")
        print(f"  Preprocessing Info:")
        print(f"    - Original length: {info['original_shape'][0]} samples")
        print(f"    - Processed length: {info['processed_shape'][0]} samples")
        print(f"    - Length ratio: {info['length_ratio']:.2f}")
        if info['warnings']:
            print(f"    - Warnings: {', '.join(info['warnings'])}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    # Example 8: Universal Batch Mode
    print("=" * 60)
    print("Example 8: Universal Batch Mode (Different Rates/Lengths)")
    print("=" * 60)
    try:
        signals = [
            np.random.randn(5000),   # 5kHz
            np.random.randn(20000),  # 20kHz
            np.random.randn(10000),  # 10kHz
        ]
        rates = [5000, 20000, 10000]
        predictions = model.predict_batch_universal(signals, original_sampling_rates=rates)
        
        for i, (pred, rate) in enumerate(zip(predictions, rates)):
            print(f"  Signal {i+1} ({rate}Hz): {pred['event_type']} "
                  f"(Risk: {pred['risk_score']:.2%})")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
