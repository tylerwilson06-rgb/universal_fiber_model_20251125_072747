"""
Flask backend for Universal Optics Detection and Degradation Tracker
Production-ready API server for model inference
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import traceback
import os
import sys
from pathlib import Path

# Add parent directory to path for model imports
web_dir = Path(__file__).parent
parent_dir = web_dir.parent
sys.path.insert(0, str(parent_dir))

# Import after path setup
from api.inference import ModelAPI

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Initialize model API (lazy loading)
model_api = None

def get_model_api():
    """Get or initialize model API"""
    global model_api
    if model_api is None:
        try:
            model_api = ModelAPI()
        except Exception as e:
            app.logger.error(f"Failed to initialize model: {e}")
            raise
    return model_api

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        api = get_model_api()
        return jsonify({
            'status': 'healthy',
            'model_loaded': api.model is not None
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/predict/standard', methods=['POST'])
def predict_standard():
    """Standard mode prediction endpoint"""
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Get parameters
        sampling_rate = float(request.form.get('sampling_rate', 10000))
        is_multichannel = request.form.get('is_multichannel', 'false').lower() == 'true'
        
        # Parse file
        file_content = file.read()
        signal = model_api.parse_file(file_content, file.filename)
        
        # Validate signal
        if signal.size == 0:
            return jsonify({
                'success': False,
                'error': 'Signal is empty'
            }), 400
        
        # Make prediction
        api = get_model_api()
        result = api.predict_standard(signal, sampling_rate, is_multichannel)
        
        if result['success']:
            # Add signal info
            result['signal_info'] = {
                'length': len(signal),
                'shape': signal.shape,
                'sampling_rate': sampling_rate,
                'is_multichannel': is_multichannel
            }
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        app.logger.error(f"Error in standard prediction: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/predict/universal', methods=['POST'])
def predict_universal():
    """Universal mode prediction endpoint"""
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Get parameters
        original_sampling_rate = request.form.get('original_sampling_rate')
        if original_sampling_rate:
            original_sampling_rate = float(original_sampling_rate)
        else:
            original_sampling_rate = None
        
        is_multichannel = request.form.get('is_multichannel', 'false').lower() == 'true'
        
        # Parse file
        file_content = file.read()
        signal = model_api.parse_file(file_content, file.filename)
        
        # Validate signal
        if signal.size == 0:
            return jsonify({
                'success': False,
                'error': 'Signal is empty'
            }), 400
        
        # Make prediction
        api = get_model_api()
        result = api.predict_universal(signal, original_sampling_rate, is_multichannel)
        
        if result['success']:
            # Add signal info
            result['signal_info'] = {
                'length': len(signal),
                'shape': signal.shape,
                'original_sampling_rate': original_sampling_rate,
                'is_multichannel': is_multichannel
            }
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        app.logger.error(f"Error in universal prediction: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/training-stats', methods=['GET'])
def training_stats():
    """Return training data statistics for dashboard"""
    return jsonify({
        'datasets': [
            {
                'name': 'DAS',
                'samples': 6456,
                'classes': 9,
                'accuracy': 80.57,
                'task': 'Event Classification'
            },
            {
                'name': 'Phi-OTDR',
                'samples': 15418,
                'classes': 6,
                'accuracy': 94.71,
                'task': 'Event Classification'
            },
            {
                'name': 'OTDR',
                'samples': 180,
                'classes': 4,
                'accuracy': 100.00,
                'task': 'Damage Detection'
            }
        ],
        'total_samples': 22054,
        'risk_mse': 0.0006
    }), 200

if __name__ == '__main__':
    # Initialize model on startup
    try:
        get_model_api()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Model not loaded: {e}")
        print("Model will be loaded on first request")
    
    # Run app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

