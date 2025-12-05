# Universal Optics Detection and Degradation Tracker - Web Application

Production-ready web application for the Universal Fiber Sensor Model.

## Features

- **Single-page design** with smooth scrolling
- **Two prediction modes**: Standard and Universal
- **File upload** with drag & drop support
- **Interactive results** with expandable metric explanations
- **Dashboard visualizations** with charts and statistics
- **Professional design** optimized for research/academic use

## Local Development

### Prerequisites

- Python 3.8+
- All dependencies from main project requirements.txt

### Installation

1. Install dependencies:
```bash
cd web
pip install -r requirements.txt
```

2. Ensure the model file exists:
```
models/trained_model.pth
```

3. Run the application:
```bash
python app.py
```

4. Open your browser to:
```
http://localhost:5000
```

## Deployment

### Option 1: Render (Recommended - Free)

1. Create a new account at [render.com](https://render.com)

2. Create a new "Web Service"

3. Connect your GitHub repository

4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment**: Python 3

5. Add environment variables (if needed):
   - `PORT`: 5000 (usually auto-set)

6. Deploy!

Your app will be available at: `https://your-app-name.onrender.com`

### Option 2: Railway (Free Tier)

1. Create account at [railway.app](https://railway.app)

2. Create new project from GitHub

3. Add `Procfile` with:
```
web: gunicorn app:app
```

4. Deploy automatically!

Your app will be available at: `https://your-app-name.railway.app`

### Option 3: Heroku (Paid)

1. Install Heroku CLI

2. Create `Procfile`:
```
web: gunicorn app:app
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## File Structure

```
web/
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── api/
│   └── inference.py       # Model inference wrapper
├── templates/
│   └── index.html         # Main HTML page
├── static/
│   ├── css/
│   │   └── style.css      # Styling
│   ├── js/
│   │   ├── app.js         # Main application logic
│   │   └── charts.js      # Chart visualizations
│   └── assets/            # Images, icons
└── README.md              # This file
```

## API Endpoints

### Health Check
```
GET /api/health
```

### Standard Prediction
```
POST /api/predict/standard
Content-Type: multipart/form-data

Parameters:
- file: Signal file (CSV, TXT, or .npy)
- sampling_rate: Sampling rate in Hz (default: 10000)
- is_multichannel: true/false (default: false)
```

### Universal Prediction
```
POST /api/predict/universal
Content-Type: multipart/form-data

Parameters:
- file: Signal file (CSV, TXT, or .npy)
- original_sampling_rate: Original sampling rate in Hz (optional)
- is_multichannel: true/false (default: false)
```

### Training Statistics
```
GET /api/training-stats
```

## Troubleshooting

### Model Not Loading

- Ensure `models/trained_model.pth` exists in the parent directory
- Check file permissions
- Verify model path in `api/inference.py`

### File Upload Issues

- Check file format (CSV, TXT, or .npy)
- Ensure file is not empty
- Verify file encoding (UTF-8)

### CORS Errors

- CORS is enabled by default
- If issues persist, check `app.py` CORS configuration

## Notes

- Model loads on first request (may take 5-10 seconds)
- Large files may take longer to process
- Free hosting services may have cold start delays

## License

Same as main project.






