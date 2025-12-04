# Deployment Guide

## Quick Start

### Local Testing

1. **Install dependencies:**
```bash
cd web
pip install -r requirements.txt
```

2. **Run the application:**
```bash
python app.py
```

3. **Open browser:**
```
http://localhost:5000
```

## Free Hosting Options

### Option 1: Render (Recommended)

**Steps:**

1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `universal-optics-tracker` (or your choice)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Root Directory**: `web` (if deploying from root, or leave blank if deploying web folder)

5. Click "Create Web Service"

**Your app will be live at:** `https://your-app-name.onrender.com`

**Note:** Free tier sleeps after 15 minutes of inactivity, but wakes automatically on first request.

### Option 2: Railway

**Steps:**

1. Go to [railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway will auto-detect Python
5. Set root directory to `web` if needed
6. Deploy!

**Your app will be live at:** `https://your-app-name.railway.app`

### Option 3: PythonAnywhere (Free Tier)

**Steps:**

1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload your code via Git or files
3. Create a new web app
4. Configure WSGI file to point to `app.py`
5. Reload!

**Note:** Free tier has limited CPU time and external access restrictions.

## Important Notes

### Model File Location

The web app expects the model at:
- `../models/trained_model.pth` (relative to web folder)
- Or `models/trained_model.pth` (if in same directory)

Make sure the model file is included in your repository or uploaded separately.

### Environment Variables

No environment variables required for basic deployment. The app will:
- Auto-detect model path
- Use port from environment (or default 5000)
- Handle CORS automatically

### File Size Limits

- Render: 100MB per file upload
- Railway: 512MB per file upload
- PythonAnywhere: 100MB per file upload

For larger files, consider:
- Client-side compression
- Streaming uploads
- External storage (S3, etc.)

## Troubleshooting

### Model Not Found

**Error:** `Model file not found`

**Solution:**
1. Check model path in `api/inference.py`
2. Ensure `models/trained_model.pth` exists
3. Verify file permissions

### Import Errors

**Error:** `ModuleNotFoundError`

**Solution:**
1. Ensure all dependencies in `requirements.txt` are installed
2. Check Python version (3.8+)
3. Verify path setup in `app.py` and `api/inference.py`

### CORS Issues

**Error:** CORS policy blocking requests

**Solution:**
- CORS is enabled by default in `app.py`
- If issues persist, check browser console
- Verify Flask-CORS is installed

### Cold Start Delays

**Issue:** First request takes 10-20 seconds

**Explanation:**
- Model loads on first request
- This is normal for free hosting
- Subsequent requests are fast

**Solution:**
- Use "always on" option (paid tier)
- Or accept cold start delay (free tier)

## Production Checklist

- [ ] Model file included/accessible
- [ ] All dependencies in requirements.txt
- [ ] Environment variables set (if needed)
- [ ] CORS configured correctly
- [ ] Error handling tested
- [ ] File upload limits understood
- [ ] HTTPS enabled (automatic on Render/Railway)
- [ ] Domain configured (optional)

## Support

For issues:
1. Check logs in hosting dashboard
2. Test locally first
3. Verify model file exists
4. Check Python version compatibility





