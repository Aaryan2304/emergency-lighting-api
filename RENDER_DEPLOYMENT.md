# Render Deployment Checklist

## ✅ Pre-Deployment Setup

- [x] Created `render.yaml` configuration
- [x] Created `requirements-render.txt` with optimized dependencies
- [x] Added health check endpoint (`/health`)
- [x] Created Postman collection for testing
- [x] Environment variables configured in `render.yaml`

## 🚀 Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Render deployment configuration"
   git push origin main
   ```

2. **Create Render Service**
   - Go to https://render.com
   - Sign up/Login
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Select the repository with your code

3. **Set Environment Variables in Render Dashboard**
   - `GEMINI_API_KEY`: Your Google Gemini API key from https://aistudio.google.com/app/apikey
   - Other variables are auto-set from `render.yaml`

4. **Deploy**
   - Render will automatically build and deploy
   - Monitor the build logs for any issues
   - Once deployed, note your service URL

## 🧪 Testing

After deployment, test these endpoints:

1. **Health Check**
   ```
   GET https://your-service-name.onrender.com/health
   Expected: {"status": "healthy", "service": "emergency-lighting-api"}
   ```

2. **Upload Blueprint**
   ```
   POST https://your-service-name.onrender.com/api/v1/blueprints/upload
   Body: multipart/form-data with PDF file
   Expected: {"status": "uploaded", "pdf_name": "...", "message": "Processing started in background."}
   ```

3. **Get Results** (after processing completes)
   ```
   GET https://your-service-name.onrender.com/api/v1/blueprints/result?pdf_name=yourfile.pdf
   Expected: Grouped lighting fixture results
   ```

## 📋 Submission Requirements

For the submission, ensure you have:

1. ✅ **Screenshot of Annotation**: Show detected emergency lights with bounding boxes
2. ✅ **Hosted API**: Working on Render with both endpoints
3. ✅ **Postman Collection**: Include the `.json` file
4. ✅ **GitHub Repository**: Full source code with README
5. ✅ **Demo Video**: 2-minute walkthrough

## 🔧 Troubleshooting

### Common Issues:
- **Build fails**: Check `requirements-render.txt` for compatible packages
- **Timeout**: Increase build time in Render settings
- **Memory issues**: Use opencv-python-headless instead of opencv-python
- **Port issues**: Ensure API_PORT is set to 10000 for Render

### Debug Commands:
```bash
# Local testing
python main.py --mode api

# Check logs
tail -f logs/app.log
```
