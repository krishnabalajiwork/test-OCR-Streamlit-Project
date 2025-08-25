# 🚀 Deployment Guide

This guide provides step-by-step instructions for deploying your OCR Text Detection Streamlit app on various platforms.

## 🎯 Quick 3-Step Deployment

### Step 1: Upload to GitHub
1. Create a new repository on [GitHub](https://github.com)
2. Upload **ALL files** from this project folder
3. Make the repository **public** (required for free deployments)

### Step 2: Choose Your Platform

#### Option A: Streamlit Cloud (Recommended - FREE)
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository from the dropdown
5. Set **Branch**: `main` (or `master`)
6. Set **Main file path**: `app.py`
7. Click **"Deploy!"**

**Deployment Time**: 2-5 minutes
**Cost**: Free
**Custom Domain**: Available with custom subdomain

#### Option B: Hugging Face Spaces (FREE)
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Choose **Streamlit** as the SDK
4. Upload all project files or connect your GitHub repo
5. The app will build and deploy automatically

**Deployment Time**: 5-10 minutes
**Cost**: Free
**Features**: Community features, GPU option available

#### Option C: Railway (FREE Tier Available)
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click **"New Project"** → **"Deploy from GitHub repo"**
4. Select your repository
5. Railway will automatically detect it's a Python app
6. Add environment variables if needed
7. Deploy!

### Step 3: Test Your Deployment
- Wait for deployment to complete (usually 2-10 minutes)
- Click the provided URL to access your app
- Test image upload and camera functionality
- The EAST model will download automatically on first use

## 📋 Pre-Deployment Checklist

- [ ] All files uploaded to GitHub repository
- [ ] Repository is public
- [ ] `app.py` is in the root directory
- [ ] `requirements.txt` and `packages.txt` are included
- [ ] No large files (>100MB) in repository
- [ ] README.md has been customized with your info

## ⚙️ Platform-Specific Configuration

### Streamlit Cloud
- **Automatic**: Uses `requirements.txt` and `packages.txt`
- **Python Version**: 3.9+ (automatic)
- **Resource Limits**: 1GB RAM, 1 CPU core
- **Custom Domains**: Available

### Hugging Face Spaces
- **Configuration**: Automatic detection
- **Resource Options**: CPU (free) or GPU (paid)
- **Community Features**: Comments, likes, discussions
- **Persistent Storage**: Available

### Railway
- **Build Command**: Automatic (Python detection)
- **Start Command**: `streamlit run app.py --server.port=$PORT`
- **Environment Variables**: Set `PORT` if needed
- **Resource Limits**: 500MB RAM (free tier)

## 🛠️ Advanced Deployment Options

### Docker Deployment
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t ocr-streamlit .
docker run -p 8501:8501 ocr-streamlit
```

### Local Development Server
```bash
# Install dependencies
pip install -r requirements.txt

# Install Tesseract (system-dependent)
# Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-eng
# macOS: brew install tesseract
# Windows: Download from GitHub

# Run locally
streamlit run app.py
```

## 🔧 Environment Variables (Optional)

You can customize the app behavior using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `STREAMLIT_SERVER_PORT` | Server port | 8501 |
| `STREAMLIT_SERVER_ADDRESS` | Server address | localhost |
| `TESSERACT_CMD` | Tesseract command path | tesseract |

### Setting Environment Variables

**Streamlit Cloud**: Go to App Settings → Advanced → Environment Variables

**Hugging Face**: Add to `secrets.toml` file:
```toml
TESSERACT_CMD = "/usr/bin/tesseract"
```

**Railway**: Set in dashboard under Variables tab

## 🚨 Troubleshooting Common Issues

### 1. "Requirements not found" Error
- **Solution**: Ensure `requirements.txt` is in the root directory
- **Check**: File name is exactly `requirements.txt` (not `requirements.txt.txt`)

### 2. "Tesseract not found" Error
- **Solution**: Ensure `packages.txt` includes `tesseract-ocr` and `tesseract-ocr-eng`
- **Platform**: Some platforms may not support system packages

### 3. "EAST model download failed" Error
- **Solution**: Check internet connectivity on deployment platform
- **Alternative**: Pre-download model and include in repository (if size allows)

### 4. "Port already in use" Error
- **Solution**: Use environment variable `PORT` or change default port
- **Local**: Kill existing processes on port 8501

### 5. Memory/Resource Limits
- **Solution**: Optimize image processing, reduce model size if possible
- **Upgrade**: Consider paid tiers for more resources

## 📊 Deployment Comparison

| Platform | Cost | Ease | Performance | Features |
|----------|------|------|-------------|----------|
| Streamlit Cloud | Free | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Basic |
| Hugging Face | Free | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Community |
| Railway | Free/Paid | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Advanced |
| Docker | Varies | ⭐⭐ | ⭐⭐⭐⭐⭐ | Full Control |

## 📞 Support

If you encounter issues during deployment:

1. **Check the logs** in your deployment platform dashboard
2. **Verify all files** are uploaded correctly
3. **Test locally** first to ensure the app works
4. **Open an issue** in the GitHub repository with:
   - Platform used
   - Error messages
   - Steps to reproduce

## 🎉 Post-Deployment

After successful deployment:

1. **Test all features** (upload, camera, text extraction)
2. **Share your app** with the community
3. **Monitor usage** through platform dashboards
4. **Update regularly** to keep dependencies current
5. **Gather feedback** from users for improvements

---

**🚀 Happy Deploying!**

Your OCR Text Detection app is now ready to help users extract text from images around the world!
