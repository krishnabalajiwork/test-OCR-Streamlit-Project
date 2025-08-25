# 📝 OCR Text Detection - Streamlit App

A powerful web application for extracting text from images using **EAST deep learning model** for text detection and **Tesseract OCR** for text recognition.

![OCR Demo](https://img.shields.io/badge/OCR-Text%20Detection-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) ![Python](https://img.shields.io/badge/Python-3.7+-green)

## 🚀 Live Demo

Deploy this app instantly on:
- **Streamlit Cloud**: https://test-ocr-east.streamlit.app

## ✨ Features

- 📁 **Upload Images**: Support for JPG, PNG, BMP, TIFF formats
- 📷 **Live Camera**: Take photos directly from webcam
- 🤖 **Auto Model Download**: EAST model downloads automatically on first use
- 🎯 **Smart Detection**: Adjustable confidence threshold (0.1-1.0)
- 📝 **Text Extraction**: Clean, formatted text output with coordinates
- 📥 **Download Results**: Export extracted text as TXT file
- 📱 **Mobile Friendly**: Responsive design works on all devices
- 🔧 **Zero Configuration**: No manual model setup required

## 🛠️ Quick Deploy

### Option 1: Streamlit Cloud (Recommended)
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file: `app.py`
5. Deploy! (Model downloads automatically)

### Option 2: Hugging Face Spaces
1. Create new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Streamlit** SDK
3. Upload all files from this repository
4. Deploy!

### Option 3: Run Locally
```bash
# Clone repository
git clone <your-repo-url>
cd OCR-Streamlit-Project

# Install Python dependencies
pip install -r requirements.txt

# Install Tesseract OCR
# Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-eng
# macOS: brew install tesseract
# Windows: Download from GitHub releases

# Run the app
streamlit run app.py
```

## 🔧 How It Works

1. **Text Detection**: EAST (Efficient and Accurate Scene Text) deep learning model locates text regions
2. **Text Recognition**: Tesseract OCR engine extracts readable text from detected regions
3. **Auto Setup**: EAST model downloads automatically on first use (no manual setup!)
4. **Results**: View detected text with bounding box coordinates and download as text file

## 📋 Project Structure

```
OCR-Streamlit-Project/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── packages.txt               # System packages (Tesseract OCR)
├── README.md                  # This documentation
├── DEPLOY.md                  # Deployment instructions
├── LICENSE                    # MIT License
├── .gitignore                 # Git ignore settings
├── download_scripts/          # Manual model download scripts
│   ├── download_model.py      # Python download script
│   └── download_model.sh      # Bash download script
├── utils/                     # Utility scripts
│   ├── batch_processor.py     # Batch processing for multiple images
│   └── evaluation.py          # Model performance evaluation
└── .streamlit/                # Streamlit configuration
    └── config.toml            # App theme and server settings
```

## ⚙️ Usage Tips

### Image Quality
- Use **clear, well-lit images** for best results
- **High contrast** between text and background improves accuracy
- **Avoid blurry or low-resolution** images

### Confidence Threshold Settings
- **Low (0.1-0.3)**: More text detections, but may include false positives
- **Medium (0.4-0.6)**: Balanced accuracy and detection rate (recommended)
- **High (0.7-1.0)**: Conservative detection, high precision but may miss some text

### Supported Formats
- **Image Upload**: JPG, JPEG, PNG, BMP, TIFF
- **Camera**: Real-time photo capture from webcam
- **Output**: Plain text file download with extracted text and coordinates

## 🚀 Zero Configuration Deployment

This app is designed for **instant deployment** with no manual setup:

- ✅ **No model files to upload** (auto-downloads on first use)
- ✅ **All dependencies specified** in requirements files
- ✅ **Works on free hosting platforms** (Streamlit Cloud, Hugging Face)
- ✅ **Mobile responsive design**
- ✅ **Cross-platform compatibility**

## 🔨 Advanced Usage

### Batch Processing
Use `utils/batch_processor.py` to process multiple images:
```bash
python utils/batch_processor.py --input_dir images/ --output_dir results/
```

### Model Evaluation
Evaluate performance with ground truth data using `utils/evaluation.py`:
```bash
python utils/evaluation.py --predictions results/ --ground_truth annotations/
```

### Manual Model Download
If needed, manually download the EAST model:
```bash
python download_scripts/download_model.py
```

## 📊 Performance

- **Text Detection**: ~90%+ accuracy on clear images
- **Processing Speed**: ~2-5 seconds per image (depending on size and complexity)
- **Model Size**: ~84MB (downloads automatically)
- **Supported Languages**: English (extensible to other languages via Tesseract)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Commit: `git commit -am 'Add new feature'`
5. Push: `git push origin feature-name`
6. Submit a pull request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/your-username/OCR-Streamlit-Project.git
cd OCR-Streamlit-Project

# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest  # For code formatting and testing

# Run tests
python -m pytest tests/

# Format code
black app.py utils/
```

## 📞 Support & Issues

- 📧 **Contact**: krishnabalajiwork@gmail.com
- 📖 **Documentation**: See `DEPLOY.md` for deployment help

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **EAST Model**: ["EAST: An Efficient and Accurate Scene Text Detector"](https://arxiv.org/abs/1704.03155) by Zhou et al.
- **Tesseract OCR**: Google's open-source OCR engine
- **Streamlit**: Amazing framework for building ML web apps
- **OpenCV**: Computer vision library for image processing

## 📈 Roadmap

- [ ] Multi-language support (Arabic, Chinese, etc.)
- [ ] Handwriting recognition
- [ ] PDF text extraction
- [ ] Batch upload interface
- [ ] API endpoint for programmatic access
- [ ] Docker containerization
- [ ] Performance optimizations

---

**⭐ Star this repository if you found it helpful!**

**Built with ❤️ using Streamlit, EAST Model & Tesseract OCR**
