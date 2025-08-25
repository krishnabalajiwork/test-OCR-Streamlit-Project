#!/bin/bash

# EAST Model Downloader Script
# Downloads the pre-trained EAST text detection model

echo "📥 EAST Model Downloader"
echo "========================"
echo "This script downloads the pre-trained EAST text detection model"
echo "required for the OCR Streamlit application."
echo ""

# Check if model already exists
if [ -f "frozen_east_text_detection.pb" ]; then
    echo "✅ EAST model already exists!"
    echo "📁 Location: $(pwd)/frozen_east_text_detection.pb"
    exit 0
fi

# Ask for confirmation
read -p "Do you want to download the EAST model? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

echo ""
echo "🔄 Downloading EAST model..."
echo "================================"

# Download the model
MODEL_URL="https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1"
TAR_FILE="frozen_east_text_detection.tar.gz"

# Check if wget or curl is available
if command -v wget &> /dev/null; then
    echo "📥 Using wget to download..."
    wget -O "$TAR_FILE" "$MODEL_URL" --progress=bar:force 2>&1
elif command -v curl &> /dev/null; then
    echo "📥 Using curl to download..."
    curl -L -o "$TAR_FILE" "$MODEL_URL" --progress-bar
else
    echo "❌ Error: Neither wget nor curl is available."
    echo "Please install wget or curl, or use the Python script instead:"
    echo "   python download_scripts/download_model.py"
    exit 1
fi

# Check if download was successful
if [ $? -eq 0 ] && [ -f "$TAR_FILE" ]; then
    echo ""
    echo "📦 Extracting model..."

    # Extract the tar file
    tar -xzf "$TAR_FILE"

    # Check if extraction was successful
    if [ -f "frozen_east_text_detection.pb" ]; then
        # Get file size
        FILE_SIZE=$(du -h "frozen_east_text_detection.pb" | cut -f1)

        echo "✅ EAST model extracted successfully!"
        echo "📁 Model file: frozen_east_text_detection.pb (${FILE_SIZE})"
        echo "📍 Location: $(pwd)/frozen_east_text_detection.pb"

        # Clean up
        rm -f "$TAR_FILE"

        echo ""
        echo "🎉 Setup complete!"
        echo "You can now run the Streamlit app:"
        echo "   streamlit run app.py"
    else
        echo "❌ Model extraction failed!"
        rm -f "$TAR_FILE"
        exit 1
    fi
else
    echo "❌ Failed to download the model."
    echo "Please check your internet connection and try again."
    echo ""
    echo "Alternative: The main app will auto-download the model on first run."
    rm -f "$TAR_FILE"
    exit 1
fi
