#!/usr/bin/env python3
"""
Download EAST text detection model for OCR application
"""

import os
import sys
import urllib.request
import tarfile
from pathlib import Path

def download_east_model():
    """Download and extract EAST model if not present"""

    model_file = "frozen_east_text_detection.pb"

    # Check if model already exists
    if os.path.exists(model_file):
        print("✅ EAST model already exists!")
        return True

    print("📥 Downloading EAST Text Detection Model...")
    print("=" * 50)

    # Model URL
    model_url = "https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1"
    tar_filename = "frozen_east_text_detection.tar.gz"

    try:
        # Download with progress
        print("🔄 Downloading from Dropbox...")
        print(f"URL: {model_url}")

        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0 * downloaded / total_size, 100.0)
                print(f"\rProgress: {percent:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)", end="")

        urllib.request.urlretrieve(model_url, tar_filename, show_progress)
        print("\n📦 Download completed!")

        # Extract the tar file
        print("📂 Extracting model...")
        with tarfile.open(tar_filename, 'r:gz') as tar:
            tar.extractall('.')

        # Clean up
        os.remove(tar_filename)

        # Verify extraction
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"✅ EAST model extracted successfully!")
            print(f"📁 Model file: {model_file} ({file_size:.1f} MB)")
            print(f"📍 Location: {os.path.abspath(model_file)}")
            return True
        else:
            print("❌ Model extraction failed!")
            return False

    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        # Clean up partial download
        if os.path.exists(tar_filename):
            os.remove(tar_filename)
        return False

def main():
    """Main function"""
    print("EAST Model Downloader")
    print("=" * 30)
    print("This script downloads the pre-trained EAST text detection model")
    print("required for the OCR Streamlit application.\n")

    # Ask for confirmation
    response = input("Do you want to download the EAST model? (y/n): ").lower().strip()

    if response not in ['y', 'yes']:
        print("Download cancelled.")
        return

    success = download_east_model()

    if success:
        print("\n🎉 Setup complete!")
        print("You can now run the Streamlit app:")
        print("   streamlit run app.py")
        print("\nOr use the model in your own scripts.")
    else:
        print("\n❌ Setup failed. Please try again or check your internet connection.")
        print("\nAlternative: The main app will auto-download the model on first run.")
        sys.exit(1)

if __name__ == "__main__":
    main()
