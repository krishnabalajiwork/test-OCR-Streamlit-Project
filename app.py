import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import time
import os
import urllib.request
import tarfile
from imutils.object_detection import non_max_suppression

# Configure Streamlit page
st.set_page_config(
    page_title="OCR Text Detection",
    page_icon="📝",
    layout="wide"
)

# Title and description
st.title("📝 OCR and Text Detection")
st.markdown("**Extract text from images using EAST deep learning model and Tesseract OCR**")

# Function to download EAST model
def download_east_model():
    """Download EAST model if not present"""
    model_file = "frozen_east_text_detection.pb"

    if os.path.exists(model_file):
        return True

    try:
        with st.spinner("📥 Downloading EAST model for first use... Please wait..."):
            # Model URL
            model_url = "https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1"
            tar_filename = "frozen_east_text_detection.tar.gz"

            # Download
            urllib.request.urlretrieve(model_url, tar_filename)

            # Extract
            with tarfile.open(tar_filename, 'r:gz') as tar:
                tar.extractall('.')

            # Clean up
            os.remove(tar_filename)

            if os.path.exists(model_file):
                st.success("✅ EAST model downloaded successfully!")
                st.rerun()
                return True
            else:
                st.error("❌ Model extraction failed!")
                return False

    except Exception as e:
        st.error(f"❌ Error downloading model: {str(e)}")
        st.info("🔗 Please check your internet connection and try again.")
        return False

# Check and download model if needed
if not os.path.exists("frozen_east_text_detection.pb"):
    st.warning("⚠️ EAST model not found. Downloading now...")
    if not download_east_model():
        st.error("Failed to download model. Please refresh the page to try again.")
        st.stop()

# Sidebar for options
st.sidebar.title("⚙️ Options")
input_method = st.sidebar.radio(
    "Choose input method:",
    ("Upload Image", "Take Photo with Camera")
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.5, 
    step=0.1,
    help="Lower values detect more text but may include false positives"
)

# Load EAST model (cached for performance)
@st.cache_resource
def load_east_model():
    try:
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")
        return net
    except Exception as e:
        st.error(f"Error loading EAST model: {str(e)}")
        return None

def decode_predictions(scores, geometry, confidence_threshold):
    """Decode EAST model predictions"""
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < confidence_threshold:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

def detect_text_regions(image, net, confidence_threshold):
    """Detect text regions using EAST model"""
    if net is None:
        return None, None

    (H, W) = image.shape[:2]
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    # Resize image and prepare blob
    resized = cv2.resize(image, (newW, newH))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (newW, newH),
                                (123.68, 116.78, 103.94), swapRB=True, crop=False)

    # Get predictions
    net.setInput(blob)
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    (scores, geometry) = net.forward(layerNames)

    # Decode predictions
    (rects, confidences) = decode_predictions(scores, geometry, confidence_threshold)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # Scale boxes back to original image size
    results = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        results.append((startX, startY, endX, endY))

    return results, image

def extract_text_from_regions(image, boxes):
    """Extract text using Tesseract OCR"""
    results = []

    for (startX, startY, endX, endY) in boxes:
        # Add padding
        dX = int((endX - startX) * 0.1)
        dY = int((endY - startY) * 0.1)
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(image.shape[1], endX + dX)
        endY = min(image.shape[0], endY + dY)

        # Extract ROI
        roi = image[startY:endY, startX:endX]

        try:
            # Configure Tesseract
            config = "-l eng --oem 1 --psm 8"
            text = pytesseract.image_to_string(roi, config=config)
            text = text.strip().replace('\n', ' ')

            if text:
                results.append(((startX, startY, endX, endY), text))
        except Exception as e:
            # Silently continue if OCR fails for a region
            continue

    return results

def draw_results(image, results):
    """Draw bounding boxes and text on image"""
    output_image = image.copy()

    for i, ((startX, startY, endX, endY), text) in enumerate(results):
        # Draw bounding box
        cv2.rectangle(output_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Add text label
        y = startY - 10 if startY - 10 > 10 else startY + 20
        cv2.putText(output_image, f"{i+1}", (startX, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return output_image

def process_image(image, confidence_threshold):
    """Main processing function"""
    # Load EAST model
    net = load_east_model()
    if net is None:
        return None, []

    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect text regions
    with st.spinner("🔍 Detecting text regions..."):
        boxes, _ = detect_text_regions(image, net, confidence_threshold)

    if not boxes:
        st.warning("No text regions detected! Try adjusting the confidence threshold.")
        return image, []

    # Extract text from regions
    with st.spinner("📝 Extracting text..."):
        results = extract_text_from_regions(image, boxes)

    # Draw results
    output_image = draw_results(image, results)

    return output_image, results

# Main app logic
if input_method == "Upload Image":
    st.subheader("📁 Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image containing text for OCR analysis"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        # Process image
        if st.button("🔍 Extract Text", type="primary"):
            start_time = time.time()

            output_image, results = process_image(image, confidence_threshold)

            if output_image is not None:
                processing_time = time.time() - start_time

                with col2:
                    st.subheader("Detected Text Regions")
                    # Convert back to RGB for display
                    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    st.image(output_rgb, use_container_width=True)

                # Display results
                st.subheader("📋 Extracted Text")
                if results:
                    st.success(f"Found {len(results)} text region(s) in {processing_time:.2f} seconds")

                    for i, ((startX, startY, endX, endY), text) in enumerate(results):
                        with st.expander(f"Text Region {i+1}: {text[:50]}..."):
                            st.write(f"**Text:** {text}")
                            st.write(f"**Coordinates:** ({startX}, {startY}) to ({endX}, {endY})")

                    # Download results
                    text_results = "\n".join([f"Region {i+1}: {text}" for i, (_, text) in enumerate(results)])
                    st.download_button(
                        "📥 Download Extracted Text",
                        data=text_results,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("No text found in the image. Try adjusting the confidence threshold.")

elif input_method == "Take Photo with Camera":
    st.subheader("📷 Camera Capture")

    # Camera input
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        # Display captured image
        image = Image.open(camera_image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Captured Image")
            st.image(image, use_container_width=True)

        # Process image
        if st.button("🔍 Extract Text from Photo", type="primary"):
            start_time = time.time()

            output_image, results = process_image(image, confidence_threshold)

            if output_image is not None:
                processing_time = time.time() - start_time

                with col2:
                    st.subheader("Detected Text Regions")
                    # Convert back to RGB for display
                    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    st.image(output_rgb, use_container_width=True)

                # Display results
                st.subheader("📋 Extracted Text")
                if results:
                    st.success(f"Found {len(results)} text region(s) in {processing_time:.2f} seconds")

                    for i, ((startX, startY, endX, endY), text) in enumerate(results):
                        with st.expander(f"Text Region {i+1}: {text[:50]}..."):
                            st.write(f"**Text:** {text}")
                            st.write(f"**Coordinates:** ({startX}, {startY}) to ({endX}, {endY})")

                    # Download results
                    text_results = "\n".join([f"Region {i+1}: {text}" for i, (_, text) in enumerate(results)])
                    st.download_button(
                        "📥 Download Extracted Text",
                        data=text_results,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("No text found in the image. Try adjusting the confidence threshold.")

# Footer
st.markdown("---")
st.markdown("**Built with Streamlit • EAST Model • Tesseract OCR**")

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    **Upload Image:**
    1. Click 'Browse files' to upload an image
    2. Adjust confidence threshold if needed (0.1-1.0)
    3. Click 'Extract Text' to process

    **Camera Capture:**
    1. Switch to 'Take Photo with Camera'
    2. Allow camera permissions
    3. Take a photo and click 'Extract Text from Photo'

    **Tips:**
    - Use clear, well-lit images for best results
    - Lower confidence = more detections but possible false positives
    - Higher confidence = fewer but more accurate detections
    - The EAST model will download automatically on first use
    """)
