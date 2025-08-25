#!/usr/bin/env python3
"""
Batch OCR Processing Utility

This script processes multiple images for text detection and OCR using
the EAST model and Tesseract OCR, with detailed logging and statistics.
"""

import os
import sys
import argparse
import json
import time
import csv
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging
from imutils.object_detection import non_max_suppression

# Add parent directory to path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging(log_dir="logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"batch_processing_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__), log_file

class BatchOCRProcessor:
    """Batch processor for OCR and text detection"""

    def __init__(self, east_model_path, output_dir="batch_results", 
                 min_confidence=0.5, width=320, height=320):
        self.east_model_path = east_model_path
        self.output_dir = output_dir
        self.min_confidence = min_confidence
        self.width = width
        self.height = height
        self.logger = logging.getLogger(__name__)

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "texts"), exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Load EAST model
        self.net = self._load_east_model()

        # Statistics
        self.stats = {
            "total_images": 0,
            "successful_processes": 0,
            "failed_processes": 0,
            "total_text_regions": 0,
            "processing_time": 0,
            "start_time": None,
            "end_time": None
        }

    def _load_east_model(self):
        """Load EAST model"""
        try:
            if not os.path.exists(self.east_model_path):
                raise FileNotFoundError(f"EAST model not found: {self.east_model_path}")

            net = cv2.dnn.readNet(self.east_model_path)
            self.logger.info(f"EAST model loaded successfully from {self.east_model_path}")
            return net
        except Exception as e:
            self.logger.error(f"Failed to load EAST model: {str(e)}")
            return None

    def get_image_files(self, directory):
        """Get all image files from a directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))

        return sorted(image_files)

    def decode_predictions(self, scores, geometry):
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
                if scoresData[x] < self.min_confidence:
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

    def detect_text_regions(self, image):
        """Detect text regions using EAST model"""
        if self.net is None:
            return []

        (H, W) = image.shape[:2]
        rW = W / float(self.width)
        rH = H / float(self.height)

        # Resize image and prepare blob
        resized = cv2.resize(image, (self.width, self.height))
        blob = cv2.dnn.blobFromImage(resized, 1.0, (self.width, self.height),
                                    (123.68, 116.78, 103.94), swapRB=True, crop=False)

        # Get predictions
        self.net.setInput(blob)
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        (scores, geometry) = self.net.forward(layerNames)

        # Decode predictions
        (rects, confidences) = self.decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # Scale boxes back to original image size
        results = []
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            results.append((startX, startY, endX, endY))

        return results

    def extract_text_from_regions(self, image, boxes):
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
                self.logger.warning(f"OCR failed for a region: {str(e)}")
                continue

        return results

    def process_single_image(self, image_path):
        """Process a single image and return results"""
        try:
            self.logger.info(f"Processing: {image_path}")
            start_time = time.time()

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Detect text regions
            boxes = self.detect_text_regions(image)

            if not boxes:
                self.logger.warning(f"No text regions detected in: {image_path}")
                self.stats["successful_processes"] += 1
                return {
                    "image_path": image_path,
                    "base_name": os.path.splitext(os.path.basename(image_path))[0],
                    "num_text_regions": 0,
                    "processing_time": time.time() - start_time,
                    "text_regions": [],
                    "status": "success_no_text"
                }

            # Extract text from regions
            results = self.extract_text_from_regions(image, boxes)

            end_time = time.time()
            processing_time = end_time - start_time

            # Save results
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            # Save annotated image
            output_image_path = os.path.join(
                self.output_dir, "images", f"{base_name}_detected.jpg"
            )

            # Create annotated image
            output_image = image.copy()
            for i, ((startX, startY, endX, endY), text) in enumerate(results):
                cv2.rectangle(output_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 10 if startY - 10 > 10 else startY + 20
                cv2.putText(output_image, str(i+1), (startX, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imwrite(output_image_path, output_image)

            # Save text results
            text_output_path = os.path.join(
                self.output_dir, "texts", f"{base_name}_text.txt"
            )

            with open(text_output_path, 'w', encoding='utf-8') as f:
                f.write(f"Image: {os.path.basename(image_path)}\n")
                f.write(f"Processing time: {processing_time:.3f} seconds\n")
                f.write(f"Number of text regions: {len(results)}\n")
                f.write("-" * 50 + "\n")

                for i, ((startX, startY, endX, endY), text) in enumerate(results):
                    f.write(f"Text {i+1}: {text}\n")
                    f.write(f"Coordinates: ({startX}, {startY}) to ({endX}, {endY})\n\n")

            # Update statistics
            self.stats["successful_processes"] += 1
            self.stats["total_text_regions"] += len(results)
            self.stats["processing_time"] += processing_time

            self.logger.info(f"Successfully processed: {image_path} "
                           f"({len(results)} text regions, {processing_time:.3f}s)")

            return {
                "image_path": image_path,
                "base_name": base_name,
                "num_text_regions": len(results),
                "processing_time": processing_time,
                "text_regions": results,
                "output_image_path": output_image_path,
                "text_output_path": text_output_path,
                "status": "success"
            }

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            self.stats["failed_processes"] += 1
            return {
                "image_path": image_path,
                "base_name": os.path.splitext(os.path.basename(image_path))[0],
                "error": str(e),
                "status": "failed"
            }

    def process_directory(self, input_directory):
        """Process all images in a directory"""
        self.logger.info(f"Starting batch processing of directory: {input_directory}")
        self.stats["start_time"] = datetime.now()

        # Get all image files
        image_files = self.get_image_files(input_directory)
        self.stats["total_images"] = len(image_files)

        if not image_files:
            self.logger.warning(f"No image files found in {input_directory}")
            return []

        self.logger.info(f"Found {len(image_files)} image(s) to process")

        # Process each image
        results = []
        for i, image_path in enumerate(image_files, 1):
            self.logger.info(f"Processing image {i}/{len(image_files)}")
            result = self.process_single_image(image_path)
            results.append(result)

        self.stats["end_time"] = datetime.now()

        # Save comprehensive results
        self.save_batch_results(results)
        self.print_summary()

        return results

    def save_batch_results(self, results):
        """Save comprehensive batch processing results"""
        # Prepare summary data
        summary = {
            "processing_summary": self.stats.copy(),
            "configuration": {
                "east_model_path": self.east_model_path,
                "min_confidence": self.min_confidence,
                "input_dimensions": (self.width, self.height)
            },
            "individual_results": []
        }

        # Convert datetime objects to strings
        if summary["processing_summary"]["start_time"]:
            summary["processing_summary"]["start_time"] = self.stats["start_time"].isoformat()
        if summary["processing_summary"]["end_time"]:
            summary["processing_summary"]["end_time"] = self.stats["end_time"].isoformat()

        # Add individual results (without full text regions to keep file size manageable)
        for result in results:
            summary_result = {
                "image_path": result["image_path"],
                "base_name": result["base_name"],
                "num_text_regions": result.get("num_text_regions", 0),
                "processing_time": result.get("processing_time", 0),
                "status": result["status"]
            }
            if "output_image_path" in result:
                summary_result["output_image_path"] = result["output_image_path"]
            if "text_output_path" in result:
                summary_result["text_output_path"] = result["text_output_path"]
            if "error" in result:
                summary_result["error"] = result["error"]

            summary["individual_results"].append(summary_result)

        # Save summary
        summary_path = os.path.join(self.output_dir, "batch_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Batch summary saved to: {summary_path}")

        # Save CSV report
        csv_path = os.path.join(self.output_dir, "batch_report.csv")
        self.save_csv_report(results, csv_path)

    def save_csv_report(self, results, csv_path):
        """Save results as CSV report"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_name', 'status', 'text_regions_count', 'processing_time', 'extracted_text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                # Combine all extracted text
                extracted_text = ""
                if "text_regions" in result and result["text_regions"]:
                    extracted_text = " | ".join([text for _, text in result["text_regions"]])

                writer.writerow({
                    'image_name': os.path.basename(result["image_path"]),
                    'status': result["status"],
                    'text_regions_count': result.get("num_text_regions", 0),
                    'processing_time': f"{result.get('processing_time', 0):.3f}",
                    'extracted_text': extracted_text[:500]  # Limit text length
                })

        self.logger.info(f"CSV report saved to: {csv_path}")

    def print_summary(self):
        """Print processing summary"""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        avg_time_per_image = self.stats["processing_time"] / max(self.stats["successful_processes"], 1)

        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total images found: {self.stats['total_images']}")
        print(f"Successfully processed: {self.stats['successful_processes']}")
        print(f"Failed to process: {self.stats['failed_processes']}")
        print(f"Success rate: {(self.stats['successful_processes']/max(self.stats['total_images'], 1)*100):.1f}%")
        print(f"Total text regions detected: {self.stats['total_text_regions']}")
        print(f"Average text regions per image: {(self.stats['total_text_regions']/max(self.stats['successful_processes'], 1)):.1f}")
        print(f"Total processing time: {self.stats['processing_time']:.2f} seconds")
        print(f"Average time per image: {avg_time_per_image:.2f} seconds")
        print(f"Total elapsed time: {duration:.2f} seconds")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Batch OCR and Text Detection')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Input directory containing images')
    parser.add_argument('--east', type=str, default='frozen_east_text_detection.pb',
                       help='Path to EAST text detector model')
    parser.add_argument('--output_dir', type=str, default='batch_results',
                       help='Output directory for results')
    parser.add_argument('--min_confidence', type=float, default=0.5,
                       help='Minimum confidence for text detection')
    parser.add_argument('--width', type=int, default=320,
                       help='Input width for EAST model (multiple of 32)')
    parser.add_argument('--height', type=int, default=320,
                       help='Input height for EAST model (multiple of 32)')

    args = parser.parse_args()

    # Setup logging
    logger, log_file = setup_logging()
    logger.info("Starting batch OCR processing")
    logger.info(f"Log file: {log_file}")

    # Validate inputs
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return

    if not os.path.exists(args.east):
        logger.error(f"EAST model file does not exist: {args.east}")
        logger.info("Please download the model first:")
        logger.info("  python download_scripts/download_model.py")
        return

    # Initialize batch processor
    processor = BatchOCRProcessor(
        east_model_path=args.east,
        output_dir=args.output_dir,
        min_confidence=args.min_confidence,
        width=args.width,
        height=args.height
    )

    # Process directory
    results = processor.process_directory(args.input_dir)

    if results:
        logger.info("Batch processing completed successfully!")
        logger.info(f"Check {args.output_dir} for results")
        logger.info(f"Log file saved to: {log_file}")
    else:
        logger.warning("No images were successfully processed")

if __name__ == "__main__":
    main()
