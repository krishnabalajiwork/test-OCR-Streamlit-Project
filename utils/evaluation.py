#!/usr/bin/env python3
"""
Text Detection Evaluation Utility

This script evaluates the performance of text detection models using
standard metrics like Precision, Recall, F1-Score, and IoU-based matching.
"""

import os
import sys
import argparse
import json
import csv
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from pathlib import Path

class TextDetectionEvaluator:
    """
    Evaluator for text detection performance using IoU-based metrics
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def calculate_iou(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes

        Args:
            box1: (x1, y1, x2, y2) format
            box2: (x1, y1, x2, y2) format

        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection area
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def match_boxes(self, pred_boxes: List[Tuple], 
                    gt_boxes: List[Tuple]) -> Tuple[List[bool], List[bool]]:
        """
        Match predicted boxes with ground truth boxes using IoU threshold

        Args:
            pred_boxes: List of predicted bounding boxes
            gt_boxes: List of ground truth bounding boxes

        Returns:
            Tuple of (pred_matched, gt_matched) boolean lists
        """
        pred_matched = [False] * len(pred_boxes)
        gt_matched = [False] * len(gt_boxes)

        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1

            for j, gt_box in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue

                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_gt_idx = j

            if best_gt_idx != -1:
                pred_matched[i] = True
                gt_matched[best_gt_idx] = True

        return pred_matched, gt_matched

    def evaluate_single_image(self, pred_boxes: List[Tuple], 
                             gt_boxes: List[Tuple]) -> Dict[str, float]:
        """
        Evaluate text detection for a single image

        Args:
            pred_boxes: List of predicted bounding boxes
            gt_boxes: List of ground truth bounding boxes

        Returns:
            Dictionary with precision, recall, f1-score, and counts
        """
        if not pred_boxes and not gt_boxes:
            return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, 
                   "tp": 0, "fp": 0, "fn": 0}

        if not pred_boxes:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                   "tp": 0, "fp": 0, "fn": len(gt_boxes)}

        if not gt_boxes:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                   "tp": 0, "fp": len(pred_boxes), "fn": 0}

        pred_matched, gt_matched = self.match_boxes(pred_boxes, gt_boxes)

        tp = sum(pred_matched)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - sum(gt_matched)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

    def evaluate_dataset(self, predictions: Dict[str, List[Tuple]], 
                        ground_truths: Dict[str, List[Tuple]]) -> Dict[str, float]:
        """
        Evaluate text detection for an entire dataset

        Args:
            predictions: Dictionary mapping image names to predicted boxes
            ground_truths: Dictionary mapping image names to ground truth boxes

        Returns:
            Overall evaluation metrics
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0

        image_results = {}

        for image_name in ground_truths.keys():
            pred_boxes = predictions.get(image_name, [])
            gt_boxes = ground_truths[image_name]

            result = self.evaluate_single_image(pred_boxes, gt_boxes)
            image_results[image_name] = result

            total_tp += result["tp"]
            total_fp += result["fp"]
            total_fn += result["fn"]

        # Calculate overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

        return {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1_score": overall_f1,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "num_images": len(ground_truths),
            "image_results": image_results
        }

def load_ground_truth_annotations(annotation_file: str) -> Dict[str, List[Tuple]]:
    """
    Load ground truth annotations from a JSON file

    Expected format:
    {
        "image1.jpg": [
            {"bbox": [x1, y1, x2, y2], "text": "sample text"},
            ...
        ],
        ...
    }
    """
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    ground_truths = {}
    for image_name, boxes in annotations.items():
        ground_truths[image_name] = [tuple(box["bbox"]) for box in boxes]

    return ground_truths

def load_predictions_from_batch_results(batch_results_dir: str) -> Dict[str, List[Tuple]]:
    """
    Load predictions from batch processing results
    """
    predictions = {}

    # Load from batch summary JSON
    summary_path = os.path.join(batch_results_dir, "batch_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        for result in summary.get("individual_results", []):
            image_name = os.path.basename(result["image_path"])
            # For now, we'll use dummy boxes - in practice, you'd load the actual coordinates
            # This would need to be implemented based on your batch results format
            predictions[image_name] = []

    return predictions

def save_evaluation_results(results: Dict, output_file: str):
    """
    Save evaluation results to a JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def save_evaluation_csv(results: Dict, csv_file: str):
    """
    Save per-image evaluation results to CSV
    """
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'precision', 'recall', 'f1_score', 'tp', 'fp', 'fn']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for image_name, metrics in results['image_results'].items():
            row = {'image_name': image_name}
            row.update(metrics)
            writer.writerow(row)

def plot_evaluation_metrics(results: Dict, output_dir: str):
    """
    Create visualization plots for evaluation metrics
    """
    try:
        import matplotlib.pyplot as plt

        # Extract per-image metrics
        image_names = list(results['image_results'].keys())
        precisions = [results['image_results'][name]['precision'] for name in image_names]
        recalls = [results['image_results'][name]['recall'] for name in image_names]
        f1_scores = [results['image_results'][name]['f1_score'] for name in image_names]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Text Detection Evaluation Results', fontsize=16)

        # Precision distribution
        axes[0, 0].hist(precisions, bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Precision Distribution')
        axes[0, 0].set_xlabel('Precision')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(results['precision'], color='red', linestyle='--', 
                          label=f'Overall: {results["precision"]:.3f}')
        axes[0, 0].legend()

        # Recall distribution
        axes[0, 1].hist(recalls, bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('Recall Distribution')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(results['recall'], color='red', linestyle='--', 
                          label=f'Overall: {results["recall"]:.3f}')
        axes[0, 1].legend()

        # F1-Score distribution
        axes[1, 0].hist(f1_scores, bins=20, alpha=0.7, color='orange')
        axes[1, 0].set_title('F1-Score Distribution')
        axes[1, 0].set_xlabel('F1-Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(results['f1_score'], color='red', linestyle='--', 
                          label=f'Overall: {results["f1_score"]:.3f}')
        axes[1, 0].legend()

        # Precision vs Recall scatter
        axes[1, 1].scatter(recalls, precisions, alpha=0.6)
        axes[1, 1].set_title('Precision vs Recall')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Evaluation plots saved to: {os.path.join(output_dir, 'evaluation_plots.png')}")

    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")

def create_evaluation_report(results: Dict, output_dir: str):
    """
    Create a comprehensive evaluation report
    """
    report_path = os.path.join(output_dir, 'evaluation_report.txt')

    with open(report_path, 'w') as f:
        f.write("TEXT DETECTION EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("OVERALL METRICS:\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1-Score: {results['f1_score']:.4f}\n\n")

        f.write("CONFUSION MATRIX:\n")
        f.write(f"True Positives: {results['total_tp']}\n")
        f.write(f"False Positives: {results['total_fp']}\n")
        f.write(f"False Negatives: {results['total_fn']}\n\n")

        f.write("DATASET STATISTICS:\n")
        f.write(f"Number of Images: {results['num_images']}\n")
        f.write(f"Total Ground Truth Boxes: {results['total_tp'] + results['total_fn']}\n")
        f.write(f"Total Predicted Boxes: {results['total_tp'] + results['total_fp']}\n\n")

        # Per-image breakdown
        f.write("PER-IMAGE RESULTS:\n")
        f.write("-" * 30 + "\n")
        for image_name, metrics in results['image_results'].items():
            f.write(f"{image_name}:\n")
            f.write(f"  Precision: {metrics['precision']:.3f}\n")
            f.write(f"  Recall: {metrics['recall']:.3f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.3f}\n")
            f.write(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}\n\n")

    print(f"Evaluation report saved to: {report_path}")

def main():
    """
    Example usage of the evaluation system
    """
    parser = argparse.ArgumentParser(description='Evaluate text detection performance')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSON file or batch results directory')
    parser.add_argument('--ground_truth', type=str, required=True,
                       help='Path to ground truth annotations JSON file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for matching boxes')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load ground truth
    print("Loading ground truth annotations...")
    try:
        ground_truths = load_ground_truth_annotations(args.ground_truth)
        print(f"Loaded ground truth for {len(ground_truths)} images")
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return

    # Load predictions
    print("Loading predictions...")
    try:
        if os.path.isdir(args.predictions):
            predictions = load_predictions_from_batch_results(args.predictions)
        else:
            with open(args.predictions, 'r') as f:
                predictions = json.load(f)
        print(f"Loaded predictions for {len(predictions)} images")
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return

    # Initialize evaluator
    evaluator = TextDetectionEvaluator(iou_threshold=args.iou_threshold)

    # Evaluate dataset
    print("Evaluating text detection performance...")
    results = evaluator.evaluate_dataset(predictions, ground_truths)

    # Print results
    print("\nEvaluation Results:")
    print("=" * 30)
    print(f"Overall Precision: {results['precision']:.3f}")
    print(f"Overall Recall: {results['recall']:.3f}")
    print(f"Overall F1-Score: {results['f1_score']:.3f}")
    print(f"True Positives: {results['total_tp']}")
    print(f"False Positives: {results['total_fp']}")
    print(f"False Negatives: {results['total_fn']}")

    # Save results
    save_evaluation_results(results, os.path.join(args.output_dir, 'evaluation_metrics.json'))
    save_evaluation_csv(results, os.path.join(args.output_dir, 'per_image_metrics.csv'))
    create_evaluation_report(results, args.output_dir)
    plot_evaluation_metrics(results, args.output_dir)

    print(f"\nAll results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
