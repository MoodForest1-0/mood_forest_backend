"""
MoodForest DeepFace Evaluation Script

Evaluates the DeepFace emotion model using a labelled test dataset.

Expected dataset structure:

test_dataset/
    angry/
    fear/
    happy/
    neutral/
    sad/

Outputs:
    evaluation_results/
        predictions.csv
        failed_images.csv
        classification_report.csv
        classification_report.txt
        metrics_summary.txt
        metrics_summary.json
        confusion_matrix_values.csv
        confusion_matrix.png
        normalized_confusion_matrix.csv
        normalized_confusion_matrix.png

Run:
    python evaluate_deepface.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepface import DeepFace
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TEST_DATASET_DIR = Path("test_dataset")
OUTPUT_DIR = Path("evaluation_results")

EMOTION_LABELS = [
    "angry",
    "fear",
    "happy",
    "neutral",
    "sad",
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

DETECTOR_BACKEND = "mediapipe"

# Use True for strict evaluation.
# Images without a detectable face will be recorded as failed.
ENFORCE_DETECTION = True

# Set to an integer such as 20 for a quick test.
# Keep None to process all images.
MAX_IMAGES_PER_CLASS: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert NumPy and Python numeric values to a normal Python float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_emotion_scores(
    emotion_scores: dict[str, Any],
) -> dict[str, float]:
    """
    Keep the five MoodForest emotions and normalize them to 100%.

    DeepFace may also return surprise and disgust. Those classes are excluded
    because the deployed MoodForest system uses five emotions.
    """
    filtered_scores = {
        emotion: safe_float(emotion_scores.get(emotion, 0.0))
        for emotion in EMOTION_LABELS
    }

    total = sum(filtered_scores.values())

    if total <= 0:
        return {
            emotion: 0.0
            for emotion in EMOTION_LABELS
        }

    return {
        emotion: round((score / total) * 100.0, 4)
        for emotion, score in filtered_scores.items()
    }


def find_test_images() -> list[tuple[Path, str]]:
    """Find all labelled images in the test dataset."""
    samples: list[tuple[Path, str]] = []

    for emotion in EMOTION_LABELS:
        emotion_directory = TEST_DATASET_DIR / emotion

        if not emotion_directory.exists():
            print(
                f"WARNING: Folder not found: {emotion_directory}"
            )
            continue

        image_paths = sorted(
            path
            for path in emotion_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if MAX_IMAGES_PER_CLASS is not None:
            image_paths = image_paths[:MAX_IMAGES_PER_CLASS]

        samples.extend(
            (image_path, emotion)
            for image_path in image_paths
        )

    return samples


def extract_deepface_result(result: Any) -> dict[str, Any]:
    """
    DeepFace may return either a dictionary or a list containing one dictionary.
    """
    if isinstance(result, list):
        if not result:
            raise ValueError("DeepFace returned an empty result.")

        result = result[0]

    if not isinstance(result, dict):
        raise TypeError(
            f"Unexpected DeepFace result type: {type(result)}"
        )

    return result


def predict_emotion(
    image_path: Path,
) -> tuple[str, float, dict[str, float]]:
    """Run DeepFace and return prediction, confidence and five-class scores."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("OpenCV could not read the image.")

    result = DeepFace.analyze(
        img_path=image,
        actions=["emotion"],
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=ENFORCE_DETECTION,
        align=True,
        expand_percentage=10,
        silent=True,
    )

    result_dictionary = extract_deepface_result(result)

    raw_scores = result_dictionary.get("emotion")

    if not isinstance(raw_scores, dict):
        raise ValueError(
            "DeepFace result does not contain valid emotion scores."
        )

    normalized_scores = normalize_emotion_scores(raw_scores)

    predicted_emotion = max(
        normalized_scores,
        key=normalized_scores.get,
    )

    confidence = normalized_scores[predicted_emotion]

    return predicted_emotion, confidence, normalized_scores


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    normalized: bool = False,
) -> None:
    """Save a confusion matrix image using matplotlib."""
    figure, axis = plt.subplots(figsize=(9, 7))

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(EMOTION_LABELS)),
        yticks=np.arange(len(EMOTION_LABELS)),
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        xlabel="Predicted emotion",
        ylabel="Actual emotion",
        title=title,
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = (
        matrix.max() / 2.0
        if matrix.size and matrix.max() > 0
        else 0
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]

            display_value = (
                f"{value:.2f}"
                if normalized
                else str(int(value))
            )

            axis.text(
                column_index,
                row_index,
                display_value,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def calculate_specificity(
    matrix: np.ndarray,
) -> dict[str, float]:
    """Calculate one-vs-rest specificity for each emotion."""
    specificity_values: dict[str, float] = {}

    total = matrix.sum()

    for index, emotion in enumerate(EMOTION_LABELS):
        true_positive = matrix[index, index]
        false_negative = matrix[index, :].sum() - true_positive
        false_positive = matrix[:, index].sum() - true_positive

        true_negative = (
            total
            - true_positive
            - false_negative
            - false_positive
        )

        denominator = true_negative + false_positive

        specificity = (
            true_negative / denominator
            if denominator > 0
            else 0.0
        )

        specificity_values[emotion] = float(specificity)

    return specificity_values


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_DATASET_DIR.exists():
        print(
            f"ERROR: Test dataset folder not found: "
            f"{TEST_DATASET_DIR.resolve()}"
        )
        sys.exit(1)

    samples = find_test_images()

    if not samples:
        print("ERROR: No valid test images were found.")
        sys.exit(1)

    print("=" * 80)
    print("MOODFOREST DEEPFACE MODEL EVALUATION")
    print("=" * 80)
    print(f"Dataset directory : {TEST_DATASET_DIR.resolve()}")
    print(f"Output directory  : {OUTPUT_DIR.resolve()}")
    print(f"Detector backend  : {DETECTOR_BACKEND}")
    print(f"Enforce detection : {ENFORCE_DETECTION}")
    print(f"Images found      : {len(samples)}")
    print(f"Emotion classes   : {', '.join(EMOTION_LABELS)}")
    print("=" * 80)

    y_true: list[str] = []
    y_pred: list[str] = []

    prediction_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, str]] = []

    start_time = time.time()

    for image_number, (image_path, actual_emotion) in enumerate(
        samples,
        start=1,
    ):
        image_start_time = time.time()

        try:
            predicted_emotion, confidence, scores = predict_emotion(
                image_path
            )

            processing_time = time.time() - image_start_time

            y_true.append(actual_emotion)
            y_pred.append(predicted_emotion)

            prediction_rows.append({
                "image": str(image_path),
                "filename": image_path.name,
                "actual_emotion": actual_emotion,
                "predicted_emotion": predicted_emotion,
                "correct": actual_emotion == predicted_emotion,
                "confidence": round(confidence, 4),
                "angry_score": scores["angry"],
                "fear_score": scores["fear"],
                "happy_score": scores["happy"],
                "neutral_score": scores["neutral"],
                "sad_score": scores["sad"],
                "processing_time_seconds": round(
                    processing_time,
                    4,
                ),
            })

            status = (
                "CORRECT"
                if actual_emotion == predicted_emotion
                else "WRONG"
            )

            print(
                f"[{image_number:04d}/{len(samples):04d}] "
                f"Actual: {actual_emotion:<7} | "
                f"Predicted: {predicted_emotion:<7} | "
                f"Confidence: {confidence:6.2f}% | "
                f"{status}"
            )

        except Exception as error:
            failed_rows.append({
                "image": str(image_path),
                "filename": image_path.name,
                "actual_emotion": actual_emotion,
                "error": str(error),
            })

            print(
                f"[{image_number:04d}/{len(samples):04d}] "
                f"FAILED: {image_path.name} | {error}"
            )

    total_time = time.time() - start_time

    if not y_true:
        print(
            "ERROR: No images were successfully evaluated."
        )
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────────────────────

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=EMOTION_LABELS,
            average="macro",
            zero_division=0,
        )
    )

    (
        precision_weighted,
        recall_weighted,
        f1_weighted,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=EMOTION_LABELS,
        average="weighted",
        zero_division=0,
    )

    report_dictionary = classification_report(
        y_true,
        y_pred,
        labels=EMOTION_LABELS,
        target_names=EMOTION_LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=EMOTION_LABELS,
        target_names=EMOTION_LABELS,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=EMOTION_LABELS,
    )

    normalized_matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=EMOTION_LABELS,
        normalize="true",
    )

    specificity = calculate_specificity(matrix)

    successful_images = len(y_true)
    failed_images = len(failed_rows)
    correct_predictions = int(
        sum(
            actual == predicted
            for actual, predicted in zip(y_true, y_pred)
        )
    )

    average_processing_time = (
        total_time / len(samples)
        if samples
        else 0.0
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE PREDICTIONS
    # ─────────────────────────────────────────────────────────────────────────

    predictions_dataframe = pd.DataFrame(prediction_rows)
    predictions_dataframe.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
    )

    failed_dataframe = pd.DataFrame(
        failed_rows,
        columns=[
            "image",
            "filename",
            "actual_emotion",
            "error",
        ],
    )

    failed_dataframe.to_csv(
        OUTPUT_DIR / "failed_images.csv",
        index=False,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE CLASSIFICATION REPORT
    # ─────────────────────────────────────────────────────────────────────────

    report_dataframe = (
        pd.DataFrame(report_dictionary)
        .transpose()
        .reset_index()
        .rename(columns={"index": "class"})
    )

    report_dataframe.to_csv(
        OUTPUT_DIR / "classification_report.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as report_file:
        report_file.write(report_text)

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE CONFUSION MATRIX VALUES
    # ─────────────────────────────────────────────────────────────────────────

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[
            f"actual_{emotion}"
            for emotion in EMOTION_LABELS
        ],
        columns=[
            f"predicted_{emotion}"
            for emotion in EMOTION_LABELS
        ],
    )

    matrix_dataframe.to_csv(
        OUTPUT_DIR / "confusion_matrix_values.csv"
    )

    normalized_matrix_dataframe = pd.DataFrame(
        normalized_matrix,
        index=[
            f"actual_{emotion}"
            for emotion in EMOTION_LABELS
        ],
        columns=[
            f"predicted_{emotion}"
            for emotion in EMOTION_LABELS
        ],
    )

    normalized_matrix_dataframe.to_csv(
        OUTPUT_DIR / "normalized_confusion_matrix.csv"
    )

    save_confusion_matrix_plot(
        matrix=matrix,
        output_path=OUTPUT_DIR / "confusion_matrix.png",
        title="DeepFace Emotion Recognition Confusion Matrix",
        normalized=False,
    )

    save_confusion_matrix_plot(
        matrix=normalized_matrix,
        output_path=OUTPUT_DIR / "normalized_confusion_matrix.png",
        title="Normalized DeepFace Emotion Confusion Matrix",
        normalized=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PER-CLASS SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    per_class_metrics: dict[str, dict[str, float | int]] = {}

    for emotion in EMOTION_LABELS:
        class_report = report_dictionary.get(emotion, {})

        per_class_metrics[emotion] = {
            "precision": round(
                safe_float(class_report.get("precision")),
                4,
            ),
            "recall": round(
                safe_float(class_report.get("recall")),
                4,
            ),
            "f1_score": round(
                safe_float(class_report.get("f1-score")),
                4,
            ),
            "specificity": round(
                specificity.get(emotion, 0.0),
                4,
            ),
            "support": int(
                safe_float(class_report.get("support"))
            ),
        }

    metrics_summary = {
        "dataset": str(TEST_DATASET_DIR),
        "labels": EMOTION_LABELS,
        "total_images_found": len(samples),
        "successful_images": successful_images,
        "failed_images": failed_images,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": (
            successful_images - correct_predictions
        ),
        "accuracy": round(float(accuracy), 4),
        "accuracy_percentage": round(
            float(accuracy * 100),
            2,
        ),
        "macro_precision": round(
            float(precision_macro),
            4,
        ),
        "macro_recall": round(
            float(recall_macro),
            4,
        ),
        "macro_f1_score": round(
            float(f1_macro),
            4,
        ),
        "weighted_precision": round(
            float(precision_weighted),
            4,
        ),
        "weighted_recall": round(
            float(recall_weighted),
            4,
        ),
        "weighted_f1_score": round(
            float(f1_weighted),
            4,
        ),
        "total_processing_time_seconds": round(
            total_time,
            2,
        ),
        "average_processing_time_seconds": round(
            average_processing_time,
            4,
        ),
        "detector_backend": DETECTOR_BACKEND,
        "enforce_detection": ENFORCE_DETECTION,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": (
            normalized_matrix.round(4).tolist()
        ),
    }

    with open(
        OUTPUT_DIR / "metrics_summary.json",
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            metrics_summary,
            json_file,
            indent=4,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE READABLE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    summary_lines = [
        "=" * 80,
        "MOODFOREST DEEPFACE EVALUATION SUMMARY",
        "=" * 80,
        f"Total images found       : {len(samples)}",
        f"Successfully evaluated  : {successful_images}",
        f"Failed images           : {failed_images}",
        f"Correct predictions     : {correct_predictions}",
        (
            "Incorrect predictions   : "
            f"{successful_images - correct_predictions}"
        ),
        "",
        f"Accuracy                 : {accuracy:.4f}",
        f"Accuracy percentage      : {accuracy * 100:.2f}%",
        "",
        f"Macro precision          : {precision_macro:.4f}",
        f"Macro recall             : {recall_macro:.4f}",
        f"Macro F1-score           : {f1_macro:.4f}",
        "",
        f"Weighted precision       : {precision_weighted:.4f}",
        f"Weighted recall          : {recall_weighted:.4f}",
        f"Weighted F1-score        : {f1_weighted:.4f}",
        "",
        (
            "Total processing time    : "
            f"{total_time:.2f} seconds"
        ),
        (
            "Average time per image   : "
            f"{average_processing_time:.4f} seconds"
        ),
        "",
        "PER-CLASS METRICS",
        "-" * 80,
    ]

    for emotion in EMOTION_LABELS:
        values = per_class_metrics[emotion]

        summary_lines.extend([
            f"{emotion.upper()}:",
            f"  Precision   : {values['precision']:.4f}",
            f"  Recall      : {values['recall']:.4f}",
            f"  F1-score    : {values['f1_score']:.4f}",
            f"  Specificity : {values['specificity']:.4f}",
            f"  Support     : {values['support']}",
            "",
        ])

    summary_lines.extend([
        "CONFUSION MATRIX",
        "-" * 80,
        "Rows = actual labels",
        "Columns = predicted labels",
        "",
        matrix_dataframe.to_string(),
        "",
        "CLASSIFICATION REPORT",
        "-" * 80,
        report_text,
    ])

    summary_text = "\n".join(summary_lines)

    with open(
        OUTPUT_DIR / "metrics_summary.txt",
        "w",
        encoding="utf-8",
    ) as summary_file:
        summary_file.write(summary_text)

    # ─────────────────────────────────────────────────────────────────────────
    # PRINT FINAL RESULTS
    # ─────────────────────────────────────────────────────────────────────────

    print()
    print(summary_text)

    print("=" * 80)
    print("FILES CREATED")
    print("=" * 80)
    print(OUTPUT_DIR / "predictions.csv")
    print(OUTPUT_DIR / "failed_images.csv")
    print(OUTPUT_DIR / "classification_report.csv")
    print(OUTPUT_DIR / "classification_report.txt")
    print(OUTPUT_DIR / "metrics_summary.txt")
    print(OUTPUT_DIR / "metrics_summary.json")
    print(OUTPUT_DIR / "confusion_matrix_values.csv")
    print(OUTPUT_DIR / "confusion_matrix.png")
    print(OUTPUT_DIR / "normalized_confusion_matrix.csv")
    print(OUTPUT_DIR / "normalized_confusion_matrix.png")


if __name__ == "__main__":
    main()
