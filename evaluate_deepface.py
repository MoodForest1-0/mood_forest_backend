"""
MoodForest DeepFace Evaluation Script (v2 — Quality-Aware + TTA)

Evaluates the DeepFace emotion model using a labelled test dataset, with:
    - CLAHE preprocessing (contrast enhancement)
    - Test-Time Augmentation (original, horizontal flip, brighter, darker, CLAHE)
    - Emotion score averaging across all successful augmentations
    - Automatic face-quality filtering (blur, brightness, face size)
    - Confidence thresholding (flags low-confidence predictions)
    - Separate CSV for skipped / low-quality images
    - Confusion matrix + classification report (unchanged output format)
    - Modular, commented code, compatible with the existing app.py fusion logic

Expected dataset structure:

test_dataset/
    angry/
    fear/
    happy/
    neutral/
    sad/

Outputs (in evaluation_results/):
    predictions.csv                     -> every image that was scored
    skipped_images.csv                  -> images rejected by quality filter
    failed_images.csv                   -> images DeepFace could not process at all
    classification_report.csv / .txt
    metrics_summary.txt / .json
    confusion_matrix_values.csv / .png
    normalized_confusion_matrix.csv / .png

Run:
    python evaluate_deepface.py
"""

from __future__ import annotations

import json
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

EMOTION_LABELS = ["angry", "fear", "happy", "neutral", "sad"]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DETECTOR_BACKEND = "opencv"

# Use True for strict evaluation. Images without a detectable face on the
# ORIGINAL frame are recorded as failed. Augmented variants fall back to
# enforce_detection=False so a borderline crop doesn't kill the whole sample.
ENFORCE_DETECTION = True

# Set to an integer such as 20 for a quick test. Keep None to process all images.
MAX_IMAGES_PER_CLASS: int | None = None

# ─── Face-quality filtering thresholds ────────────────────────────────────
# Images that fail these checks are skipped BEFORE any DeepFace call and
# logged to skipped_images.csv instead of being scored / counted as failures.
MIN_BLUR_VARIANCE = 60.0        # Laplacian variance; lower = blurrier
MIN_BRIGHTNESS = 40.0           # mean grayscale intensity (0-255)
MAX_BRIGHTNESS = 220.0
MIN_FACE_FRACTION = 0.06        # face bbox area / image area, minimum
QUALITY_DETECTOR_BACKEND = "opencv"  # backend used purely for the quality pre-check

# ─── Confidence thresholding ───────────────────────────────────────────────
# Predictions below this are still counted (so accuracy stays comparable to
# the old script) but are flagged so you can see how much of the accuracy
# gap is driven by genuinely uncertain calls.
LOW_CONFIDENCE_THRESHOLD = 40.0  # percent

# ─── Test-Time Augmentation ─────────────────────────────────────────────────
# Each entry: (name, function(image) -> augmented image)
BRIGHTER_BETA = 30
DARKER_BETA = -30


# ─────────────────────────────────────────────────────────────────────────────
# GENERAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert NumPy / Python numeric values to a normal Python float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_emotion_scores(emotion_scores: dict[str, Any]) -> dict[str, float]:
    """Keep the five MoodForest emotions and normalize them to 100%."""
    filtered_scores = {
        emotion: safe_float(emotion_scores.get(emotion, 0.0))
        for emotion in EMOTION_LABELS
    }

    total = sum(filtered_scores.values())

    if total <= 0:
        return {emotion: 0.0 for emotion in EMOTION_LABELS}

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
            print(f"WARNING: Folder not found: {emotion_directory}")
            continue

        image_paths = sorted(
            path
            for path in emotion_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if MAX_IMAGES_PER_CLASS is not None:
            image_paths = image_paths[:MAX_IMAGES_PER_CLASS]

        samples.extend((image_path, emotion) for image_path in image_paths)

    return samples


def extract_deepface_result(result: Any) -> dict[str, Any]:
    """DeepFace may return either a dict or a list containing one dict."""
    if isinstance(result, list):
        if not result:
            raise ValueError("DeepFace returned an empty result.")
        result = result[0]

    if not isinstance(result, dict):
        raise TypeError(f"Unexpected DeepFace result type: {type(result)}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FACE-QUALITY FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def compute_blur_variance(gray_image: np.ndarray) -> float:
    """Higher = sharper. Values below ~50-100 usually mean a blurry crop."""
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def compute_brightness(gray_image: np.ndarray) -> float:
    """Mean pixel intensity, 0 (black) - 255 (white)."""
    return float(np.mean(gray_image))


def assess_face_quality(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Detect the face and check blur / brightness / relative face size.

    Returns a dict with the raw metrics plus a boolean 'passed' flag and a
    'reason' string describing the first failed check (if any).
    """
    quality: dict[str, Any] = {
        "face_detected": False,
        "blur_variance": 0.0,
        "brightness": 0.0,
        "face_fraction": 0.0,
        "passed": False,
        "reason": "",
    }

    try:
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend=QUALITY_DETECTOR_BACKEND,
            enforce_detection=True,
            align=False,
        )
    except Exception as error:
        quality["reason"] = f"no_face_detected ({error})"
        return quality

    if not faces:
        quality["reason"] = "no_face_detected"
        return quality

    # Use the largest detected face region for the quality checks.
    face_info = max(
        faces,
        key=lambda f: f.get("facial_area", {}).get("w", 0)
        * f.get("facial_area", {}).get("h", 0),
    )

    facial_area = face_info.get("facial_area", {})
    face_w = safe_float(facial_area.get("w"))
    face_h = safe_float(facial_area.get("h"))

    image_h, image_w = image.shape[:2]
    image_area = max(image_w * image_h, 1)
    face_fraction = (face_w * face_h) / image_area

    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_variance = compute_blur_variance(gray_full)
    brightness = compute_brightness(gray_full)

    quality.update({
        "face_detected": True,
        "blur_variance": round(blur_variance, 2),
        "brightness": round(brightness, 2),
        "face_fraction": round(face_fraction, 4),
    })

    if blur_variance < MIN_BLUR_VARIANCE:
        quality["reason"] = "too_blurry"
        return quality

    if not (MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS):
        quality["reason"] = "bad_brightness"
        return quality

    if face_fraction < MIN_FACE_FRACTION:
        quality["reason"] = "face_too_small"
        return quality

    quality["passed"] = True
    quality["reason"] = "ok"
    return quality


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING / TEST-TIME AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization on the L channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_equalized = clahe.apply(l_channel)

    merged = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def adjust_brightness(image: np.ndarray, beta: int) -> np.ndarray:
    """Shift brightness up (positive beta) or down (negative beta)."""
    return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)


def build_tta_variants(image: np.ndarray) -> dict[str, np.ndarray]:
    """
    Build the set of test-time-augmentation variants for one image.
    Every variant is run through DeepFace and the resulting scores are
    averaged, which smooths out lighting / pose noise that a single pass
    is sensitive to.
    """
    return {
        "original": image,
        "flipped": cv2.flip(image, 1),
        "brighter": adjust_brightness(image, BRIGHTER_BETA),
        "darker": adjust_brightness(image, DARKER_BETA),
        "clahe": apply_clahe(image),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION (TTA-AVERAGED)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_single_variant(variant_image: np.ndarray) -> dict[str, float] | None:
    """Run DeepFace on one augmented image. Returns normalized scores or None."""
    try:
        result = DeepFace.analyze(
            img_path=variant_image,
            actions=["emotion"],
            detector_backend=DETECTOR_BACKEND,
            # Augmented crops (flip/brightness/CLAHE) can occasionally trip up
            # the detector even though the original face was already
            # confirmed by assess_face_quality(), so we don't re-enforce here.
            enforce_detection=False,
            align=True,
            expand_percentage=10,
            silent=True,
        )
        result_dictionary = extract_deepface_result(result)
        raw_scores = result_dictionary.get("emotion")

        if not isinstance(raw_scores, dict):
            return None

        return normalize_emotion_scores(raw_scores)

    except Exception:
        return None


def predict_emotion_tta(
    image: np.ndarray,
) -> tuple[str, float, dict[str, float], int, list[str]]:
    """
    Run DeepFace across all TTA variants and average the resulting scores.

    Returns:
        predicted_emotion, confidence, averaged_scores,
        variants_used_count, list_of_variant_names_used
    """
    variants = build_tta_variants(image)

    accumulated = {emotion: 0.0 for emotion in EMOTION_LABELS}
    used_variants: list[str] = []

    for variant_name, variant_image in variants.items():
        scores = analyze_single_variant(variant_image)

        if scores is None:
            continue

        for emotion in EMOTION_LABELS:
            accumulated[emotion] += scores[emotion]

        used_variants.append(variant_name)

    if not used_variants:
        raise ValueError("All TTA variants failed DeepFace analysis.")

    averaged_scores = {
        emotion: round(accumulated[emotion] / len(used_variants), 4)
        for emotion in EMOTION_LABELS
    }

    predicted_emotion = max(averaged_scores, key=averaged_scores.get)
    confidence = averaged_scores[predicted_emotion]

    return predicted_emotion, confidence, averaged_scores, len(used_variants), used_variants


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX / METRICS HELPERS (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix_plot(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    normalized: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))

    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
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

    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 0

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            display_value = f"{value:.2f}" if normalized else str(int(value))

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


def calculate_specificity(matrix: np.ndarray) -> dict[str, float]:
    specificity_values: dict[str, float] = {}
    total = matrix.sum()

    for index, emotion in enumerate(EMOTION_LABELS):
        true_positive = matrix[index, index]
        false_negative = matrix[index, :].sum() - true_positive
        false_positive = matrix[:, index].sum() - true_positive
        true_negative = total - true_positive - false_negative - false_positive

        denominator = true_negative + false_positive
        specificity = true_negative / denominator if denominator > 0 else 0.0
        specificity_values[emotion] = float(specificity)

    return specificity_values


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_DATASET_DIR.exists():
        print(f"ERROR: Test dataset folder not found: {TEST_DATASET_DIR.resolve()}")
        sys.exit(1)

    samples = find_test_images()

    if not samples:
        print("ERROR: No valid test images were found.")
        sys.exit(1)

    print("=" * 80)
    print("MOODFOREST DEEPFACE MODEL EVALUATION (Quality-Aware + TTA)")
    print("=" * 80)
    print(f"Dataset directory   : {TEST_DATASET_DIR.resolve()}")
    print(f"Output directory    : {OUTPUT_DIR.resolve()}")
    print(f"Detector backend    : {DETECTOR_BACKEND}")
    print(f"Enforce detection   : {ENFORCE_DETECTION}")
    print(f"TTA variants        : original, flipped, brighter, darker, clahe")
    print(f"Quality thresholds  : blur>={MIN_BLUR_VARIANCE}, "
          f"brightness in [{MIN_BRIGHTNESS},{MAX_BRIGHTNESS}], "
          f"face_fraction>={MIN_FACE_FRACTION}")
    print(f"Confidence flag     : < {LOW_CONFIDENCE_THRESHOLD}%")
    print(f"Images found        : {len(samples)}")
    print(f"Emotion classes     : {', '.join(EMOTION_LABELS)}")
    print("=" * 80)

    y_true: list[str] = []
    y_pred: list[str] = []

    prediction_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, str]] = []

    start_time = time.time()

    for image_number, (image_path, actual_emotion) in enumerate(samples, start=1):
        image_start_time = time.time()

        image = cv2.imread(str(image_path))

        if image is None:
            failed_rows.append({
                "image": str(image_path),
                "filename": image_path.name,
                "actual_emotion": actual_emotion,
                "error": "OpenCV could not read the image.",
            })
            print(f"[{image_number:04d}/{len(samples):04d}] FAILED (unreadable): {image_path.name}")
            continue

        # ── Step 1: quality gate ────────────────────────────────────────
        quality = assess_face_quality(image)

        if not quality["passed"]:
            skipped_rows.append({
                "image": str(image_path),
                "filename": image_path.name,
                "actual_emotion": actual_emotion,
                "reason": quality["reason"],
                "face_detected": quality["face_detected"],
                "blur_variance": quality["blur_variance"],
                "brightness": quality["brightness"],
                "face_fraction": quality["face_fraction"],
            })
            print(
                f"[{image_number:04d}/{len(samples):04d}] "
                f"SKIPPED ({quality['reason']}): {image_path.name}"
            )
            continue

        # ── Step 2: TTA-averaged prediction ─────────────────────────────
        try:
            predicted_emotion, confidence, scores, variants_used, variant_names = (
                predict_emotion_tta(image)
            )

            processing_time = time.time() - image_start_time
            is_low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

            y_true.append(actual_emotion)
            y_pred.append(predicted_emotion)

            prediction_rows.append({
                "image": str(image_path),
                "filename": image_path.name,
                "actual_emotion": actual_emotion,
                "predicted_emotion": predicted_emotion,
                "correct": actual_emotion == predicted_emotion,
                "confidence": round(confidence, 4),
                "low_confidence": is_low_confidence,
                "angry_score": scores["angry"],
                "fear_score": scores["fear"],
                "happy_score": scores["happy"],
                "neutral_score": scores["neutral"],
                "sad_score": scores["sad"],
                "tta_variants_used": variants_used,
                "tta_variant_names": "|".join(variant_names),
                "blur_variance": quality["blur_variance"],
                "brightness": quality["brightness"],
                "face_fraction": quality["face_fraction"],
                "processing_time_seconds": round(processing_time, 4),
            })

            status = "CORRECT" if actual_emotion == predicted_emotion else "WRONG"
            flag = " (LOW-CONF)" if is_low_confidence else ""

            print(
                f"[{image_number:04d}/{len(samples):04d}] "
                f"Actual: {actual_emotion:<7} | "
                f"Predicted: {predicted_emotion:<7} | "
                f"Confidence: {confidence:6.2f}% | "
                f"TTA: {variants_used}/5 | "
                f"{status}{flag}"
            )

        except Exception as error:
            failed_rows.append({
                "image": str(image_path),
                "filename": image_path.name,
                "actual_emotion": actual_emotion,
                "error": str(error),
            })
            print(f"[{image_number:04d}/{len(samples):04d}] FAILED: {image_path.name} | {error}")

    total_time = time.time() - start_time

    if not y_true:
        print("ERROR: No images were successfully evaluated.")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────
    # METRICS (all images that were actually scored, low-confidence included)
    # ─────────────────────────────────────────────────────────────────────

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=EMOTION_LABELS, average="macro", zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=EMOTION_LABELS, average="weighted", zero_division=0
    )

    report_dictionary = classification_report(
        y_true, y_pred, labels=EMOTION_LABELS, target_names=EMOTION_LABELS,
        output_dict=True, zero_division=0,
    )

    report_text = classification_report(
        y_true, y_pred, labels=EMOTION_LABELS, target_names=EMOTION_LABELS,
        digits=4, zero_division=0,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=EMOTION_LABELS)
    normalized_matrix = confusion_matrix(
        y_true, y_pred, labels=EMOTION_LABELS, normalize="true"
    )

    specificity = calculate_specificity(matrix)

    # High-confidence-only accuracy, for comparison
    high_conf_rows = [row for row in prediction_rows if not row["low_confidence"]]
    if high_conf_rows:
        high_conf_accuracy = sum(
            row["correct"] for row in high_conf_rows
        ) / len(high_conf_rows)
    else:
        high_conf_accuracy = 0.0

    total_found = len(samples)
    scored_images = len(y_true)
    skipped_images = len(skipped_rows)
    failed_images = len(failed_rows)
    correct_predictions = int(sum(a == p for a, p in zip(y_true, y_pred)))
    low_confidence_count = sum(row["low_confidence"] for row in prediction_rows)

    average_processing_time = total_time / total_found if total_found else 0.0

    # ─────────────────────────────────────────────────────────────────────
    # SAVE CSVs
    # ─────────────────────────────────────────────────────────────────────

    pd.DataFrame(prediction_rows).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    pd.DataFrame(
        skipped_rows,
        columns=[
            "image", "filename", "actual_emotion", "reason",
            "face_detected", "blur_variance", "brightness", "face_fraction",
        ],
    ).to_csv(OUTPUT_DIR / "skipped_images.csv", index=False)

    pd.DataFrame(
        failed_rows,
        columns=["image", "filename", "actual_emotion", "error"],
    ).to_csv(OUTPUT_DIR / "failed_images.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────
    # SAVE CLASSIFICATION REPORT
    # ─────────────────────────────────────────────────────────────────────

    report_dataframe = (
        pd.DataFrame(report_dictionary).transpose().reset_index().rename(columns={"index": "class"})
    )
    report_dataframe.to_csv(OUTPUT_DIR / "classification_report.csv", index=False)

    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # ─────────────────────────────────────────────────────────────────────
    # SAVE CONFUSION MATRICES
    # ─────────────────────────────────────────────────────────────────────

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[f"actual_{e}" for e in EMOTION_LABELS],
        columns=[f"predicted_{e}" for e in EMOTION_LABELS],
    )
    matrix_dataframe.to_csv(OUTPUT_DIR / "confusion_matrix_values.csv")

    normalized_matrix_dataframe = pd.DataFrame(
        normalized_matrix,
        index=[f"actual_{e}" for e in EMOTION_LABELS],
        columns=[f"predicted_{e}" for e in EMOTION_LABELS],
    )
    normalized_matrix_dataframe.to_csv(OUTPUT_DIR / "normalized_confusion_matrix.csv")

    save_confusion_matrix_plot(
        matrix=matrix,
        output_path=OUTPUT_DIR / "confusion_matrix.png",
        title="DeepFace Emotion Recognition Confusion Matrix (TTA)",
        normalized=False,
    )

    save_confusion_matrix_plot(
        matrix=normalized_matrix,
        output_path=OUTPUT_DIR / "normalized_confusion_matrix.png",
        title="Normalized DeepFace Emotion Confusion Matrix (TTA)",
        normalized=True,
    )

    # ─────────────────────────────────────────────────────────────────────
    # PER-CLASS SUMMARY
    # ─────────────────────────────────────────────────────────────────────

    per_class_metrics: dict[str, dict[str, float | int]] = {}

    for emotion in EMOTION_LABELS:
        class_report = report_dictionary.get(emotion, {})
        per_class_metrics[emotion] = {
            "precision": round(safe_float(class_report.get("precision")), 4),
            "recall": round(safe_float(class_report.get("recall")), 4),
            "f1_score": round(safe_float(class_report.get("f1-score")), 4),
            "specificity": round(specificity.get(emotion, 0.0), 4),
            "support": int(safe_float(class_report.get("support"))),
        }

    metrics_summary = {
        "dataset": str(TEST_DATASET_DIR),
        "labels": EMOTION_LABELS,
        "total_images_found": total_found,
        "scored_images": scored_images,
        "skipped_low_quality": skipped_images,
        "failed_images": failed_images,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": scored_images - correct_predictions,
        "low_confidence_predictions": int(low_confidence_count),
        "accuracy": round(float(accuracy), 4),
        "accuracy_percentage": round(float(accuracy * 100), 2),
        "high_confidence_only_accuracy": round(float(high_conf_accuracy), 4),
        "high_confidence_only_accuracy_percentage": round(float(high_conf_accuracy * 100), 2),
        "macro_precision": round(float(precision_macro), 4),
        "macro_recall": round(float(recall_macro), 4),
        "macro_f1_score": round(float(f1_macro), 4),
        "weighted_precision": round(float(precision_weighted), 4),
        "weighted_recall": round(float(recall_weighted), 4),
        "weighted_f1_score": round(float(f1_weighted), 4),
        "total_processing_time_seconds": round(total_time, 2),
        "average_processing_time_seconds": round(average_processing_time, 4),
        "detector_backend": DETECTOR_BACKEND,
        "enforce_detection": ENFORCE_DETECTION,
        "tta_variants": ["original", "flipped", "brighter", "darker", "clahe"],
        "quality_thresholds": {
            "min_blur_variance": MIN_BLUR_VARIANCE,
            "min_brightness": MIN_BRIGHTNESS,
            "max_brightness": MAX_BRIGHTNESS,
            "min_face_fraction": MIN_FACE_FRACTION,
        },
        "low_confidence_threshold_percent": LOW_CONFIDENCE_THRESHOLD,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized_matrix.round(4).tolist(),
    }

    with open(OUTPUT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)

    # ─────────────────────────────────────────────────────────────────────
    # READABLE SUMMARY
    # ─────────────────────────────────────────────────────────────────────

    summary_lines = [
        "=" * 80,
        "MOODFOREST DEEPFACE EVALUATION SUMMARY (Quality-Aware + TTA)",
        "=" * 80,
        f"Total images found        : {total_found}",
        f"Scored (used in metrics)  : {scored_images}",
        f"Skipped (low quality)     : {skipped_images}",
        f"Failed (DeepFace error)   : {failed_images}",
        f"Correct predictions       : {correct_predictions}",
        f"Incorrect predictions     : {scored_images - correct_predictions}",
        f"Low-confidence predictions: {low_confidence_count}",
        "",
        f"Accuracy (all scored)          : {accuracy:.4f} ({accuracy * 100:.2f}%)",
        f"Accuracy (high-confidence only): {high_conf_accuracy:.4f} ({high_conf_accuracy * 100:.2f}%)",
        "",
        f"Macro precision           : {precision_macro:.4f}",
        f"Macro recall              : {recall_macro:.4f}",
        f"Macro F1-score            : {f1_macro:.4f}",
        "",
        f"Weighted precision        : {precision_weighted:.4f}",
        f"Weighted recall           : {recall_weighted:.4f}",
        f"Weighted F1-score         : {f1_weighted:.4f}",
        "",
        f"Total processing time     : {total_time:.2f} seconds",
        f"Average time per image    : {average_processing_time:.4f} seconds",
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
        "SKIP REASONS BREAKDOWN",
        "-" * 80,
    ])

    if skipped_rows:
        reason_counts = pd.DataFrame(skipped_rows)["reason"].value_counts()
        for reason, count in reason_counts.items():
            summary_lines.append(f"  {reason:<20}: {count}")
    else:
        summary_lines.append("  (none skipped)")

    summary_lines.extend([
        "",
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

    with open(OUTPUT_DIR / "metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    # ─────────────────────────────────────────────────────────────────────
    # PRINT FINAL RESULTS
    # ─────────────────────────────────────────────────────────────────────

    print()
    print(summary_text)

    print("=" * 80)
    print("FILES CREATED")
    print("=" * 80)
    for filename in [
        "predictions.csv",
        "skipped_images.csv",
        "failed_images.csv",
        "classification_report.csv",
        "classification_report.txt",
        "metrics_summary.txt",
        "metrics_summary.json",
        "confusion_matrix_values.csv",
        "confusion_matrix.png",
        "normalized_confusion_matrix.csv",
        "normalized_confusion_matrix.png",
    ]:
        print(OUTPUT_DIR / filename)


if __name__ == "__main__":
    main()