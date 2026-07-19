from deepface import DeepFace
from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import cv2
import numpy as np
import base64
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

ALLOWED_EMOTIONS = ["angry", "fear", "happy", "neutral", "sad"]

SCORE_MAP = {
    "happy": 5,
    "neutral": 3,
    "sad": 2,
    "fear": 1,
    "angry": 1
}

vader = SentimentIntensityAnalyzer()


# ─────────────────────────────────────────
# SAFE JSON CONVERSION
# ─────────────────────────────────────────

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def clean_scores(scores):
    return {
        str(k): float(round(safe_float(v), 2))
        for k, v in scores.items()
    }


# ─────────────────────────────────────────
# TIME CONTEXT
# ─────────────────────────────────────────

def get_time_context(hour=None):
    if hour is None:
        hour = datetime.now().hour

    if 5 <= hour < 9:
        return "early_morning"
    elif 9 <= hour < 12:
        return "morning"
    elif 12 <= hour < 14:
        return "midday"
    elif 14 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 20:
        return "evening"
    elif 20 <= hour < 23:
        return "night"
    else:
        return "late_night"


# ─────────────────────────────────────────
# FUSION HELPERS
# ─────────────────────────────────────────

def empty_scores():
    return {
        "happy": 0.0,
        "neutral": 0.0,
        "sad": 0.0,
        "angry": 0.0,
        "fear": 0.0
    }


def normalize_scores(scores):
    clean = {
        k: safe_float(v)
        for k, v in scores.items()
    }

    total = sum(clean.values())

    if total == 0:
        return clean_scores(clean)

    normalized = {
        k: (v / total) * 100
        for k, v in clean.items()
    }

    return clean_scores(normalized)


def dominant_emotion(scores):
    if not scores:
        return "neutral"

    return str(max(scores, key=scores.get))


def journal_to_emotion_scores(journals):
    scores = empty_scores()

    if not journals:
        scores["neutral"] = 100.0
        return clean_scores(scores)

    for journal in journals:
        text = journal.get("text", "")
        sentiment = journal.get("sentiment")
        compound = journal.get("score", journal.get("sentimentScore", 0))

        compound = safe_float(compound)

        if not sentiment:
            vs = vader.polarity_scores(text)
            compound = safe_float(vs["compound"])

            if compound > 0.05:
                sentiment = "positive"
            elif compound < -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"

        if sentiment == "positive":
            scores["happy"] += 1.0

        elif sentiment == "negative":
            if compound < -0.5:
                scores["sad"] += 0.7
                scores["fear"] += 0.3
            else:
                scores["sad"] += 1.0

        else:
            scores["neutral"] += 1.0

    return normalize_scores(scores)


def face_to_emotion_scores(emotions):
    scores = empty_scores()

    if not emotions:
        scores["neutral"] = 100.0
        return clean_scores(scores)

    count = 0

    for record in emotions:
        record_scores = record.get("scores")

        if record_scores:
            for e in ALLOWED_EMOTIONS:
                scores[e] += safe_float(record_scores.get(e, 0))
            count += 1

        else:
            emotion = record.get("emotion", "neutral")
            confidence = safe_float(record.get("confidence", 50))

            if emotion in scores:
                scores[emotion] += confidence
                count += 1

    if count == 0:
        scores["neutral"] = 100.0
        return clean_scores(scores)

    averaged = {
        e: scores[e] / count
        for e in scores
    }

    return normalize_scores(averaged)


def fuse_scores(face_scores, journal_scores, face_weight=0.6, journal_weight=0.4):
    fused = {}

    for emotion in ALLOWED_EMOTIONS:
        fused[emotion] = (
            safe_float(face_scores.get(emotion, 0)) * safe_float(face_weight)
            +
            safe_float(journal_scores.get(emotion, 0)) * safe_float(journal_weight)
        )

    return normalize_scores(fused)


def generate_recommendation(emotion, mood_score, face_dom, journal_dom):
    if emotion == "happy":
        return "Your mood looks positive. Keep doing the activities that supported your emotional balance."

    if emotion == "neutral":
        if journal_dom in ["sad", "fear", "angry"]:
            return "Your face data looks mostly neutral, but your journal shows some negative feelings. Try resting, journaling more, or talking with someone you trust."
        return "Your mood is mostly balanced. Add one enjoyable activity to make your week more positive."

    if emotion == "sad":
        return "Your mood seems low. Try to rest properly, eat well, and talk to a trusted friend or family member."

    if emotion == "fear":
        return "Your data shows anxious or fearful patterns. Break tasks into smaller steps and try breathing or relaxation exercises."

    if emotion == "angry":
        return "Your mood shows frustration. Physical activity, taking a break, or writing down your thoughts may help."

    return "Keep tracking your mood regularly to understand your emotional patterns better."


# ─────────────────────────────────────────
# DEEPFACE DETECTION
# ─────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect_emotion():
    try:
        data = request.get_json()
        images = data.get("images", [])

        if not images:
            return jsonify({
                "success": False,
                "error": "No images"
            }), 400

        all_scores = []

        for img in images:
            img_bytes = base64.b64decode(img.split(",")[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="retinaface"
            )

            emotion_scores = result[0]["emotion"]

            filtered_scores = {
                emotion: safe_float(score)
                for emotion, score in emotion_scores.items()
                if emotion in ALLOWED_EMOTIONS
            }

            all_scores.append(filtered_scores)

        avg_scores = {}

        for emotion in ALLOWED_EMOTIONS:
            total = sum(
                safe_float(score.get(emotion, 0))
                for score in all_scores
            )

            avg_scores[emotion] = total / len(all_scores)

        avg_scores = normalize_scores(avg_scores)

        dominant = dominant_emotion(avg_scores)
        confidence = safe_float(avg_scores[dominant])

        return jsonify({
            "success": True,
            "emotion": str(dominant),
            "confidence": float(round(confidence, 2)),
            "scores": clean_scores(avg_scores),
            "model_used": "DeepFace Emotion Model"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ─────────────────────────────────────────
# JOURNAL ANALYSIS
# ─────────────────────────────────────────

@app.route("/analyze-journal", methods=["POST"])
def analyze_journal():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        vs = vader.polarity_scores(text)
        compound = safe_float(vs["compound"])

        if compound > 0.05:
            sentiment = "positive"
        elif compound < -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return jsonify({
            "success": True,
            "sentiment": sentiment,
            "score": float(round(compound, 4))
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ─────────────────────────────────────────
# FUSION ENDPOINT
# ─────────────────────────────────────────

@app.route("/fusion", methods=["POST"])
def fusion():
    try:
        data = request.get_json()

        emotions = data.get("emotions", [])
        journals = data.get("journals", [])

        period = data.get("period", "weekly")
        start_date = data.get("startDate")
        end_date = data.get("endDate")

        face_weight = safe_float(data.get("faceWeight", 0.6))
        journal_weight = safe_float(data.get("journalWeight", 0.4))

        total_weight = face_weight + journal_weight

        if total_weight == 0:
            face_weight = 0.6
            journal_weight = 0.4
        else:
            face_weight = face_weight / total_weight
            journal_weight = journal_weight / total_weight

        face_scores = face_to_emotion_scores(emotions)
        journal_scores = journal_to_emotion_scores(journals)

        fused_scores = fuse_scores(
            face_scores,
            journal_scores,
            face_weight,
            journal_weight
        )

        final_emotion = dominant_emotion(fused_scores)
        confidence = safe_float(fused_scores[final_emotion])

        mood_score = SCORE_MAP.get(final_emotion, 3)
        mood_score_100 = int(round((mood_score / 5) * 100))

        face_dom = dominant_emotion(face_scores)
        journal_dom = dominant_emotion(journal_scores)

        recommendation = generate_recommendation(
            final_emotion,
            mood_score_100,
            face_dom,
            journal_dom
        )

        return jsonify({
            "success": True,

            "period": str(period),
            "startDate": start_date,
            "endDate": end_date,

            "finalEmotion": str(final_emotion),
            "confidence": float(round(confidence, 2)),
            "moodScore": mood_score_100,

            "faceEmotion": str(face_dom),
            "journalEmotion": str(journal_dom),

            "faceScores": clean_scores(face_scores),
            "journalScores": clean_scores(journal_scores),
            "fusedScores": clean_scores(fused_scores),

            "faceWeight": float(round(face_weight, 2)),
            "journalWeight": float(round(journal_weight, 2)),

            "totalFaceRecords": int(len(emotions)),
            "totalJournalRecords": int(len(journals)),

            "recommendation": str(recommendation)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ─────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "model": "DeepFace + Journal Fusion Engine"
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )