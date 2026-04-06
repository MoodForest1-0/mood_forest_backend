from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)  # allows React to call this API


@app.route('/detect', methods=['POST'])
def detect_emotion():
    try:
        data = request.get_json()

        # The image comes as base64 from React webcam
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)

        # Convert to numpy array (what OpenCV understands)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run DeepFace emotion analysis
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False  # won't crash if no face found
        )

        dominant_emotion = result[0]['dominant_emotion']
        emotion_scores = {
            key: float(value)
            for key, value in result[0]['emotion'].items()
        }

        return jsonify({
            'success': True,
            'emotion': dominant_emotion,
            'scores': emotion_scores
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
