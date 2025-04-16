from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import base64
import numpy as np
from fer import FER
import random
import logging
from utils import emotion_to_youtube

# Configure logging
logging.basicConfig(level=logging.ERROR)

app = Flask(__name__)
CORS(app)
detector = FER()

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering index.html: {str(e)}")
        return jsonify({'error': 'Template not found'}), 500

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        # Validate JSON payload
        if not request.json or 'image' not in request.json:
            return jsonify({'error': 'Missing image in request'}), 400
        data_url = request.json['image']

        # Validate data URL
        if ',' not in data_url:
            return jsonify({'error': 'Invalid image data URL'}), 400
        encoded_data = data_url.split(',')[1]

        # Limit input size
        if len(encoded_data) > 10_000_000:  # ~10MB
            return jsonify({'error': 'Image too large'}), 400

        # Decode image
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Resize image for performance
        image = cv2.resize(image, (640, 480))

        # Detect emotion
        emotion, score = detector.top_emotion(image)
        if not emotion:
            return jsonify({'emotion': 'no_face', 'song_link': ''})

        if emotion and score > 0.95:
            song_list = emotion_to_youtube.get(emotion, emotion_to_youtube.get('neutral', []))
            song_link = random.choice(song_list) if song_list else ''
            return jsonify({"emotion": emotion, "score": score, "song_link": song_link})
        return jsonify({'emotion': 'none', 'song_link': ''})
    except base64.binascii.Error:
        return jsonify({'error': 'Invalid base64 encoding'}), 400
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    import os
    app.run(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true')