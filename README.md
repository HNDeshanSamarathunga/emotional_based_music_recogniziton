# emotional_based_music_recogniziton

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import base64
import numpy as np
from fer import FER
import random
from utils import emotion_to_youtube

Flask Imports:
Flask: The main class to create the web application.
render_template: Renders HTML templates (e.g., index.html for a frontend).
request: Accesses data sent in HTTP requests (e.g., JSON data in POST requests).
jsonify: Converts Python dictionaries to JSON responses for the frontend.
CORS:
CORS (from flask_cors): Allows the backend to accept requests from a frontend running on a different domain or port (e.g., a React app on http://localhost:5173).
Image Processing:
cv2 (OpenCV): A library for computer vision tasks, used here to decode images.
base64: Decodes base64-encoded strings (used for images sent from the frontend).
numpy: Converts image data into arrays that OpenCV can process.
Emotion Detection:
fer: The Facial Expression Recognition library, which uses machine learning to detect emotions in facial images.
Utilities:
random: Used to pick a random YouTube link from a list of songs.
utils: A custom module (not shown) that provides emotion_to_youtube, likely a dictionary mapping emotions (e.g., "happy") to lists of YouTube URLs.