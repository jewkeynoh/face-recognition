# -*- coding: utf-8 -*-
"""
Configuration settings for the Face Recognition project.

This module centralizes settings like file paths, model parameters,
and processing options to make the application easier to configure
and maintain.
"""

import os
from pathlib import Path

# --- Project Root ---
# Assumes this config file is in the project root or a subdirectory.
# Adjust if your structure is different.
PROJECT_ROOT = Path(__file__).parent.resolve()

# --- Directory Paths ---
# Use Path objects for easier path manipulation across OS.
KNOWN_FACES_DIR = PROJECT_ROOT / "known_faces"
OUTPUT_DIR = PROJECT_ROOT  # Directory to save the encodings file
LOG_DIR = PROJECT_ROOT / "logs" # Directory for log files

# --- File Paths ---
ENCODINGS_FILE_NAME = "known_face_encodings.pkl"
ENCODINGS_PATH = OUTPUT_DIR / ENCODINGS_FILE_NAME

# --- Face Recognition Model ---
# Model for locating faces in images ('hog' is faster, 'cnn' is more accurate but slower)
DETECTION_MODEL = 'hog'
# Model for generating face encodings. Options depend on the library,
# but face_recognition uses a standard model.
# Number of times to upsample image when looking for faces (higher = find smaller faces)
UPSAMPLE_TIMES = 1
# How many jitters to apply when encoding (higher = more robust but slower)
NUM_JITTERS = 1

# --- Face Comparison ---
# Tolerance for face comparison. Lower means stricter matching. 0.6 is common.
# Experiment with this value based on performance.
MATCH_TOLERANCE = 0.6

# --- Real-time Recognition Settings ---
# Scale down frame for faster processing (e.g., 0.5 = half size, 1.0 = original size)
FRAME_PROCESS_SCALE = 0.5
# Process only every Nth frame to save resources (1 = process every frame)
PROCESS_EVERY_N_FRAMES = 2

# --- Display Settings ---
# Font for displaying names/text on the video feed
FONT = 'FONT_HERSHEY_DUPLEX' # cv2 Font type (use string name)
FONT_SCALE = 0.6
FONT_THICKNESS = 1
BOX_COLOR_KNOWN = (0, 255, 0)      # Green for known faces
TEXT_COLOR_KNOWN = (255, 255, 255) # White text
BOX_COLOR_UNKNOWN = (0, 0, 255)    # Red for unknown faces
TEXT_COLOR_UNKNOWN = (255, 255, 255)# White text
FPS_POSITION = (5, 25) # Position for FPS counter (x, y)

# --- Logging Configuration ---
LOG_FILE = LOG_DIR / "face_recognition.log"
LOG_LEVEL = "INFO"  # Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

# --- Ensure directories exist ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)