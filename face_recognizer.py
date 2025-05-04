# -*- coding: utf-8 -*-
"""
Performs real-time face recognition using a webcam feed.

Loads known face encodings, captures video from the webcam, detects faces
in each frame (or every Nth frame), compares them against known faces,
and displays the results with bounding boxes and names.
"""

import face_recognition
import cv2
import pickle
import numpy as np
import logging
import sys
import time
from pathlib import Path

# Import configuration settings
try:
    import config
except ImportError:
    print("[ERROR] config.py not found. Please ensure it's in the same directory or Python path.")
    sys.exit(1)

# --- Setup Logging ---
# (Could be moved to a shared utility module in larger projects)
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler() # Also print logs to console
    ]
)

# --- Load OpenCV Font ---
# We get the font constant dynamically from cv2 based on config string
try:
    CV2_FONT = getattr(cv2, config.FONT)
except AttributeError:
    logging.warning(f"Could not find OpenCV font '{config.FONT}'. Falling back to FONT_HERSHEY_SIMPLEX.")
    CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_known_encodings(filepath):
    """
    Loads known face encodings and names from a pickle file.

    Args:
        filepath (Path): The path to the pickle file containing encodings.

    Returns:
        tuple: A tuple containing:
            - list: A list of known face encodings (numpy arrays).
            - list: A list of corresponding known names (strings).
        Returns (None, None) if loading fails or file is empty/invalid.
    """
    logging.info(f"Loading known face encodings from {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        if "encodings" not in data or "names" not in data:
             logging.error(f"Invalid data structure in encodings file: {filepath}")
             return None, None

        known_encodings = data["encodings"]
        known_names = data["names"]

        if not known_encodings or not known_names:
            logging.warning(f"No encodings or names found in {filepath}. Recognition will not work.")
            # Return empty lists instead of None? Consistent with encode function output.
            return [], []

        logging.info(f"Loaded {len(known_names)} known encodings.")
        return known_encodings, known_names

    except FileNotFoundError:
        logging.error(f"Encodings file not found: {filepath}")
        logging.error("Please run the face_encoder.py script first.")
        return None, None
    except pickle.UnpicklingError as e:
        logging.error(f"Failed to unpickle encodings file {filepath}: {e}")
        return None, None
    except Exception as e: # Catch other potential errors
        logging.error(f"An unexpected error occurred while loading encodings: {e}")
        return None, None


def initialize_webcam(source=0):
    """
    Initializes and returns the webcam video capture object.

    Args:
        source (int or str): The camera source index (e.g., 0 for default)
                             or video file path.

    Returns:
        cv2.VideoCapture: The initialized video capture object, or None if failed.
    """
    logging.info(f"Initializing video capture source: {source}...")
    video_capture = cv2.VideoCapture(source)
    if not video_capture.isOpened():
        logging.critical(f"Could not open video source: {source}")
        return None
    logging.info("Video capture initialized successfully.")
    return video_capture


def process_frame(frame, known_encodings, known_names):
    """
    Detects and recognizes faces in a single video frame.

    Args:
        frame (numpy.ndarray): The video frame (in BGR format).
        known_encodings (list): List of known face encodings.
        known_names (list): List of corresponding known names.

    Returns:
        tuple: A tuple containing:
            - list: List of face locations (top, right, bottom, left).
            - list: List of names corresponding to the detected faces.
    """
    # --- Frame Preprocessing ---
    # Resize frame for faster processing (if scale factor is not 1.0)
    if config.FRAME_PROCESS_SCALE != 1.0:
        small_frame = cv2.resize(frame, (0, 0), fx=config.FRAME_PROCESS_SCALE, fy=config.FRAME_PROCESS_SCALE)
    else:
        small_frame = frame

    # Convert BGR (OpenCV default) to RGB (face_recognition default)
    # Use cvtColor directly on the potentially smaller frame
    try:
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        logging.error(f"OpenCV error during color conversion: {e}")
        return [], [] # Return empty if conversion fails

    # --- Face Detection and Encoding ---
    try:
        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(
            rgb_small_frame,
            model=config.DETECTION_MODEL,
            number_of_times_to_upsample=config.UPSAMPLE_TIMES # Use same upsample as encoder? Maybe less for speed? Configurable?
        )
        # Encode the faces found in the current frame
        # This can be time-consuming
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame,
            known_face_locations=face_locations,
            num_jitters=config.NUM_JITTERS # Use same jitters as encoder? Maybe less for speed?
        )
    except Exception as e:
        # Catch potential errors from face_recognition library (e.g., memory)
        logging.error(f"Error during face detection/encoding: {e}")
        return [], []

    # --- Face Matching ---
    detected_names = []
    if not known_encodings: # Skip matching if no known faces are loaded
        detected_names = ["Unknown"] * len(face_encodings)
    else:
        for face_encoding in face_encodings:
            # Compare the current face encoding against all known encodings
            matches = face_recognition.compare_faces(
                known_encodings,
                face_encoding,
                tolerance=config.MATCH_TOLERANCE
            )
            name = "Unknown" # Default name

            # Use face_distance to find the best match among the 'True' matches
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            # Check if the best match index corresponds to a 'True' match in 'matches'
            # AND if the distance is within tolerance (redundant check but safe)
            if matches[best_match_index] and face_distances[best_match_index] < config.MATCH_TOLERANCE:
                name = known_names[best_match_index]
                logging.debug(f"Match found: {name} (Distance: {face_distances[best_match_index]:.2f})")

            detected_names.append(name)

    return face_locations, detected_names


def draw_detections(frame, face_locations, detected_names):
    """
    Draws bounding boxes and names onto the frame.

    Args:
        frame (numpy.ndarray): The original video frame to draw on.
        face_locations (list): List of face locations (top, right, bottom, left)
                                in the *processed* frame's scale.
        detected_names (list): List of names for each detected face.
    """
    # Scale factor to convert locations back to original frame size
    scale_factor = 1.0 / config.FRAME_PROCESS_SCALE

    for (top, right, bottom, left), name in zip(face_locations, detected_names):
        # Scale back up face locations
        top = int(top * scale_factor)
        right = int(right * scale_factor)
        bottom = int(bottom * scale_factor)
        left = int(left * scale_factor)

        # Choose color based on whether the face is known or unknown
        is_known = name != "Unknown"
        box_color = config.BOX_COLOR_KNOWN if is_known else config.BOX_COLOR_UNKNOWN
        text_color = config.TEXT_COLOR_KNOWN if is_known else config.TEXT_COLOR_UNKNOWN

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

        # Draw a label with a name below the face
        # Create a filled rectangle for the text background for better visibility
        label_y = bottom - 15 # Default position inside the box
        # Adjust if label would be too high (e.g., face near top) - basic check
        # A more robust check might be needed depending on font size
        if top < 30:
            label_y = bottom + 15

        try:
            cv2.rectangle(frame, (left, label_y - int(config.FONT_SCALE*25)), (right, bottom), box_color, cv2.FILLED) # Adjust height based on font scale
            cv2.putText(frame, name, (left + 6, bottom - 6), CV2_FONT, config.FONT_SCALE, text_color, config.FONT_THICKNESS)
        except Exception as e:
            logging.error(f"Error drawing text/rectangle: {e}") # Catch potential drawing errors


def run_recognition():
    """Main loop for capturing video, processing frames, and displaying results."""
    known_encodings, known_names = load_known_encodings(config.ENCODINGS_PATH)
    if known_encodings is None: # Loading failed critically
        return # Exit if encodings couldn't be loaded

    video_capture = initialize_webcam(0) # Use 0 for default webcam
    if video_capture is None:
        return # Exit if webcam failed

    frame_count = 0
    prev_frame_time = 0 # For FPS calculation

    logging.info("Starting real-time recognition loop...")
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            logging.error("Failed to grab frame from video source. Exiting.")
            break

        # --- Frame Processing (Optional Skipping) ---
        # Only process every Nth frame if configured
        if frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
            # Process the frame: find locations and recognize faces
            face_locations, detected_names = process_frame(frame, known_encodings, known_names)
            logging.debug(f"Processed frame {frame_count}: Found {len(face_locations)} faces. Names: {detected_names}")
        # We still need to draw on every frame using the *last known* locations/names
        # This prevents flickering if we skip processing frames.

        # --- Drawing Results ---
        # Draw boxes and names based on the *last processed* results
        draw_detections(frame, face_locations, detected_names)

        # --- FPS Calculation ---
        new_frame_time = time.time()
        # Avoid division by zero on the first frame
        if prev_frame_time != 0:
            fps = 1 / (new_frame_time - prev_frame_time)
            fps_text = f"FPS: {int(fps)}"
            try:
                 cv2.putText(frame, fps_text, config.FPS_POSITION, CV2_FONT, config.FONT_SCALE, config.TEXT_COLOR_KNOWN, config.FONT_THICKNESS)
            except Exception as e:
                logging.error(f"Error drawing FPS text: {e}")

        prev_frame_time = new_frame_time

        # --- Display Frame ---
        try:
            cv2.imshow('Video Face Recognition', frame)
        except Exception as e:
            logging.error(f"Error displaying frame with cv2.imshow: {e}")
            break # Exit loop if display fails

        frame_count += 1

        # --- Exit Condition ---
        # Hit 'q' on the keyboard to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exit key ('q') pressed. Stopping recognition.")
            break

    # --- Cleanup ---
    logging.info("Releasing video capture device and closing windows...")
    video_capture.release()
    cv2.destroyAllWindows()
    logging.info("Cleanup complete.")


def main():
    """Main function to start the face recognition process."""
    run_recognition()


if __name__ == "__main__":
    main()