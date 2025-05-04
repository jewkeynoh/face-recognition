# Python Real-time Face Recognition Project

**(README updated: Sunday, May 4, 2025, 4:05 PM PST - Calasiao, Philippines)**

## Overview

This project implements a real-time face recognition system using Python, OpenCV, and the `face_recognition` library (based on dlib). It can:

1.  Learn known faces from a directory of images.
2.  Recognize those known faces in a live webcam feed.
3.  Label faces as either known individuals or "Unknown".

This version has been refactored to follow better software development practices, including configuration management, logging, modular code structure, and improved error handling.

## Features

* **Encode Known Faces:** Processes images from a specified directory structure (`known_faces/PersonName/image.jpg`) and saves facial encodings.
* **Real-time Recognition:** Captures video from a webcam, detects faces, and compares them against known encodings.
* **Configurable:** Uses a `config.py` file to manage paths, model parameters, tolerance, and performance settings.
* **Logging:** Implements logging (to console and file) for better tracking and debugging.
* **Performance Options:** Includes options to scale down video frames and process only every Nth frame for performance tuning.
* **Modular Code:** Functionality is broken down into separate functions and scripts (`face_encoder.py`, `face_recognizer.py`).
* **Basic Error Handling:** Includes checks for missing files, directories, and camera access issues.

## Project Structure

```text
face_recognition_project/
├── .gitignore              # Git ignore file
├── venv/                   # Python virtual environment directory (created by user)
├── known_faces/            # Root directory for known face images
│   ├── Person1_Name/       # Subdirectory for each known person
│   │   └── image1.jpg      # Image(s) of Person1
│   └── Person2_Name/
│       └── picA.png        # Image(s) of Person2
├── logs/                   # Directory for log output (created automatically)
│   └── face_recognition.log
├── config.py               # Central configuration settings
├── face_encoder.py         # Script to encode known faces
├── face_recognizer.py      # Script for real-time recognition via webcam
├── requirements.txt        # Python package dependencies
├── README.md               # This documentation file
│
├── known_face_encodings.pkl # (Optional) Output of face_encoder.py
└── LICENSE                 # (Optional) Add your chosen license file (e.g., MIT)

## Prerequisites

Before running this project, ensure you have the following installed on your system:

1.  **Python:** Version 3.6 or higher is recommended.
2.  **pip:** Python's package installer (usually comes with Python).
3.  **CMake:** Required to build the `dlib` library. Download from [cmake.org](https://cmake.org/download/) or install via your system's package manager (e.g., `sudo apt install cmake`, `brew install cmake`). **Ensure it's added to your system PATH.**
4.  **C++ Compiler:** Required to build `dlib`.
    * **Windows:** Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) (Select "Desktop development with C++" workload during installation).
    * **macOS:** Install Xcode Command Line Tools (`xcode-select --install`).
    * **Linux (Debian/Ubuntu):** Install `build-essential` (`sudo apt update && sudo apt install build-essential`).

## Setup

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd face_recognition_project
    ```
    Or simply create the project directory and place the files inside.

2.  **Create and Activate Virtual Environment:** It's highly recommended to use a virtual environment.
    ```bash
    # Create the virtual environment (use python3 if needed)
    python -m venv venv

    # Activate the environment
    # Windows (cmd.exe):
    .\venv\Scripts\activate.bat
    # Windows (PowerShell):
    .\venv\Scripts\Activate.ps1
    # Linux / macOS:
    source venv/bin/activate
    ```

3.  **Install Dependencies:** Install the required Python packages using `pip`.
    ```bash
    # Ensure your virtual environment is activated
    pip install -r requirements.txt
    ```
    *Note: Installing `dlib` (a dependency of `face_recognition`) can take several minutes as it often needs to be compiled from source.*

## Configuration

Review and modify the settings in `config.py` before running the scripts. Key options include:

* `KNOWN_FACES_DIR`: Path to the directory containing known faces images.
* `ENCODINGS_PATH`: Path where the generated face encodings file will be saved.
* `DETECTION_MODEL`: Face detection model (`'hog'` or `'cnn'`).
* `MATCH_TOLERANCE`: Threshold for face matching (lower is stricter).
* `FRAME_PROCESS_SCALE`: Factor to resize video frames for faster processing.
* `PROCESS_EVERY_N_FRAMES`: Set > 1 to skip processing frames for performance.
* `LOG_LEVEL`: Adjust logging verbosity (`DEBUG`, `INFO`, `WARNING`, etc.).

## Usage

Ensure your virtual environment is activated before running the scripts.

### 1. Encoding Known Faces

* **Prepare Images:** Create subdirectories within the `known_faces` directory (specified in `config.py`). Name each subdirectory after the person. Place one or more clear images of that person inside their respective directory (e.g., `known_faces/John_Doe/photo1.jpg`). The encoder currently expects exactly one face per image file.
* **Run the Encoder:** Execute the `face_encoder.py` script.
    ```bash
    python face_encoder.py
    ```
    This will process the images, generate encodings, and save them to the file specified by `ENCODINGS_PATH` in `config.py` (default: `known_face_encodings.pkl`). Check the console output and `logs/face_recognition.log` for details.

### 2. Running Real-time Recognition

* **Run the Recognizer:** Execute the `face_recognizer.py` script. Make sure your webcam is connected.
    ```bash
    python face_recognizer.py
    ```
    * A window will appear showing your webcam feed.
    * Detected faces will be outlined with a bounding box.
    * Known faces will be labeled with their names (green box).
    * Unknown faces will be labeled "Unknown" (red box).
    * An FPS counter is displayed.
* **Quit:** Press the 'q' key while the video window is active to stop the recognition script.

## Logging

* The application uses Python's `logging` module.
* Logs are printed to the console and also saved to `logs/face_recognition.log` (configurable in `config.py`).
* Log level can be adjusted in `config.py` for more or less detailed output. Check the log file for detailed information, especially if errors occur.

## Dependencies

Key Python libraries used:

* `face_recognition`: For face detection, encoding, and comparison (uses `dlib`).
* `opencv-python`: For accessing the webcam and image/video manipulation (drawing).
* `numpy`: For numerical operations, especially with encodings.

See `requirements.txt` for a full list of direct Python dependencies. Remember the underlying system dependencies: `CMake` and a C++ compiler.

## Ethical Considerations

Face recognition technology has significant ethical implications. Please consider the following when using or developing this project:

* **Privacy:** Be mindful of privacy concerns. Avoid using this technology in a way that violates individuals' reasonable expectations of privacy. Obtain consent where necessary.
* **Bias:** Face recognition models can exhibit biases based on the data they were trained on, potentially performing differently for various demographic groups (e.g., race, gender, age). Be aware of and test for potential biases.
* **Consent:** Do not use this software to identify individuals without their explicit consent, especially in non-public spaces.
* **Security:** If storing face encodings or personal information, ensure appropriate security measures are in place.
* **Transparency:** Be transparent about how the technology is used if deployed in any application.

**Use this technology responsibly and ethically.**

## Future Improvements

* Store face encodings in a database (e.g., SQLite, PostgreSQL with vector extensions) instead of a pickle file for better scalability and management.
* Implement alternative, potentially faster face detectors (e.g., OpenCV's DNN-based detector, MTCNN).
* Develop a graphical user interface (GUI) instead of relying solely on the OpenCV display window.
* Add functionality to handle unknown faces (e.g., clustering unknowns, providing an interface to label new faces).
* Explore asynchronous processing for video capture and face analysis to potentially improve responsiveness.
* Implement more sophisticated error handling and recovery mechanisms.

## License

Please add a license file (e.g., `LICENSE`) to the project root. A common choice for open-source projects is the [MIT License](https://opensource.org/licenses/MIT).

*(Example: This project is licensed under the MIT License - see the LICENSE file for details.)*