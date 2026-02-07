# Face Recognition Web Dashboard - Project Structure

This project implements a face recognition web dashboard using Python, OpenCV, and Streamlit.

## Features
- **Real-time Camera Feed**: Captures video from the webcam.
- **Face Detection**: Uses Hugging Face transformer models to detect faces.
- **Web Dashboard**: Interactive UI built with Streamlit.

## Getting Started

1.  **Clone the repository**
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application**:
    ```bash
    streamlit run src/main.py
    ```

## Project Structure
- `src/`: Source code
  - `main.py`: Application entry point
  - `camera.py`: Camera handling
  - `detector.py`: Face detection logic
  - `utils.py`: Helper functions
