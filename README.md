# Luggage Detection

<img width="505" height="900" alt="image" src="https://github.com/user-attachments/assets/35d7d4af-532a-4b53-84f9-75f2a13d68a1" />


## Overview

**Luggage Detection** is a Python-based computer vision project that detects and tracks luggage (bags, suitcases) on a conveyor belt using the YOLO (You Only Look Once) object detection framework. The system can process both images and video streams to identify bags, estimate their distance from the camera, and optionally read any QR codes or text (via OCR) present on the luggage for identification or automation purposes.

## Features

- **Real-Time Bag Detection:** Utilizes YOLOv8 for accurate suitcase detection in images and videos.
- **Distance Estimation:** Calculates distance of each detected bag from the camera using known bag width and camera focal length.
- **QR & Text Recognition:** Reads QR codes and uses OCR to extract relevant information from bags.
- **Audio Feedback:** Announces detection and information about each bag using text-to-speech.
- **Visual Overlay:** Draws bounding boxes and labels (with distance and info) over detected luggage.

## Example

The image above demonstrates the detection of a suitcase on a conveyor belt, with a bounding box and label indicating the detected object and its estimated distance.

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Dharaneesh0205/Luggage-Detection.git
   cd Luggage-Detection
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Required packages include: `opencv-python`, `ultralytics`, `easyocr`, `pyttsx3`, `numpy`*

3. **Download YOLOv8 Model Weights:**
   - By default, the code uses `yolov8n.pt` (nano version). Download it from [Ultralytics YOLOv8 release page](https://github.com/ultralytics/ultralytics/releases) or run:
     ```bash
     yolo download model=yolov8n.pt
     ```

## Usage

- **To process a video or camera stream:**
  ```bash
  python conveyer_belt.py
  ```
  - By default, processes the file `IMG_3252.MOV`. Change the file path in the script to your own video/image.

- **To process a static image:**
  - Set the `VIDEO_PATH` variable in `conveyer_belt.py` to your image file path.

## How It Works

1. **Detection:** Each frame is processed with YOLO to detect suitcases.
2. **Distance Estimation:** The width of the bounding box is used with camera calibration to estimate how far the bag is.
3. **QR/OCR:** The region inside the bounding box is scanned for QR codes and readable text.
4. **Audio + Visual Output:** Results are drawn on the frame, and information is spoken aloud if distance changes significantly.

## Configuration

- `MODEL_PATH`: Path to YOLO model weights.
- `BAG_CLASS_ID`: Class ID for "suitcase" (default is 28 for COCO dataset).
- `FOCAL_LENGTH`, `BAG_REAL_WIDTH`: Adjust these for accurate distance estimation based on your camera and typical bag size.
- `AUDIO_DISTANCE_THRESHOLD`: Minimum change in distance (meters) before speaking again.

## Requirements

- Python 3.7+
- OpenCV
- Ultralytics (YOLO)
- easyocr
- pyttsx3
- numpy

## License

This project is for demonstration and educational purposes. Please check with the repository owner for licensing details.

## Author

[Dharaneesh0205](https://github.com/Dharaneesh0205)
