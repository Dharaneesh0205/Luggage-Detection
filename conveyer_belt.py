import cv2
from ultralytics import YOLO
import numpy as np
import easyocr
import pyttsx3
import os
from collections import defaultdict, deque

# ========== CONFIGURATION ==========
MODEL_PATH = 'yolov8n.pt'
BAG_CLASS_ID = 28  # 28 is 'suitcase' in COCO. Change if your class is different.

FOCAL_LENGTH = 700  # pixels (calibrate for your camera)
BAG_REAL_WIDTH = 40  # centimeters

VIDEO_PATH = "IMG_3252.MOV"  # <-- Change this to your file name or path

AUDIO_DISTANCE_THRESHOLD = 0.2  # meters, only announce if bag moves more than this

model = YOLO(MODEL_PATH)
qr_detector = cv2.QRCodeDetector()
ocr_reader = easyocr.Reader(['en'], gpu=False)
engine = pyttsx3.init()

# For smoothing distance estimation per object
distance_histories = defaultdict(lambda: deque(maxlen=5))
last_speak_distance = {}

def estimate_distance(box):
    x1, y1, x2, y2 = box
    pixel_width = max(x2 - x1, 1)
    distance_cm = (BAG_REAL_WIDTH * FOCAL_LENGTH) / pixel_width
    return round(distance_cm / 100, 2)  # meters

def speak(text, key, distance):
    # Only speak if distance changed by more than threshold for the given key (object)
    if key not in last_speak_distance or abs(last_speak_distance[key] - distance) > AUDIO_DISTANCE_THRESHOLD:
        engine.say(text)
        engine.runAndWait()
        last_speak_distance[key] = distance

def process_bag(frame, box):
    x1, y1, x2, y2 = [int(i) for i in box]
    roi = frame[y1:y2, x1:x2]
    qr_data, _, _ = qr_detector.detectAndDecode(roi)
    if qr_data:
        return qr_data, "QR"
    ocr_results = ocr_reader.readtext(roi)
    if ocr_results:
        text = ocr_results[0][1]
        return text, "OCR"
    return None, None

def is_image_file(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    return ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

def process_frame(frame, frame_idx):
    results = model(frame)[0]
    detected = False

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == BAG_CLASS_ID:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            distance = estimate_distance((x1, y1, x2, y2))
            # Use the box coordinates as a key for smoothing (could use tracking ID for better results)
            box_key = (x1, y1, x2, y2)
            distance_histories[box_key].append(distance)
            avg_distance = round(np.mean(distance_histories[box_key]), 2)
            info, method = process_bag(frame, (x1, y1, x2, y2))
            label = f"Bag {i+1} ({avg_distance}m)"
            speak_text = f"Bag {i+1} detected at {avg_distance} meters."
            if info:
                label += f" [{method}: {info}]"
                speak_text += f" {method} reads {info}."
            cv2.putText(frame, label, (x1, max(0, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            speak(speak_text, box_key, avg_distance)
            detected = True
    return frame, detected

def main(input_path=VIDEO_PATH):
    if isinstance(input_path, str) and is_image_file(input_path):
        frame = cv2.imread(input_path)
        if frame is None:
            print("Error: Cannot open image file.")
            return
        frame, detected = process_frame(frame, 0)
        display_frame = resize_to_fit(frame, 1280, 720)
        cv2.imshow("Bag Detection - Image", display_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error: Cannot open camera or video file.")
            return
        frame_idx = 0
        frame_skip = 1  # Set to >1 to skip frames and process faster (e.g., 2 for every other frame)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                frame, detected = process_frame(frame, frame_idx)
                display_frame = resize_to_fit(frame, 1280, 720)
                cv2.imshow("Bag Detection - Video", display_frame)
            frame_idx += 1
            # Fast playback: 1ms between frames, quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def resize_to_fit(frame, target_width, target_height):
    h, w = frame.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Create a black canvas and center the image
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

if __name__ == "__main__":
    main()