import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Convert frame to MediaPipe Image format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect landmarks
    result = detector.detect(mp_image)

    # Draw landmark points
    if result.hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.imshow("MediaPipe Hands (Tasks API)", cv2.flip(frame, 1))
    if cv2.waitKey(5) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()