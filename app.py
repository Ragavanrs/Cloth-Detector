from flask import Flask, jsonify, request
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to estimate the depth of the person based on chest size
def estimate_depth(chest_pixel_distance, known_chest_width_cm=40):
    focal_length = 500
    depth_cm = (known_chest_width_cm * focal_length) / chest_pixel_distance
    return depth_cm

# Function to process the image and detect body size (chest, waist)
def detect_body_size(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None, None
    
    landmarks = results.pose_landmarks.landmark
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

    frame_height, frame_width, _ = frame.shape
    left_hip_pixel = [int(left_hip[0] * frame_width), int(left_hip[1] * frame_height)]
    right_hip_pixel = [int(right_hip[0] * frame_width), int(right_hip[1] * frame_height)]
    left_shoulder_pixel = [int(left_shoulder[0] * frame_width), int(left_shoulder[1] * frame_height)]
    right_shoulder_pixel = [int(right_shoulder[0] * frame_width), int(right_shoulder[1] * frame_height)]

    waist_pixel_distance = calculate_distance(left_hip_pixel, right_hip_pixel)
    chest_pixel_distance = calculate_distance(left_shoulder_pixel, right_shoulder_pixel)

    known_pixel_distance = 50
    known_real_world_distance_cm = 10
    pixels_per_cm = known_pixel_distance / known_real_world_distance_cm

    waist_cm = waist_pixel_distance / pixels_per_cm
    chest_cm = chest_pixel_distance / pixels_per_cm

    estimated_depth_cm = estimate_depth(chest_pixel_distance)

    return waist_cm, chest_cm, estimated_depth_cm

@app.route('/process_video', methods=['POST'])
def process_video():
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    
    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 400
    
    waist, chest, depth = detect_body_size(frame)
    video_capture.release()

    if waist is None or chest is None:
        return jsonify({"error": "No pose detected"}), 400

    response = {
        "waist_cm": round(waist, 2),
        "chest_cm": round(chest, 2),
        "depth_cm": round(depth, 2)
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
