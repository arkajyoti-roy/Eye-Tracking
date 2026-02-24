import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
import time
import math

class EyeTracker:
    def __init__(self):
        self.model_path = 'face_landmarker.task'
        if not os.path.exists(self.model_path):
            print("Downloading Face Landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # --- CAMERA SETUP (Default Resolution) ---
        self.cap = cv2.VideoCapture(0)
        
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        self.start_time = time.time()
        self.prev_frame_time = time.time()
        self.blink_count = 0
        self.eye_closed = False
        self.ear_threshold = 0.22 
        self.attention_score = 100.0

    def calculate_ear(self, landmarks, eye_indices):
        pts = [landmarks[i] for i in eye_indices]
        v1 = math.hypot(pts[1].x - pts[5].x, pts[1].y - pts[5].y)
        v2 = math.hypot(pts[2].x - pts[4].x, pts[2].y - pts[4].y)
        h = math.hypot(pts[0].x - pts[3].x, pts[0].y - pts[3].y)
        return (v1 + v2) / (2.0 * h) if h != 0 else 0

    def get_head_direction(self, landmarks):
        nose_tip = landmarks[1].x
        left_edge = landmarks[234].x
        right_edge = landmarks[454].x
        face_width = right_edge - left_edge
        if face_width == 0: return "FORWARD"
        ratio = (nose_tip - left_edge) / face_width
        if ratio < 0.40: return "RIGHT"
        if ratio > 0.60: return "LEFT"
        return "FORWARD"

    def draw_dashboard(self, frame, fps, session_time, ear, head_dir, gaze_text, attention):
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (360, 260), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (200, 200, 200) 
        
        cv2.putText(frame, f"Blinks : {self.blink_count}", (20, 45), font, 0.7, color, 2)
        cv2.putText(frame, f"EAR    : {ear:.3f}", (20, 75), font, 0.7, color, 2)
        cv2.putText(frame, f"FPS    : {int(fps)}", (20, 105), font, 0.7, color, 2)
        cv2.putText(frame, f"Time   : {session_time}", (20, 135), font, 0.7, color, 2)
        cv2.putText(frame, f"Gaze   : {gaze_text}", (20, 165), font, 0.7, color, 2)
        cv2.putText(frame, f"Head   : {head_dir}", (20, 195), font, 0.7, color, 2)
        
        attn_color = (0, 255, 0) if attention > 70 else (0, 255, 255) if attention > 40 else (0, 0, 255)
        cv2.putText(frame, f"Attn   : {int(attention)}%", (20, 235), font, 0.8, attn_color, 2)

        cv2.rectangle(frame, (20, 250), (340, 260), (100, 100, 100), -1) 
        cv2.rectangle(frame, (20, 250), (20 + int(3.2 * attention), 260), attn_color, -1) 

    def get_pupil_coords(self):
        success, frame = self.cap.read()
        if not success: return None

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        current_time = time.time()
        fps = 1 / (current_time - self.prev_frame_time) if (current_time - self.prev_frame_time) > 0 else 0
        self.prev_frame_time = current_time
        
        elapsed = int(current_time - self.start_time)
        session_time = f"{elapsed // 60:02d}:{elapsed % 60:02d}"

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        coords = None

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            
            left_ear = self.calculate_ear(landmarks, self.LEFT_EYE)
            right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < self.ear_threshold:
                if not self.eye_closed:
                    self.blink_count += 1
                    self.eye_closed = True
            else:
                self.eye_closed = False

            head_dir = self.get_head_direction(landmarks)

            # Pupil centers
            left_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.LEFT_IRIS]
            (lcx, lcy), l_radius = cv2.minEnclosingCircle(np.array(left_pts, dtype=np.int32))
            
            right_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.RIGHT_IRIS]
            (rcx, rcy), r_radius = cv2.minEnclosingCircle(np.array(right_pts, dtype=np.int32))
            
            # Eye socket geometric centers
            left_eye_center_x = (landmarks[362].x + landmarks[263].x) / 2.0 * w
            left_eye_center_y = (landmarks[385].y + landmarks[373].y) / 2.0 * h
            right_eye_center_x = (landmarks[33].x + landmarks[133].x) / 2.0 * w
            right_eye_center_y = (landmarks[160].y + landmarks[144].y) / 2.0 * h
            
            # --- NEW: Red dot perfectly centered between the two eyes ---
            mid_eyes_x = int((left_eye_center_x + right_eye_center_x) / 2.0)
            mid_eyes_y = int((left_eye_center_y + right_eye_center_y) / 2.0)
            cv2.circle(frame, (mid_eyes_x, mid_eyes_y), 4, (0, 0, 255), -1)

            # Draw circles on the pupils (Lines removed!)
            cv2.circle(frame, (int(lcx), int(lcy)), int(l_radius)+2, (0, 255, 0), 2)
            cv2.circle(frame, (int(rcx), int(rcy)), int(r_radius)+2, (0, 255, 0), 2)

            # Gaze Text Logic
            gaze_dir = "FORWARD"
            avg_gaze_ratio = 0.5 
            
            if head_dir == "FORWARD":
                l_min_x = min(landmarks[362].x, landmarks[263].x) * w
                l_max_x = max(landmarks[362].x, landmarks[263].x) * w
                l_ratio = (lcx - l_min_x) / (l_max_x - l_min_x) if (l_max_x - l_min_x) > 0 else 0.5

                r_min_x = min(landmarks[33].x, landmarks[133].x) * w
                r_max_x = max(landmarks[33].x, landmarks[133].x) * w
                r_ratio = (rcx - r_min_x) / (r_max_x - r_min_x) if (r_max_x - r_min_x) > 0 else 0.5

                avg_gaze_ratio = (l_ratio + r_ratio) / 2.0

                if avg_gaze_ratio < 0.43: gaze_dir = "RIGHT"
                elif avg_gaze_ratio > 0.53: gaze_dir = "LEFT"

            display_gaze = f"{gaze_dir} ({avg_gaze_ratio:.2f})"

            target_attn = 100.0
            if avg_ear < self.ear_threshold: target_attn -= 50 
            if head_dir != "FORWARD": target_attn -= 30        
            if gaze_dir != "FORWARD": target_attn -= 20        
            self.attention_score = self.attention_score * 0.9 + target_attn * 0.1

            self.draw_dashboard(frame, fps, session_time, avg_ear, head_dir, display_gaze, self.attention_score)

            coords = {
                "average": {"x": ((lcx + rcx) / 2) / w, "y": ((lcy + rcy) / 2) / h},
                "metrics": {
                    "blinks": self.blink_count,
                    "ear": round(avg_ear, 3),
                    "attention": round(self.attention_score, 1),
                    "head": head_dir
                }
            }

        cv2.imshow("Eye Tracking System", frame)
        cv2.waitKey(1) 

        return coords

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()