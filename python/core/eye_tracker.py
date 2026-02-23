import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
import base64 # ---> ADD THIS IMPORT

class EyeTracker:
    # ... (__init__ stays exactly the same) ...
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
        self.cap = cv2.VideoCapture(0)
        
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

    def get_pupil_coords(self):
        success, frame = self.cap.read()
        if not success:
            return None

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        coords = None

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            
            # 1. Process Eyes
            left_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.LEFT_IRIS]
            (lcx, lcy), l_radius = cv2.minEnclosingCircle(np.array(left_pts, dtype=np.int32))
            
            right_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.RIGHT_IRIS]
            (rcx, rcy), r_radius = cv2.minEnclosingCircle(np.array(right_pts, dtype=np.int32))
            
            # 2. Draw Center Points & Circles for HTML view
            cv2.circle(frame, (int(lcx), int(lcy)), int(l_radius), (0, 255, 0), 2)
            cv2.circle(frame, (int(lcx), int(lcy)), 2, (0, 0, 255), -1) # Red dot in center
            
            cv2.circle(frame, (int(rcx), int(rcy)), int(r_radius), (0, 255, 0), 2)
            cv2.circle(frame, (int(rcx), int(rcy)), 2, (0, 0, 255), -1) # Red dot in center
            
            # 3. Compress frame for WebSocket streaming
            # We resize it to keep the data lightweight so it doesn't lag
            small_frame = cv2.resize(frame, (480, 360))
            _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # 4. Pack the data
            coords = {
                "left": {"x": lcx / w, "y": lcy / h},
                "right": {"x": rcx / w, "y": rcy / h},
                "average": {"x": ((lcx + rcx) / 2) / w, "y": ((lcy + rcy) / 2) / h},
                "video_frame": f"data:image/jpeg;base64,{frame_base64}" # Add image to payload
            }

        # You can comment out the lines below if you don't want the python popup window anymore
        cv2.imshow("Python Vision Debug", frame)
        cv2.waitKey(1) 

        return coords

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()