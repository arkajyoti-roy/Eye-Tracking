import time
from core.eye_tracker import EyeTracker

def main():
    print("🚀 Starting Pure Real-Time Eye Tracker...")
    tracker = EyeTracker()
    
    print("🎥 Camera running. Press 'CTRL + C' in this terminal to stop.")
    
    try:
        # The Main Vision Loop
        while True:
            # This function reads the camera, calculates the math, 
            # and draws the dashboard window automatically.
            tracker.get_pupil_coords()
            
            # A tiny pause so your CPU doesn't run at 100%
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Session Stopped! Closing camera...")
    finally:
        tracker.release()
        print("Camera released successfully.")

if __name__ == '__main__':
    main()