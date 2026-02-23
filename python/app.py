import socketio
import time
from core.eye_tracker import EyeTracker

# Create a Socket.IO client
sio = socketio.Client()

@sio.event
def connect():
    print("✅ Connected to Node.js Server!")

@sio.event
def disconnect():
    print("❌ Disconnected from Node.js Server!")

def main():
    # Connect to Node.js
    try:
        sio.connect('http://localhost:5000')
    except Exception as e:
        print(f"🚨 Could not connect to Node server: {e}")
        print("Make sure your Node server (server.js) is running on port 5000!")
        return

    tracker = EyeTracker()
    print("🎥 Starting camera loop...")

    try:
        # Simple loop running on the main thread! No blocking!
        while True:
            coords = tracker.get_pupil_coords()
            if coords:
                # Push the data to Node.js
                sio.emit('python_data', coords)
            
            time.sleep(0.01) # Small pause to save CPU
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        tracker.release()
        sio.disconnect()

if __name__ == '__main__':
    main()