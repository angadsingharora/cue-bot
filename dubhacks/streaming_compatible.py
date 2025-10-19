"""
Streaming-Compatible Eye Tracking and Gesture Detection
This version is designed to work alongside OBS/streaming software
"""

import cv2
import mediapipe as mp
import time
import obsws_python as obs
import pygame
import numpy as np
import sys

def initialize_audio():
    """Initialize pygame mixer for audio feedback."""
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        return True
    except Exception as e:
        print(f"Audio initialization failed: {e}")
        return False

def play_gesture_sound(gesture_type):
    """Play different sounds for different gestures."""
    try:
        # Generate different tones for different gestures
        if gesture_type == "thumbs_up":
            frequency = 440  # A4 note
            duration = 0.3
        elif gesture_type == "peace":
            frequency = 523  # C5 note
            duration = 0.3
        elif gesture_type == "fist":
            frequency = 330  # E4 note
            duration = 0.2
        elif gesture_type == "open_hand":
            frequency = 659  # E5 note
            duration = 0.4
        else:
            frequency = 220  # A3 note
            duration = 0.2
        
        # Create a simple sine wave tone
        sample_rate = 22050
        frames = int(duration * sample_rate)
        arr = np.zeros((frames, 2))
        
        for i in range(frames):
            wave = 4096 * np.sin(2 * np.pi * frequency * i / sample_rate)
            arr[i][0] = wave  # Left channel
            arr[i][1] = wave  # Right channel
        
        sound = pygame.sndarray.make_sound(arr.astype(np.int16))
        sound.play()
        print(f"🎵 Gesture detected: {gesture_type.upper()}")
        
    except Exception as e:
        print(f"Error playing sound: {e}")

def detect_gesture(landmarks):
    """Detect hand gestures based on MediaPipe landmarks."""
    if not landmarks:
        return None
    
    # Get key points
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]  # Thumb metacarpal
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    
    # Check if fingers are extended
    fingers = []
    
    # Thumb detection - more robust for thumbs up
    thumb_extended = (thumb_tip.x > thumb_ip.x and thumb_tip.y < thumb_ip.y) or \
                    (thumb_tip.x > thumb_mcp.x and thumb_tip.y < thumb_mcp.y)
    fingers.append(1 if thumb_extended else 0)
    
    # Other fingers (compare y coordinates)
    for tip, pip in [(index_tip, index_pip), (middle_tip, middle_pip), 
                     (ring_tip, ring_pip), (pinky_tip, pinky_pip)]:
        if tip.y < pip.y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    # Gesture recognition
    if fingers == [1, 0, 0, 0, 0]:  # Only thumb up (proper thumbs up)
        return "thumbs_up"
    elif fingers == [0, 1, 1, 0, 0]:  # Index and middle up
        return "peace"
    elif fingers == [0, 0, 0, 0, 0]:  # All fingers down
        return "fist"
    elif fingers == [1, 1, 1, 1, 1]:  # All fingers up
        return "open_hand"
    elif fingers == [1, 1, 0, 0, 0]:  # Thumb and index up (alternative thumbs up)
        return "thumbs_up"
    else:
        return None

def find_available_camera():
    """Find an available camera that's not being used by streaming software."""
    print("Searching for available camera...")
    
    # Try different camera indices
    for cam_id in range(10):  # Try cameras 0-9
        print(f"Trying camera {cam_id}...")
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                # Check if frame is not completely black
                if np.mean(frame) > 10:  # If average pixel value > 10, it's not black
                    print(f"✓ Found working camera: {cam_id}")
                    return cap, cam_id
                else:
                    print(f"✗ Camera {cam_id} returns black frames (likely in use)")
            cap.release()
        else:
            cap.release()
    
    print("✗ No available cameras found")
    return None, None

def main():
    """Main application function for streaming compatibility."""
    print("Streaming-Compatible Eye Tracking & Gesture Detection")
    print("=" * 60)
    print("This version is designed to work alongside OBS/streaming software")
    print()
    
    # Initialize audio
    audio_available = initialize_audio()
    if audio_available:
        print("✓ Audio system initialized")
    else:
        print("⚠ Audio system unavailable - gestures will be detected but no sound will play")
    
    # OBS Configuration
    OBS_HOST = "localhost"
    OBS_PORT = 4455
    OBS_PASSWORD = "mysecret"  # Set this if you have a password
    
    # Scene names (must match your OBS scenes exactly)
    SCENES = {
        "left": "left cam",      # Webcam scene
        "center": "center cam",  # Front camera scene
        "right": "center cam"    # Front camera scene
    }
    
    # Initialize OBS connection
    obs_client = None
    obs_connected = False
    
    try:
        print("Connecting to OBS...")
        if OBS_PASSWORD:
            obs_client = obs.ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)
        else:
            obs_client = obs.ReqClient(host=OBS_HOST, port=OBS_PORT)
        
        # Test connection
        version_info = obs_client.get_version()
        print(f"✓ Connected to OBS {version_info.obs_version}")
        obs_connected = True
        
        # Get current scene
        current_scene = obs_client.get_current_program_scene()
        print(f"✓ Current OBS scene: {current_scene.current_program_scene_name}")
        
    except Exception as e:
        print(f"⚠ OBS connection failed: {e}")
        print("Continuing without OBS integration...")
        obs_connected = False

    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Find available camera
    cap, camera_id = find_available_camera()
    if cap is None:
        print("❌ No cameras available! Make sure:")
        print("   1. You have a camera connected")
        print("   2. The camera is not being used by OBS or other software")
        print("   3. Try closing OBS temporarily to test")
        input("Press Enter to exit...")
        return

    print(f"✓ Using camera {camera_id} for detection")
    print("✓ Eye tracking and gesture detection active")
    print("✓ OBS scene switching enabled" if obs_connected else "⚠ OBS scene switching disabled")
    print()
    print("🎯 GESTURES:")
    print("   👍 Thumbs Up    → Switch to webcam scene")
    print("   ✌️ Peace Sign   → Switch to center scene") 
    print("   ✊ Fist         → Toggle between scenes")
    print("   ✋ Open Hand    → Reset to center scene")
    print()
    print("👁️ EYE TRACKING:")
    print("   Look LEFT  → Switch to webcam scene")
    print("   Look RIGHT → Switch to center scene")
    print()
    print("Press 'q' to quit, 's' to show/hide camera window")

    # Variables
    frame_count = 0
    gaze_counter = 0
    last_gaze_direction = "CENTER/RIGHT"
    last_gesture = None
    gesture_counter = 0
    last_gesture_time = 0
    show_window = True  # Start with window visible
    
    current_camera_scene = "center"  # Track current scene
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera")
            break
        
        frame_count += 1
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for face detection
        results = face_mesh.process(rgb_frame)
        
        # Process the frame for hand detection
        hand_results = hands.process(rgb_frame)
        
        # Eye tracking logic
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # Get eye landmarks for gaze detection
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            left_iris_indices = [468, 469, 470, 471, 472]
            right_iris_indices = [473, 474, 475, 476, 477]
            left_eye_corners = [33, 133]
            right_eye_corners = [362, 263]
            
            try:
                # Extract iris centers
                left_iris_center = [0, 0]
                right_iris_center = [0, 0]
                
                for idx in left_iris_indices:
                    landmark = face_landmarks.landmark[idx]
                    left_iris_center[0] += landmark.x * w
                    left_iris_center[1] += landmark.y * h
                
                for idx in right_iris_indices:
                    landmark = face_landmarks.landmark[idx]
                    right_iris_center[0] += landmark.x * w
                    right_iris_center[1] += landmark.y * h
                
                left_iris_center[0] = int(left_iris_center[0] / len(left_iris_indices))
                left_iris_center[1] = int(left_iris_center[1] / len(left_iris_indices))
                right_iris_center[0] = int(right_iris_center[0] / len(right_iris_indices))
                right_iris_center[1] = int(right_iris_center[1] / len(right_iris_indices))
                
                # Calculate gaze direction
                avg_horizontal_ratio = 0.5
                if len(left_eye_corners) >= 2 and len(right_eye_corners) >= 2:
                    # Simplified gaze calculation
                    left_corner = min([face_landmarks.landmark[idx] for idx in left_eye_corners], key=lambda p: p.x)
                    right_corner = max([face_landmarks.landmark[idx] for idx in left_eye_corners], key=lambda p: p.x)
                    eye_width = right_corner.x - left_corner.x
                    if eye_width > 0:
                        iris_offset = (left_iris_center[0] / w) - left_corner.x
                        avg_horizontal_ratio = iris_offset / eye_width
                
                # Determine gaze direction
                if avg_horizontal_ratio < 0.4:
                    gaze_direction = "LEFT"
                    target_scene = "left"
                else:
                    gaze_direction = "CENTER/RIGHT"
                    target_scene = "center"
                
                # Count stable gaze direction
                if gaze_direction == last_gaze_direction:
                    gaze_counter += 1
                else:
                    gaze_counter = 0
                    last_gaze_direction = gaze_direction
                
                # Switch OBS scene if needed
                current_time = time.time()
                if (target_scene != current_camera_scene and 
                    current_time - last_gesture_time > 2.0 and 
                    gaze_counter > 10 and
                    obs_connected):
                    current_camera_scene = target_scene
                    last_gesture_time = current_time
                    gaze_counter = 0
                    scene_name = SCENES[target_scene]
                    print(f"👁️ Eye tracking: Switching to {scene_name} (Gaze: {gaze_direction})")
                    
                    try:
                        obs_client.set_current_program_scene(scene_name)
                    except Exception as e:
                        print(f"✗ OBS scene switch failed: {e}")
                
                # Draw debug info
                cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Scene: {current_camera_scene}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Error in gaze detection: {e}")
        else:
            cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Process hand gestures
        current_gesture = None
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Detect gesture
                gesture = detect_gesture(hand_landmarks.landmark)
                if gesture:
                    current_gesture = gesture
                    break
        
        # Handle gesture detection and OBS switching
        current_time = time.time()
        if current_gesture:
            if current_gesture == last_gesture:
                gesture_counter += 1
            else:
                gesture_counter = 1
                last_gesture = current_gesture
            
            # Process gesture after it's stable for 5 frames
            if gesture_counter >= 5 and current_time - last_gesture_time > 1.0:
                last_gesture_time = current_time
                
                # Play sound if audio is available
                if audio_available:
                    play_gesture_sound(current_gesture)
                
                # Handle OBS scene switching based on gestures
                if obs_connected:
                    try:
                        if current_gesture == "thumbs_up":
                            scene_name = SCENES["left"]
                            current_camera_scene = "left"
                            print(f"👍 Thumbs up: Switching to {scene_name}")
                        elif current_gesture == "peace":
                            scene_name = SCENES["center"]
                            current_camera_scene = "center"
                            print(f"✌️ Peace sign: Switching to {scene_name}")
                        elif current_gesture == "fist":
                            # Toggle between scenes
                            if current_camera_scene == "center":
                                scene_name = SCENES["left"]
                                current_camera_scene = "left"
                            else:
                                scene_name = SCENES["center"]
                                current_camera_scene = "center"
                            print(f"✊ Fist: Toggling to {scene_name}")
                        elif current_gesture == "open_hand":
                            scene_name = SCENES["center"]
                            current_camera_scene = "center"
                            print(f"✋ Open hand: Resetting to {scene_name}")
                        
                        obs_client.set_current_program_scene(scene_name)
                        
                    except Exception as e:
                        print(f"✗ OBS scene switch failed: {e}")
            
            # Display gesture info
            cv2.putText(frame, f"GESTURE: {current_gesture.upper()}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            last_gesture = None
            gesture_counter = 0
        
        # Add status info
        cv2.putText(frame, f"Camera: {camera_id}", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'q' to quit, 's' to toggle window", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show window if enabled
        if show_window:
            cv2.imshow("Streaming Eye Tracking & Gestures", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_window = not show_window
            if show_window:
                print("📺 Camera window shown")
            else:
                print("📺 Camera window hidden")
                cv2.destroyAllWindows()
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # Disconnect from OBS
    if obs_connected and obs_client:
        try:
            obs_client.disconnect()
            print("✓ Disconnected from OBS")
        except:
            pass
    
    print("✓ Application closed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Application interrupted by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
