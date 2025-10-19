"""
Clean Eye Tracking Camera Switcher - OBS Only
Handles gaze detection and OBS scene switching only.
OBS owns all camera feeds.
"""

import cv2
import mediapipe as mp
import time
import obsws_python as obs
from config import OBS_HOST, OBS_PORT, OBS_PASSWORD, SCENES

def main():
    """Main application function - gaze detection + OBS control only."""
    print("Eye Tracking Camera Switcher - OBS Control Only")
    print("=" * 50)
    
    # Initialize OBS connection
    obs_client = None
    obs_connected = False
    
    try:
        print("Connecting to OBS...")
        obs_client = obs.ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)
        version_info = obs_client.get_version()
        print(f"✓ Connected to OBS {version_info.obs_version}")
        obs_connected = True
        
        # Get current scene
        current_scene = obs_client.get_current_program_scene()
        print(f"✓ Current OBS scene: {current_scene.current_program_scene_name}")
        
    except Exception as e:
        print(f"⚠ OBS connection failed: {e}")
        print("Continuing without OBS - you can still see eye tracking")
        obs_connected = False

    # Initialize MediaPipe for gaze detection
    print("Initializing gaze detection...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    # Open camera for gaze detection only (NO DISPLAY WINDOWS)
    print("Opening camera for gaze detection...")
    
    # Use real cameras for Python (OBS will use virtual cameras)
    cap = None
    for camera_id in [0, 1, 2]:  # Try real cameras first
        print(f"Trying camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_brightness = frame.mean()
                print(f"Camera {camera_id} brightness: {frame_brightness}")
                if frame_brightness > 10:  # Check for actual content
                    print(f"✓ Using camera {camera_id} for gaze detection")
                    break
                else:
                    print(f"Camera {camera_id} too dark")
            else:
                print(f"Camera {camera_id} can't read frames")
            cap.release()
            cap = None
        else:
            print(f"Camera {camera_id} failed to open")
            cap.release()
            cap = None
    
    if not cap or not cap.isOpened():
        print("Failed to open any camera for gaze detection")
        return
    
    print("✓ Camera opened for gaze detection")
    print("✓ OBS handles all video display")
    print("Press 'q' to quit")

    # Gaze tracking variables
    last_switch_time = time.time()
    gaze_counter = 0
    last_gaze_direction = "CENTER/RIGHT"
    current_obs_scene = "center cam"  # Track current OBS scene
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        gaze_direction = "NO_FACE"
        target_scene = "center cam"
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # Eye landmarks for gaze detection
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
                
                # Extract eye corners
                left_corners = []
                right_corners = []
                
                for idx in left_eye_corners:
                    landmark = face_landmarks.landmark[idx]
                    left_corners.append([int(landmark.x * w), int(landmark.y * h)])
                
                for idx in right_eye_corners:
                    landmark = face_landmarks.landmark[idx]
                    right_corners.append([int(landmark.x * w), int(landmark.y * h)])
                
                # Calculate horizontal gaze ratio
                left_horizontal_ratio = 0.5
                right_horizontal_ratio = 0.5
                
                if len(left_corners) >= 2:
                    left_corner = min(left_corners, key=lambda p: p[0])
                    right_corner = max(left_corners, key=lambda p: p[0])
                    eye_width = right_corner[0] - left_corner[0]
                    if eye_width > 0:
                        iris_offset = left_iris_center[0] - left_corner[0]
                        left_horizontal_ratio = iris_offset / eye_width
                
                if len(right_corners) >= 2:
                    left_corner = min(right_corners, key=lambda p: p[0])
                    right_corner = max(right_corners, key=lambda p: p[0])
                    eye_width = right_corner[0] - left_corner[0]
                    if eye_width > 0:
                        iris_offset = right_iris_center[0] - left_corner[0]
                        right_horizontal_ratio = iris_offset / eye_width
                
                # Average the ratios
                avg_horizontal_ratio = (left_horizontal_ratio + right_horizontal_ratio) / 2
                
                # Determine gaze direction
                if avg_horizontal_ratio < 0.4:  # Looking left
                    gaze_direction = "LEFT"
                    target_scene = "left cam"
                else:  # Looking center/right
                    gaze_direction = "CENTER/RIGHT"
                    target_scene = "center cam"
                
                # Count stable gaze direction
                if gaze_direction == last_gaze_direction:
                    gaze_counter += 1
                else:
                    gaze_counter = 0
                    last_gaze_direction = gaze_direction
                
                # Switch OBS scene if needed (only if OBS is connected)
                if obs_connected:
                    current_time = time.time()
                    if (target_scene != current_obs_scene and 
                        current_time - last_switch_time > 2.0 and 
                        gaze_counter > 10):
                        
                        try:
                            obs_client.set_current_program_scene(target_scene)
                            print(f"✓ OBS switched to: {target_scene} (Gaze: {gaze_direction}, Ratio: {avg_horizontal_ratio:.2f})")
                            current_obs_scene = target_scene
                            last_switch_time = current_time
                            gaze_counter = 0
                        except Exception as e:
                            print(f"✗ OBS scene switch failed: {e}")
                
                # Draw debug info on frame
                cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Target: {target_scene}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Current: {current_obs_scene}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Ratio: {avg_horizontal_ratio:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw iris centers
                cv2.circle(frame, tuple(left_iris_center), 5, (0, 255, 0), -1)
                cv2.circle(frame, tuple(right_iris_center), 5, (0, 255, 0), -1)
                
                # Console output for debugging
                if frame_count % 30 == 0:  # Print every 30 frames to reduce spam
                    print(f"Gaze: {gaze_direction} | Target: {target_scene} | Current: {current_obs_scene} | Ratio: {avg_horizontal_ratio:.2f}")
                
            except Exception as e:
                print(f"Error in gaze detection: {e}")
                cv2.putText(frame, "FACE DETECTED - GAZE ERROR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if frame_count % 30 == 0:  # Print every 30 frames to reduce spam
                print("NO FACE DETECTED")
        
        frame_count += 1
        
        # Add instructions and OBS status
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show OBS connection status
        obs_status = "OBS: Connected" if obs_connected else "OBS: Disconnected"
        obs_color = (0, 255, 0) if obs_connected else (0, 0, 255)
        cv2.putText(frame, obs_status, (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, obs_color, 2)
        
        # Show the frame
        cv2.imshow("Eye Tracking Camera Switcher", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Clean up
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    
    if obs_connected and obs_client:
        try:
            obs_client.disconnect()
            print("✓ Disconnected from OBS")
        except:
            pass
    
    print("Done!")

if __name__ == "__main__":
    main()
