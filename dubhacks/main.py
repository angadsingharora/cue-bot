"""
Eye Tracking Camera Switcher - Main Application
Switches between cameras based on eye gaze direction.
"""

import cv2
import mediapipe as mp
import time
import obsws_python as obs

def main():
    """Main application function."""
    print("Eye Tracking Camera Switcher with OBS Integration")
    print("=" * 50)
    print("Starting Eye Tracking Camera Switcher...")
    
    try:
        print("DEBUG: Starting main function...")
        # OBS Configuration
        OBS_HOST = "localhost"
        OBS_PORT = 4455
        OBS_PASSWORD = "mysecret"  # Set this if you have a password
        
        # Scene names (must match your OBS scenes exactly)
        # Update these to match your actual OBS scene names
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
            
            # List all available scenes
            try:
                scenes = obs_client.get_scene_list()
                print("Available OBS scenes:")
                for scene in scenes.scenes:
                    print(f"  - {scene['sceneName']}")
            except Exception as e:
                print(f"Could not list scenes: {e}")
            
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

        # Try to open both cameras with different backends
        print("Opening cameras...")
        print("NOTE: Make sure OBS is using different cameras than this script!")
        cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Front camera with DirectShow
        cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Webcam with DirectShow
        
        # If DirectShow fails, try without backend
        if not cap0.isOpened():
            print("DirectShow failed for camera 0, trying default backend...")
            cap0 = cv2.VideoCapture(0)
        if not cap1.isOpened():
            print("DirectShow failed for camera 1, trying default backend...")
            cap1 = cv2.VideoCapture(1)

        # Check which cameras work
        cameras = []
        
        # Test camera 0
        if cap0.isOpened():
            ret, frame = cap0.read()
            if ret and frame is not None:
                cameras.append(("Camera 0 (Front)", cap0))
                print("✓ Camera 0 (Front) working")
            else:
                print("✗ Camera 0 opened but cannot read frames")
                cap0.release()
        else:
            print("✗ Camera 0 failed to open")
            cap0.release()

        # Test camera 1
        if cap1.isOpened():
            ret, frame = cap1.read()
            if ret and frame is not None:
                cameras.append(("Camera 1 (Webcam)", cap1))
                print("✓ Camera 1 (Webcam) working")
            else:
                print("✗ Camera 1 opened but cannot read frames")
                cap1.release()
        else:
            print("✗ Camera 1 failed to open")
            cap1.release()

        # If we don't have 2 cameras, try other camera IDs
        if len(cameras) < 2:
            print("Trying other cameras...")
            for cam_id in [2, 3, 4, 5]:
                print(f"Trying camera {cam_id}...")
                cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cameras.append((f"Camera {cam_id}", cap))
                        print(f"✓ Camera {cam_id} working")
                        if len(cameras) >= 2:
                            break
                    else:
                        cap.release()
                else:
                    cap.release()
        
        if len(cameras) < 1:
            print("✗ No cameras available")
            return
        
        if len(cameras) < 2:
            print("⚠ Only 1 camera available - will use it for both eye tracking and display")
            print("Available cameras:", len(cameras))
            # Use the same camera for both eye tracking and display
            eye_camera = cameras[0][1]
            current_camera = 0
            last_switch_time = time.time()
            print("✓ Using camera for both eye tracking and display:", cameras[0][0])
            print("Press 'q' to quit")
            
            # Single camera mode - just show eye tracking
            while True:
                ret, frame = eye_camera.read()
                if not ret:
                    break
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    cv2.putText(frame, "FACE DETECTED - SINGLE CAMERA MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Eye Tracking Camera Switcher", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # Clean up
            for _, cap in cameras:
                cap.release()
            cv2.destroyAllWindows()
            return

        # Use front camera (camera 0) for eye tracking, webcam (camera 1) for display
        eye_camera = cameras[0][1]  # Front camera for eye tracking
        current_camera = 0  # Start with front camera
        last_switch_time = time.time()

        print("✓ Eye tracking camera:", cameras[0][0])
        print("✓ Display camera:", cameras[1][0])
        print("Press 'q' to quit")

        frame_count = 0
        gaze_counter = 0
        last_gaze_direction = "CENTER/RIGHT"
        
        while True:
            # Capture frame from eye tracking camera
            ret, frame = eye_camera.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = face_mesh.process(rgb_frame)
            
            # Check if face is detected
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w, _ = frame.shape
                
                # Get eye landmarks for gaze detection
                # Left eye landmarks
                left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                # Right eye landmarks  
                right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                
                # Iris landmarks
                left_iris_indices = [468, 469, 470, 471, 472]
                right_iris_indices = [473, 474, 475, 476, 477]
                
                # Eye corner landmarks
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
                    if avg_horizontal_ratio < 0.4:  # Looking left (more sensitive)
                        gaze_direction = "LEFT"
                        target_camera = 1  # Show webcam when looking left
                    else:  # Looking center/right
                        gaze_direction = "CENTER/RIGHT"
                        target_camera = 0  # Show front camera when looking center/right
                    
                    # Count stable gaze direction
                    if gaze_direction == last_gaze_direction:
                        gaze_counter += 1
                    else:
                        gaze_counter = 0
                        last_gaze_direction = gaze_direction
                    
                    # Switch camera if needed (only if gaze is stable for 10 frames)
                    current_time = time.time()
                    if (target_camera != current_camera and 
                        current_time - last_switch_time > 2.0 and 
                        gaze_counter > 10):  # Wait 2 seconds AND stable gaze for 10 frames
                        current_camera = target_camera
                        last_switch_time = current_time
                        gaze_counter = 0
                        print(f"SWITCHING TO {cameras[current_camera][0]} (Gaze: {gaze_direction}, Ratio: {avg_horizontal_ratio:.2f})")
                        
                        # Switch OBS scene if connected
                        if obs_connected and obs_client:
                            try:
                                # Map gaze direction to scene name
                                if gaze_direction == "LEFT":
                                    scene_name = SCENES["left"]
                                else:
                                    scene_name = SCENES["center"]
                                
                                print(f"Attempting to switch OBS to scene: {scene_name}")
                                obs_client.set_current_program_scene(scene_name)
                                print(f"✓ OBS switched to scene: {scene_name}")
                                
                            except Exception as e:
                                print(f"✗ OBS scene switch failed: {e}")
                                print(f"  Tried to switch to: {scene_name}")
                                print(f"  Available scenes: {[scene['sceneName'] for scene in obs_client.get_scene_list().scenes]}")
                    
                    # Draw debug info
                    cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Current Camera: {cameras[current_camera][0]}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Target Camera: {cameras[target_camera][0]}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Ratio: {avg_horizontal_ratio:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Show if switching is needed
                    if target_camera != current_camera:
                        cv2.putText(frame, "SWITCHING NEEDED!", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Draw iris centers
                    cv2.circle(frame, tuple(left_iris_center), 5, (0, 255, 0), -1)
                    cv2.circle(frame, tuple(right_iris_center), 5, (0, 255, 0), -1)
                    
                except Exception as e:
                    print(f"Error in gaze detection: {e}")
                    cv2.putText(frame, "FACE DETECTED - GAZE ERROR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Reset gaze tracking variables to prevent crashes
                    gaze_direction = "CENTER/RIGHT"
                    target_camera = 0
            else:
                cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame from the current camera (not just eye tracking camera)
            if current_camera == 0:
                display_frame = frame  # Show front camera
            else:
                # Get frame from webcam
                ret_webcam, webcam_frame = cameras[1][1].read()
                if ret_webcam:
                    display_frame = webcam_frame
                else:
                    display_frame = frame  # Fallback to eye tracking camera
            
            # Add camera info to display frame
            cv2.putText(display_frame, f"ACTIVE CAMERA: {cameras[current_camera][0]}", (10, display_frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add OBS status
            obs_status = "OBS: Connected" if obs_connected else "OBS: Disconnected"
            obs_color = (0, 255, 0) if obs_connected else (0, 0, 255)
            cv2.putText(display_frame, obs_status, (10, display_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, obs_color, 2)
            
            # Add current OBS scene if connected
            if obs_connected and obs_client:
                try:
                    current_obs_scene = obs_client.get_current_program_scene()
                    cv2.putText(display_frame, f"OBS Scene: {current_obs_scene.current_program_scene_name}", (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                except:
                    pass
            
            cv2.imshow("Eye Tracking Camera Switcher", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
    
    finally:
        # Clean up
        print("Cleaning up...")
        try:
            for _, cap in cameras:
                cap.release()
            cv2.destroyAllWindows()
            
            # Disconnect from OBS
            if obs_connected and obs_client:
                try:
                    obs_client.disconnect()
                    print("✓ Disconnected from OBS")
                except:
                    pass
        except:
            pass
        
        print("Done!")

if __name__ == "__main__":
    print("Starting from __main__ block...")
    main()