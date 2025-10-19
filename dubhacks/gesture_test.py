"""
Simple gesture detection test script
"""
import cv2
import mediapipe as mp
import pygame
import numpy as np

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
        print(f"🎵 Playing sound for {gesture_type}")
        
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
    # Check if thumb is extended outward (both x and y coordinates)
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

def main():
    print("Gesture Detection Test")
    print("=" * 30)
    
    # Initialize audio
    audio_available = initialize_audio()
    if audio_available:
        print("✓ Audio system initialized")
    else:
        print("⚠ Audio system unavailable")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully")
    print("Show gestures to the camera:")
    print("- Thumbs up (thumb + index finger)")
    print("- Peace sign (index + middle finger)")
    print("- Fist (all fingers down)")
    print("- Open hand (all fingers up)")
    print("Press 'q' to quit")
    
    last_gesture = None
    gesture_counter = 0
    last_gesture_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        
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
        
        # Handle gesture detection and audio feedback
        import time
        current_time = time.time()
        if current_gesture:
            if current_gesture == last_gesture:
                gesture_counter += 1
            else:
                gesture_counter = 1
                last_gesture = current_gesture
            
            # Play sound if gesture is stable for 5 frames and enough time has passed
            if (gesture_counter >= 5 and 
                current_time - last_gesture_time > 1.0 and 
                audio_available):
                play_gesture_sound(current_gesture)
                last_gesture_time = current_time
            
            # Display gesture info
            cv2.putText(frame, f"GESTURE: {current_gesture.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            last_gesture = None
            gesture_counter = 0
            cv2.putText(frame, "NO GESTURE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Gesture Detection Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed!")

if __name__ == "__main__":
    main()
