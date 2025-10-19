print("Testing imports...")
try:
    import cv2
    print("✓ OpenCV imported")
except Exception as e:
    print(f"✗ OpenCV import failed: {e}")

try:
    import mediapipe as mp
    print("✓ MediaPipe imported")
except Exception as e:
    print(f"✗ MediaPipe import failed: {e}")

try:
    import obsws_python as obs
    print("✓ OBS WebSocket imported")
except Exception as e:
    print(f"✗ OBS WebSocket import failed: {e}")

print("All imports successful!")
