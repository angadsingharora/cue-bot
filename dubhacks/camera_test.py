#!/usr/bin/env python3
"""
Simple camera test
"""

import cv2

print("Testing camera access...")

try:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print(f"Camera opened: {cap.isOpened()}")
    
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"Frame read: {ret}")
        if ret:
            print(f"Frame shape: {frame.shape}")
            print("Camera is working!")
        else:
            print("Failed to read frame")
    else:
        print("Failed to open camera")
    
    cap.release()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete")
