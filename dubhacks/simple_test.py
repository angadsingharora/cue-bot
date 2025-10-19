"""
Simple test to check if everything is working
"""
import cv2
import mediapipe as mp
import pygame
import numpy as np

print("Testing imports...")
print("✓ OpenCV imported")
print("✓ MediaPipe imported") 
print("✓ Pygame imported")
print("✓ NumPy imported")

print("\nTesting camera...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✓ Camera is available")
    ret, frame = cap.read()
    if ret:
        print("✓ Camera can read frames")
        print(f"Frame shape: {frame.shape}")
    else:
        print("✗ Camera cannot read frames")
    cap.release()
else:
    print("✗ Camera is not available")

print("\nTesting MediaPipe hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
print("✓ MediaPipe hands initialized")

print("\nTesting pygame audio...")
try:
    pygame.mixer.init()
    print("✓ Pygame audio initialized")
except Exception as e:
    print(f"✗ Pygame audio failed: {e}")

print("\nAll tests completed!")
