# Gesture Recognition Guide

## Overview
The application now includes gesture recognition with audio feedback! Make simple hand gestures to the camera and hear different sounds for each gesture.

## Supported Gestures

### 1. Thumbs Up 👍
- **Gesture**: Thumb and index finger extended, other fingers closed
- **Sound**: A4 note (440 Hz) - Pleasant ascending tone
- **Use**: Positive feedback, approval

### 2. Peace Sign ✌️
- **Gesture**: Index and middle finger extended, other fingers closed
- **Sound**: C5 note (523 Hz) - Higher pitched tone
- **Use**: Victory, peace, or just for fun

### 3. Fist ✊
- **Gesture**: All fingers closed into a fist
- **Sound**: E4 note (330 Hz) - Lower, shorter tone
- **Use**: Determination, power

### 4. Open Hand ✋
- **Gesture**: All five fingers extended
- **Sound**: E5 note (659 Hz) - Higher, longer tone
- **Use**: Stop, high five, or greeting

## How It Works

1. **Detection**: The camera continuously monitors for hand gestures using MediaPipe
2. **Recognition**: When a gesture is detected and held stable for 5 frames, it's recognized
3. **Audio Feedback**: A unique sound plays for each gesture type
4. **Cooldown**: There's a 1-second cooldown between sounds to prevent spam

## Tips for Best Results

- **Lighting**: Ensure good lighting on your hands
- **Distance**: Keep your hands about 1-2 feet from the camera
- **Stability**: Hold gestures steady for a moment to trigger the sound
- **Background**: Use a contrasting background for better hand detection

## Technical Details

- Uses MediaPipe for hand landmark detection
- Pygame for audio generation and playback
- Gesture recognition based on finger joint positions
- Real-time processing with OpenCV

## Troubleshooting

- **No sound**: Check if pygame is properly installed
- **Gestures not detected**: Ensure good lighting and hand positioning
- **False positives**: Adjust hand position or lighting conditions

Enjoy making gestures and hearing the sounds! 🎵
