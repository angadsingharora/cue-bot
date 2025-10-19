# Eye Tracking Camera Switcher with OBS Integration

A Python application that tracks your eyes in real-time using MediaPipe and automatically switches between cameras and OBS scenes based on your gaze direction. Perfect for streaming on Twitch!

## Features

- **Real-time eye tracking** using MediaPipe face mesh
- **Automatic camera switching** based on gaze direction
- **OBS scene switching** for live streaming
- **Twitch integration** - stream with automatic scene switching
- **Stable switching** with debouncing to prevent flickering
- **Debug display** showing gaze direction, camera status, and OBS connection

## How It Works

- **Eye Tracking**: Uses front camera (Camera 0) to detect your gaze direction
- **Left Gaze**: Switches to webcam (Camera 1) and OBS "left cam" scene when you look left
- **Center/Right Gaze**: Switches to front camera (Camera 0) and OBS "center cam" scene when you look center or right
- **OBS Integration**: Automatically switches OBS scenes for live streaming
- **Stable Switching**: Waits for stable gaze direction before switching to prevent rapid changes

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install OBS Studio** from https://obsproject.com/

3. **Make sure you have at least 2 cameras:**
   - Camera 0: Front camera (for eye tracking)
   - Camera 1: Webcam (for display when looking left)

4. **Set up OBS scenes** (see `OBS_SETUP.md` for detailed instructions):
   - Create "center cam" scene with front camera
   - Create "left cam" scene with webcam
   - Enable WebSocket Server in OBS

## Usage

1. **Start OBS Studio** and set up your scenes (see `OBS_SETUP.md`)

2. **Run the application:**
   ```bash
   python main.py
   ```

3. **Position yourself** in front of the front camera

4. **Look in different directions:**
   - Look **left** → switches to webcam view and OBS "left cam" scene
   - Look **center/right** → switches to front camera view and OBS "center cam" scene

5. **Start streaming** in OBS to go live on Twitch

6. **Press 'q'** to quit

## Controls

- `q` - Quit application

## Debug Information

The application displays:
- **Gaze Direction**: LEFT or CENTER/RIGHT
- **Current Camera**: Which camera is currently showing
- **Target Camera**: Which camera it wants to switch to
- **Ratio**: Gaze detection ratio (lower = more left)
- **Active Camera**: Which camera is currently active
- **OBS Status**: Connected/Disconnected
- **OBS Scene**: Current OBS scene name

## Requirements

- **2 Cameras**: Front camera and webcam
- **OBS Studio**: For scene switching and streaming
- **Good Lighting**: Ensure your face is well-lit
- **Python 3.7+**
- **MediaPipe**, **OpenCV**, and **OBS WebSocket**

## Troubleshooting

### "Need at least 2 cameras"
- Make sure both cameras are connected and working
- Check camera permissions in Windows settings

### "NO FACE DETECTED"
- Ensure good lighting on your face
- Position yourself directly in front of the front camera
- Make sure the front camera is not blocked

### "OBS connection failed"
- Make sure OBS Studio is running
- Check WebSocket server is enabled in OBS
- Verify port 4455 is not blocked

### "OBS scene switch failed"
- Check scene names match exactly (case-sensitive)
- Make sure scenes exist in OBS
- Verify WebSocket connection is stable

### Rapid switching between cameras
- The app has built-in debouncing (2-second delay + 10-frame stability)
- This prevents rapid switching and ensures stable operation

## File Structure

```
├── main.py              # Main application with OBS integration
├── requirements.txt     # Python dependencies
├── OBS_SETUP.md        # Detailed OBS setup guide
└── README.md           # This file
```

## Dependencies

- `mediapipe==0.10.7` - Face mesh and landmark detection
- `opencv-python==4.8.1.78` - Video capture and display
- `obs-websocket-py==1.5.3` - OBS WebSocket integration

## License

This project is open source. Feel free to modify and distribute as needed.