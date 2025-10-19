# OBS Setup Guide

## Prerequisites

1. **Install OBS Studio** from https://obsproject.com/
2. **Enable WebSocket Server** in OBS:
   - Go to `Tools` → `WebSocket Server Settings`
   - Check "Enable WebSocket Server"
   - Set port to `4455` (default)
   - Leave password empty (or set one and update `main.py`)

## Scene Setup

Create these scenes in OBS with exact names:

### 1. "center cam" Scene
- **Purpose**: Front camera view (when looking center/right)
- **Setup**: Add your front camera as a video source
- **Name**: Must be exactly "center cam"

### 2. "left cam" Scene  
- **Purpose**: Webcam view (when looking left)
- **Setup**: Add your webcam as a video source
- **Name**: Must be exactly "left cam"

## Configuration

In `main.py`, you can modify these settings:

```python
# OBS Configuration
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = ""  # Set this if you have a password

# Scene names (must match your OBS scenes exactly)
SCENES = {
    "left": "left cam",      # Webcam scene
    "center": "center cam",  # Front camera scene
    "right": "center cam"    # Front camera scene
}
```

## Testing

1. **Start OBS Studio**
2. **Create the scenes** as described above
3. **Run the eye tracking app**: `python main.py`
4. **Look left** → should switch to "left cam" scene
5. **Look center/right** → should switch to "center cam" scene

## Troubleshooting

### "OBS connection failed"
- Make sure OBS is running
- Check WebSocket server is enabled
- Verify port 4455 is not blocked

### "OBS scene switch failed"
- Check scene names match exactly (case-sensitive)
- Make sure scenes exist in OBS
- Verify WebSocket connection is stable

### Scenes not switching
- Check console output for error messages
- Verify gaze detection is working (should show "LEFT" or "CENTER/RIGHT")
- Make sure the 2-second delay has passed

## Twitch Integration

Once OBS is set up, you can stream to Twitch:

1. **In OBS**: Go to `Settings` → `Stream`
2. **Service**: Select "Twitch"
3. **Stream Key**: Get from your Twitch dashboard
4. **Start Streaming**: Click "Start Streaming" in OBS

The eye tracking will automatically switch scenes while you're live on Twitch!
