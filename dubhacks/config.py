"""
Configuration file for Eye Tracking Camera Switcher
"""

# OBS WebSocket Configuration
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = "mysecret"

# Scene names (must match your OBS scenes exactly)
SCENES = {
    "center": "center cam",
    "left": "left cam", 
    "right": "center cam",
    "down": "center cam"
}
