"""
Test script to verify OBS WebSocket connection and scene switching
"""

from obsws_python import ReqClient
import time

HOST = "localhost"
PORT = 4455
PASSWORD = "mysecret"  # Change this to match your OBS password

try:
    print("Connecting to OBS...")
    client = ReqClient(host=HOST, port=PORT, password=PASSWORD)
    print("✅ Connected to OBS")

    # Get current scene
    current_scene = client.get_current_program_scene()
    print(f"Current scene: {current_scene.current_program_scene_name}")

    # List all scenes
    scenes = client.get_scene_list()
    print("Available scenes:")
    for scene in scenes.scenes:
        print(f"  - {scene['sceneName']}")

    # Test switching between scenes
    print("\nTesting scene switching...")
    for i in range(3):
        print(f"\nTest {i+1}/3:")
        
        # Switch to center cam
        client.set_current_program_scene("center cam")
        print("✅ Switched to center cam")
        time.sleep(2)
        
        # Switch to left cam
        client.set_current_program_scene("left cam")
        print("✅ Switched to left cam")
        time.sleep(2)

    print("\n✅ Test complete - OBS connection is working!")

except Exception as e:
    print(f"❌ Failed to connect or switch scenes: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure OBS is running")
    print("2. Check that WebSocket Server is enabled in OBS (Tools → WebSocket Server Settings)")
    print("3. Verify the password matches your OBS settings")
    print("4. Make sure you have scenes named 'Cam_Center' and 'Cam_Left' in OBS")
