# Streaming Setup Guide

## 🎯 **The Problem**
When streaming, OBS and other streaming software use your cameras, causing the Python script to get black screens or crash because multiple applications can't access the same camera simultaneously.

## ✅ **The Solution**
Use the `streaming_compatible.py` script which is designed to work alongside streaming software.

## 🚀 **Setup Instructions**

### **Step 1: Camera Setup**
1. **In OBS**: Set up your scenes with different cameras
   - Scene 1: "center cam" (your main camera)
   - Scene 2: "left cam" (your webcam/secondary camera)

2. **Camera Assignment**:
   - Make sure OBS is using specific cameras
   - The Python script will find an available camera automatically

### **Step 2: OBS WebSocket Setup**
1. **Enable OBS WebSocket**:
   - Go to Tools → WebSocket Server Settings
   - Enable WebSocket server
   - Set port to 4455 (default)
   - Set password to "mysecret" (or change it in the script)

2. **Test Connection**:
   - The script will automatically connect to OBS
   - You'll see "✓ Connected to OBS" in the console

### **Step 3: Run the Streaming Script**
```bash
cd dubhacks
python streaming_compatible.py
```

## 🎮 **How It Works**

### **Gesture Controls**:
- 👍 **Thumbs Up** → Switch to webcam scene
- ✌️ **Peace Sign** → Switch to center scene
- ✊ **Fist** → Toggle between scenes
- ✋ **Open Hand** → Reset to center scene

### **Eye Tracking**:
- 👁️ **Look LEFT** → Switch to webcam scene
- 👁️ **Look RIGHT** → Switch to center scene

### **Features**:
- ✅ **Automatic camera detection** - finds available cameras
- ✅ **OBS integration** - switches scenes automatically
- ✅ **Audio feedback** - plays sounds for gestures
- ✅ **Hidden mode** - press 's' to hide camera window
- ✅ **Streaming compatible** - works alongside OBS

## 🔧 **Troubleshooting**

### **Black Screen Issues**:
1. **Close OBS temporarily** to test if cameras work
2. **Check camera permissions** in Windows settings
3. **Try different camera indices** (the script tries 0-9 automatically)

### **OBS Connection Issues**:
1. **Enable WebSocket server** in OBS
2. **Check port 4455** is not blocked
3. **Verify password** matches in the script

### **Performance Issues**:
1. **Press 's'** to hide the camera window during streaming
2. **Lower camera resolution** in OBS if needed
3. **Close other applications** using cameras

## 📺 **Streaming Tips**

1. **Start OBS first**, then run the Python script
2. **Use 's' key** to hide the detection window during streaming
3. **Test gestures** before going live
4. **Have backup scenes** ready in OBS
5. **Monitor console output** for any errors

## 🎵 **Audio Settings**

- **For streaming**: Audio feedback will be heard by viewers
- **To disable audio**: Comment out the `play_gesture_sound()` calls
- **For private use**: Audio feedback helps you know when gestures are detected

## 🚀 **Ready to Stream!**

The script is now optimized for streaming and should work perfectly alongside OBS without camera conflicts!
