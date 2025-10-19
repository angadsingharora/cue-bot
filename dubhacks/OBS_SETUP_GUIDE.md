# OBS Setup Guide for Eye Tracking Camera Switcher

## 🎯 Goal
Set up OBS with two scenes that have actual camera sources, so you can see the camera switching when Python changes scenes.

## 📋 Step-by-Step Setup

### 1️⃣ Create First Scene (Cam_Center)

1. **Open OBS Studio**
2. **Create Scene:**
   - Click the "+" button under the "Scenes" panel
   - Name it: `Cam_Center`
   - Click "OK"

3. **Add Camera Source:**
   - Click the "+" button under the "Sources" panel
   - Select "Video Capture Device"
   - Name it: `Front Camera`
   - Click "OK"

4. **Configure Camera:**
   - In the "Properties" dialog, select your built-in laptop camera
   - Click "OK"
   - You should now see yourself in the preview

### 2️⃣ Create Second Scene (Cam_Left)

1. **Create Scene:**
   - Click the "+" button under the "Scenes" panel
   - Name it: `Cam_Left`
   - Click "OK"

2. **Add Camera Source:**
   - Click the "+" button under the "Sources" panel
   - Select "Video Capture Device"
   - Name it: `Left Camera`
   - Click "OK"

3. **Configure Camera:**
   - In the "Properties" dialog, select your external webcam
   - Click "OK"
   - You should now see the other camera angle

### 3️⃣ Enable WebSocket Server

1. **Open WebSocket Settings:**
   - Go to `Tools` → `WebSocket Server Settings`

2. **Configure Settings:**
   - ✅ Check "Enable WebSocket Server"
   - Port: `4455`
   - Password: `admin`
   - Click "OK"

### 4️⃣ Verify Setup

Your OBS should now have:

| Scene Name | Source Name | Camera |
|------------|-------------|---------|
| Cam_Center | Front Camera | Built-in laptop camera |
| Cam_Left | Left Camera | External webcam |

## 🧪 Test the Setup

Run the test script:
```bash
python test_switch_obs.py
```

**Expected Result:**
- OBS preview should switch between the two cameras every 2 seconds
- You should see yourself from different angles
- Console should show successful scene switches

## 🚨 Troubleshooting

### If you see black screens:
1. **Check camera sources:** Make sure each scene has a camera source added
2. **Verify camera selection:** Ensure the correct camera is selected in each source
3. **Check camera permissions:** Make sure OBS has permission to access your cameras

### If WebSocket connection fails:
1. **Check password:** Make sure the password in config.py matches OBS settings
2. **Check port:** Verify port 4455 is not blocked
3. **Restart OBS:** Sometimes OBS needs a restart after enabling WebSocket

### If scenes don't switch:
1. **Check scene names:** Ensure scene names match exactly (case-sensitive)
2. **Check WebSocket status:** Look for WebSocket connection indicator in OBS
3. **Check console output:** Look for error messages in the Python console

## ✅ Success Indicators

- ✅ OBS shows video from both cameras
- ✅ Python script connects to OBS successfully
- ✅ Scenes switch automatically every 2 seconds during test
- ✅ Eye tracking app switches scenes based on gaze direction
