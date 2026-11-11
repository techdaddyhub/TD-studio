# Deep-Live-Cam — Gold Studio Edition (`TD studio`)

Real-time AI face swapper & studio broadcaster featuring a **Binance Obsidian Black & Gold** UI, floating controls, responsive layout, desktop Virtual Camera output, and an Android companion APK.

## Easy Downloads (GitHub Releases & Actions)

Pre-built artifacts are generated automatically via [`.github/workflows/build-releases.yml`](.github/workflows/build-releases.yml):

| Platform | Artifact | Icon Asset |
| :--- | :--- | :--- |
| **Windows x64** | `Deep-Live-Cam-Windows-x64.zip` | `assets/icons/win/app_icon.ico` |
| **Linux x64** | `Deep-Live-Cam-Linux-x64.tar.gz` | `assets/icons/linux/app_icon.png` |
| **macOS x64** | `Deep-Live-Cam-macOS-x64.zip` | `assets/icons/mac/app_icon.icns` |
| **Android APK** | `Deep-Live-Cam-Android.apk` | `android/app/src/main/res/mipmap-*` |

## Desktop Virtual Camera Streaming

1. Install OBS Studio (provides **OBS Virtual Camera** on Windows/macOS) or `v4l2loopback` on Linux, plus:
   ```bash
   pip install pyvirtualcam
   ```
2. Toggle **Virtual Cam Output** ON inside **Live Camera & Stream** before or during **● Live**.
3. Select the virtual camera inside Zoom, Discord, OBS, TikTok Live Studio, WhatsApp Desktop, or Google Meet.

## Android APK Streaming & Social Media Setup

Because unrooted stock Android blocks third-party APKs from registering as a system-wide hardware camera inside Instagram/WhatsApp/TikTok mobile apps, the Android Gold Studio APK provides **three working streaming & sharing workflows**:

1. **Direct RTMP / RTMPS Broadcast**: Stream swapped video + mic audio straight to **YouTube Live, Twitch, Facebook Live, Kick, or OBS** using your stream key (`Stream` dock button).
2. **PC Virtual Cam Bridge**: Toggle **PC Virtual Cam Bridge** in the Android app (`http://<phone-ip>:8765/stream.mjpg`), click **`+ Bridge`** in Desktop Deep-Live-Cam, and turn on **Virtual Cam Output** to broadcast your phone's face-swapped feed into desktop social/video apps.
3. **Instant Gallery Snap**: Save swapped frames to `Pictures/DeepLiveCam` for Reels, Shorts, Stories, and WhatsApp.
