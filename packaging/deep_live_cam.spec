# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Deep-Live-Cam (Windows x64, Linux x64, macOS x64)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

if sys.platform == "win32":
    ICON_PATH = os.path.join(ROOT, "assets", "icons", "win", "app_icon.ico")
elif sys.platform == "darwin":
    ICON_PATH = os.path.join(ROOT, "assets", "icons", "mac", "app_icon.icns")
else:
    ICON_PATH = os.path.join(ROOT, "assets", "icons", "linux", "app_icon.png")

datas = [
    (os.path.join(ROOT, "assets"), "assets"),
    (os.path.join(ROOT, "locales"), "locales"),
]

hiddenimports = [
    "modules.processors.frame.face_swapper",
    "modules.processors.frame.face_enhancer",
    "modules.processors.frame.face_enhancer_gpen256",
    "modules.processors.frame.face_enhancer_gpen512",
]

a = Analysis(
    [os.path.join(ROOT, "run.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Deep-Live-Cam",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=ICON_PATH,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Deep-Live-Cam",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Deep-Live-Cam.app",
        icon=ICON_PATH,
        bundle_identifier="com.tdstudio.deeplivecam",
        info_plist={
            "CFBundleDisplayName": "Deep Live Cam",
            "CFBundleShortVersionString": "2.1.5",
            "NSCameraUsageDescription": "Deep Live Cam uses your camera for real-time face swapping.",
            "NSHighResolutionCapable": True,
        },
    )
