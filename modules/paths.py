"""Shared path constants for the Deep-Live-Cam project."""

import os
import sys

if getattr(sys, "frozen", False):
    _BUNDLE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    ROOT_DIR = os.path.dirname(sys.executable)
    ASSETS_DIR = os.path.join(_BUNDLE_DIR, "assets")
    LOCALES_DIR = os.path.join(_BUNDLE_DIR, "locales")
else:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
    LOCALES_DIR = os.path.join(ROOT_DIR, "locales")

MODELS_DIR = os.path.join(ROOT_DIR, "models")
ICONS_DIR = os.path.join(ASSETS_DIR, "icons")
