"""Unit tests for Binance Black & Gold UI theme, icons, and VirtualCamStreamer."""

import os
import sys
import types
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import modules.ui_theme as ui_theme  # noqa: E402
from modules.video_capture import VideoCapturer  # noqa: E402
from modules.virtual_cam import (  # noqa: E402
    VirtualCamStreamer,
    virtual_cam_setup_hint,
)

_APP = None


def setUpModule():
    global _APP
    _APP = QApplication.instance() or QApplication([])


class IconAndThemeTests(unittest.TestCase):
    def test_app_icon_loads_non_empty(self):
        icon = ui_theme.load_app_icon()
        self.assertFalse(icon.isNull())

    def test_floating_dock_and_button_construct(self):
        dock = ui_theme.FloatingDock()
        btn = ui_theme.FloatingButton("Start", dock, glow_gold=True)
        self.assertEqual(btn.text(), "Start")
        self.assertIsNotNone(dock.graphicsEffect())
        self.assertIsNotNone(btn.graphicsEffect())

    def test_media_dropzone_clear_resets_property(self):
        zone = ui_theme.MediaDropZone("Drop")
        zone.set_media_active(True)
        self.assertEqual(zone.property("hasMedia"), "true")
        zone.clear_media()
        self.assertEqual(zone.property("hasMedia"), "false")
        self.assertEqual(zone.text(), "Drop")


class VirtualCamStreamerTests(unittest.TestCase):
    def test_rejects_invalid_frame(self):
        streamer = VirtualCamStreamer(30.0)
        self.assertFalse(streamer.send(None))
        self.assertFalse(streamer.send(np.zeros((10, 10), dtype=np.uint8)))

    def test_missing_pyvirtualcam_returns_false_with_hint(self):
        streamer = VirtualCamStreamer(30.0)
        with patch.dict(sys.modules, {"pyvirtualcam": None}):
            ok = streamer.send(np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertFalse(ok)
        self.assertIn("pyvirtualcam", virtual_cam_setup_hint().lower())
        self.assertTrue(len(streamer.last_error) > 0)

    def test_sends_frame_and_reopens_on_resolution_change(self):
        opened = []
        sent = []
        closed = []

        class _FakeCam:
            def __init__(self, width, height, fps, fmt):
                self.device = "FakeVCam"
                opened.append((width, height, fps, fmt))

            def send(self, frame):
                sent.append(frame.shape)

            def close(self):
                closed.append(True)

        fake_mod = types.ModuleType("pyvirtualcam")
        fake_mod.Camera = _FakeCam
        fake_mod.PixelFormat = types.SimpleNamespace(BGR="bgr")

        streamer = VirtualCamStreamer(30.0)
        with patch.dict(sys.modules, {"pyvirtualcam": fake_mod}):
            self.assertTrue(streamer.send(np.zeros((24, 32, 3), dtype=np.uint8)))
            self.assertTrue(streamer.send(np.zeros((48, 64, 3), dtype=np.uint8)))
            streamer.close()

        self.assertEqual([(o[0], o[1]) for o in opened], [(32, 24), (64, 48)])
        self.assertEqual(sent, [(24, 32, 3), (48, 64, 3)])
        self.assertEqual(len(closed), 2)

    def test_video_capturer_accepts_stream_url_string(self):
        cap = VideoCapturer("http://192.168.1.50:8765/stream.mjpg")
        self.assertEqual(cap.device_index, "http://192.168.1.50:8765/stream.mjpg")
        self.assertFalse(cap.start())


if __name__ == "__main__":
    unittest.main()
