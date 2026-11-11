"""Optional desktop virtual-camera output helper (OBS Virtual Cam / v4l2loopback).

Uses ``pyvirtualcam`` when installed; degrades cleanly with a helpful status
hint when absent or when no virtual-camera driver is active.
"""

from __future__ import annotations

import platform
from typing import Optional

import numpy as np


def is_virtual_cam_library_installed() -> bool:
    try:
        import pyvirtualcam  # noqa: F401
        return True
    except ImportError:
        return False


def virtual_cam_setup_hint() -> str:
    sys_name = platform.system()
    if sys_name == "Windows":
        return "Install pyvirtualcam + OBS Studio (Virtual Camera) to stream to other apps."
    if sys_name == "Darwin":
        return "Install pyvirtualcam + OBS Studio (macOS Camera Extension) to stream to other apps."
    return "Install pyvirtualcam + v4l2loopback kernel module (modprobe v4l2loopback) for virtual cam."


class VirtualCamStreamer:
    """Lazy-opened virtual camera sink that adapts to frame dimensions."""

    def __init__(self, fps: float = 30.0) -> None:
        self._fps = max(1.0, float(fps or 30.0))
        self._cam = None
        self._size: Optional[tuple[int, int]] = None
        self.device_name: str = ""
        self.last_error: str = ""

    def send(self, bgr_frame: np.ndarray) -> bool:
        if bgr_frame is None or bgr_frame.ndim != 3:
            return False
        h, w = bgr_frame.shape[:2]
        if w <= 0 or h <= 0:
            return False
        try:
            import pyvirtualcam
        except ImportError:
            self.last_error = virtual_cam_setup_hint()
            return False

        if self._cam is None or self._size != (w, h):
            self.close()
            try:
                self._cam = pyvirtualcam.Camera(
                    width=w,
                    height=h,
                    fps=self._fps,
                    fmt=pyvirtualcam.PixelFormat.BGR,
                )
                self._size = (w, h)
                self.device_name = getattr(self._cam, "device", "Virtual Camera")
                self.last_error = ""
            except Exception as exc:
                self._cam = None
                self._size = None
                self.last_error = f"Virtual camera unavailable: {exc}"
                return False

        try:
            self._cam.send(np.ascontiguousarray(bgr_frame))
            return True
        except Exception as exc:
            self.last_error = f"Virtual camera write error: {exc}"
            self.close()
            return False

    def close(self) -> None:
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None
            self._size = None
