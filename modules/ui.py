"""PySide6 Studio UI for Deep-Live-Cam (Binance Obsidian Black & Gold theme).

Public API kept stable for the rest of the codebase:
    init(start, destroy, lang) -> _Window
        Returned object has .mainloop() that core.py calls.
    update_status(text)
        Thread-safe; routed through Qt signal when called off-UI.
    check_and_ignore_nsfw(target, destroy=None) -> bool
"""

from __future__ import annotations

import json
import os
import platform
import queue
import sys
import tempfile
import threading
import time
import webbrowser
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QObject,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import modules.globals
import modules.metadata
from modules import imread_unicode
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    add_blank_map,
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    has_valid_map,
    reset_face_analyser,
    simplify_maps,
)
from modules.gettext import LanguageManager
from modules.gpu_processing import gpu_cvt_color, gpu_flip, gpu_resize
from modules.model_downloader import is_present
from modules.paths import ICONS_DIR
from modules.platform_info import accelerator_label
from modules.processors.frame.core import get_frame_processors_modules
from modules.ui_theme import (
    BINANCE_QSS,
    FloatingButton,
    FloatingDock,
    MediaDropZone,
    StudioSwitch,
    apply_floating_shadow,
    load_app_icon,
)
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
)
from modules.video_capture import VideoCapturer
from modules.virtual_cam import (
    VirtualCamStreamer,
    is_virtual_cam_library_installed,
    virtual_cam_setup_hint,
)

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph


# ─── constants ────────────────────────────────────────────────────────────

ROOT_MIN_WIDTH = 440
ROOT_MIN_HEIGHT = 640
ROOT_HEIGHT = 840
ROOT_WIDTH = 1020
RESPONSIVE_TWO_COL_BP = 820
RESPONSIVE_HEADER_BP = 740
RESPONSIVE_MEDIA_STACK_BP = 640
RESPONSIVE_DOCK_WRAP_BP = 600
RESPONSIVE_OPTS_ONE_COL_BP = 560

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 720
PREVIEW_DEFAULT_HEIGHT = 440

POPUP_WIDTH = 780
POPUP_HEIGHT = 820
POPUP_LIVE_WIDTH = 920
POPUP_LIVE_HEIGHT = 820

MAPPER_PREVIEW_SIZE = 100
SOURCE_TARGET_PREVIEW_SIZE = 210


# ─── module-level state ───────────────────────────────────────────────────

_APP: Optional[QApplication] = None
_MAIN: Optional["MainWindow"] = None
_PREVIEW: Optional["PreviewWindow"] = None
_WEBCAM_PREVIEW: Optional["WebcamPreviewWindow"] = None
# How long to wait quietly for a live-preview worker to notice the stop flag
# before saying so. Shutdown then keeps waiting: it has to finish.
WORKER_SHUTDOWN_GRACE_MS = 30000
# How long the face may go undetected before the live preview drops the last
# async swap result. A short detector miss keeps it, so the swap resumes at
# once instead of showing the real face for a whole inference.
ASYNC_SWAP_FACE_LOSS_GRACE_S = 0.5
_MAPPER: Optional["MapperDialog"] = None
_LIVE_MAPPER: Optional["LiveMapperDialog"] = None
_LANG: Optional[LanguageManager] = None
_BRIDGE: Optional["_UIBridge"] = None


def _(text: str) -> str:
    """Translate via LanguageManager; falls back to identity."""
    if _LANG is None:
        return text
    return _LANG._(text)


# Preserve original cwd state for file dialogs.
_RECENT_SOURCE_DIR: Optional[str] = None
_RECENT_TARGET_DIR: Optional[str] = None
_RECENT_OUTPUT_DIR: Optional[str] = None

# QFileDialog filter strings, built from the canonical extension sets in
# globals so every dialog stays in sync (no hand-copied lists to drift).
_IMAGE_FILE_FILTER = "Images (" + " ".join(
    f"*{ext}" for ext in modules.globals.IMAGE_EXTENSIONS
) + ")"
_MEDIA_FILE_FILTER = "Media (" + " ".join(
    f"*{ext}" for ext in (*modules.globals.IMAGE_EXTENSIONS, *modules.globals.VIDEO_EXTENSIONS)
) + ")"
_VIDEO_FILE_FILTER = "Videos (" + " ".join(
    f"*{ext}" for ext in modules.globals.VIDEO_EXTENSIONS
) + ")"


# ─── image utilities ─────────────────────────────────────────────────────


def fit_image_to_size(image, width: int, height: int):
    """BGR ndarray -> BGR ndarray scaled to fit within (width, height)."""
    if (width is None and height is None) or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio_w = width / w
    ratio_h = height / h
    ratio = min(ratio_w, ratio_h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return gpu_resize(image, dsize=new_size)


def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Zero-copy BGR ndarray -> QPixmap."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """PIL.Image -> QPixmap."""
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimg = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def render_image_preview(image_path: str, size: Tuple[int, int]) -> QPixmap:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return _pil_to_qpixmap(image)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[QPixmap]:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame:
            return None
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return _pil_to_qpixmap(image)
    finally:
        capture.release()


# ─── persistence ─────────────────────────────────────────────────────────


def save_switch_states():
    state = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "live_async_swap": modules.globals.live_async_swap,
        "virtual_cam": modules.globals.virtual_cam,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "mouth_mask_size": modules.globals.mouth_mask_size,
        "capture_resolution": list(modules.globals.capture_resolution),
        "det_size": modules.globals.det_size,
    }
    try:
        with open("switch_states.json", "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            state = json.load(f)
        modules.globals.keep_fps = state.get("keep_fps", True)
        modules.globals.keep_audio = state.get("keep_audio", True)
        modules.globals.keep_frames = state.get("keep_frames", False)
        modules.globals.many_faces = state.get("many_faces", False)
        modules.globals.map_faces = state.get("map_faces", False)
        modules.globals.poisson_blend = state.get("poisson_blend", False)
        modules.globals.color_correction = state.get("color_correction", False)
        modules.globals.nsfw_filter = state.get("nsfw_filter", False)
        modules.globals.live_mirror = state.get("live_mirror", False)
        modules.globals.live_resizable = state.get("live_resizable", False)
        modules.globals.fp_ui = state.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = state.get("show_fps", False)
        modules.globals.live_async_swap = state.get("live_async_swap", False)
        modules.globals.virtual_cam = state.get("virtual_cam", False)
        # Mouth mask always starts disabled (slider at 0) on launch,
        # regardless of the persisted value — enable it explicitly each session.
        modules.globals.mouth_mask_size = 0.0
        modules.globals.mouth_mask = False
        modules.globals.show_mouth_mask_box = False
        res = state.get("capture_resolution")
        if isinstance(res, (list, tuple)) and len(res) == 2:
            try:
                w, h = int(res[0]), int(res[1])
                if w > 0 and h > 0:
                    modules.globals.capture_resolution = (w, h)
            except (TypeError, ValueError):
                pass
        if state.get("det_size") in (160, 320, 640):
            modules.globals.det_size = int(state["det_size"])
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError):
        pass


# ─── thread-safe status bridge ───────────────────────────────────────────


class _UIBridge(QObject):
    """Single QObject that owns cross-thread signals."""

    statusChanged = Signal(str)


def _emit_status(text: str) -> None:
    if _BRIDGE is None:
        print(text)
        return
    _BRIDGE.statusChanged.emit(text)


# ─── public API ──────────────────────────────────────────────────────────


def update_status(text: str) -> None:
    """Thread-safe status update — uses signal if called off-UI thread."""
    _emit_status(_(text))
    if _APP is not None and QThread.currentThread() is _APP.thread():
        _APP.processEvents()


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Processing ignored!")
        return True
    return False


# ─── camera enumeration ──────────────────────────────────────────────────


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                return list(range(len(devices))), devices
            return [], ["No cameras found"]
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
            return [], ["No cameras found"]

    if platform.system() == "Darwin":
        return [0, 1], ["Camera 0", "Camera 1"]

    # Linux probe
    indices: List[int] = []
    names: List[str] = []
    for i in range(10):
        cap = cv2.VideoCapture(f"/dev/video{i}")
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()
    return (indices, names) if names else ([], ["No cameras found"])


# ─── main window ─────────────────────────────────────────────────────────


def _make_card_frame(object_name: str = "studioCard") -> QFrame:
    card = QFrame()
    card.setObjectName(object_name)
    apply_floating_shadow(card, color=QColor(0, 0, 0, 145), blur=22, y_offset=5)
    return card


def _check_core_models_present() -> bool:
    return any(
        is_present(name)
        for name in ("inswapper_128_fp16.onnx", "inswapper_128.onnx")
    )


class MainWindow(QMainWindow):
    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        load_switch_states()
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb
        self._is_two_col: Optional[bool] = None
        self._dock_compact: Optional[bool] = None
        self._header_compact: Optional[bool] = None
        self._media_stacked: Optional[bool] = None
        self._opts_one_col: Optional[bool] = None
        self._custom_streams: List[Tuple[str, str]] = []

        self.setWindowTitle(
            f"{modules.metadata.name} {modules.metadata.version} • {modules.metadata.edition}"
        )
        icon = load_app_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)
        self.setMinimumSize(ROOT_MIN_WIDTH, ROOT_MIN_HEIGHT)
        self.resize(ROOT_WIDTH, ROOT_HEIGHT)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 14, 16, 12)
        outer.setSpacing(12)

        # Top studio header
        outer.addWidget(self._build_header_bar())

        # Responsive scrollable studio workspace
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._workspace = QWidget()
        self._workspace.setObjectName("studioWorkspace")
        self._workspace_grid = QGridLayout(self._workspace)
        self._workspace_grid.setContentsMargins(2, 2, 4, 6)
        self._workspace_grid.setHorizontalSpacing(14)
        self._workspace_grid.setVerticalSpacing(14)

        self._media_card = self._build_media_studio_card()
        self._refinement_card = self._build_sliders_card()
        self._camera_card = self._build_camera_card()
        self._options_card = self._build_options_card()

        self._left_col = QWidget()
        left_lay = QVBoxLayout(self._left_col)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(14)
        left_lay.addWidget(self._media_card)
        left_lay.addWidget(self._refinement_card)
        left_lay.addStretch(1)

        self._right_col = QWidget()
        right_lay = QVBoxLayout(self._right_col)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(14)
        right_lay.addWidget(self._camera_card)
        right_lay.addWidget(self._options_card)
        right_lay.addStretch(1)

        self._scroll.setWidget(self._workspace)
        outer.addWidget(self._scroll, 1)

        # Floating bottom action dock
        self._floating_dock = self._build_floating_dock()
        outer.addWidget(self._floating_dock, alignment=Qt.AlignmentFlag.AlignCenter)

        # Telemetry status bar & link footer
        footer_row = QHBoxLayout()
        footer_row.setContentsMargins(4, 0, 4, 0)
        footer_row.setSpacing(10)

        self._status_label = QLabel(_("Ready • Drop a face & target or press Live"))
        self._status_label.setObjectName("statusLabel")
        self._status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        footer_row.addWidget(self._status_label, 1)

        footer = QLabel("Deep Live Cam • Gold Studio")
        footer.setObjectName("linkLabel")
        footer.setCursor(Qt.CursorShape.PointingHandCursor)
        footer.mousePressEvent = lambda _e: webbrowser.open("https://deeplivecam.net")
        footer_row.addWidget(footer)
        outer.addLayout(footer_row)

        self._apply_responsive_layout(self.width())

    # ── header bar ───────────────────────────────────────────────────────

    def _build_header_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("headerBar")
        apply_floating_shadow(bar, color=QColor(0, 0, 0, 155), blur=20, y_offset=4)

        self._header_grid = QGridLayout(bar)
        self._header_grid.setContentsMargins(14, 10, 14, 10)
        self._header_grid.setHorizontalSpacing(10)
        self._header_grid.setVerticalSpacing(8)

        # Left brand cluster
        self._header_brand = QWidget()
        brand_row = QHBoxLayout(self._header_brand)
        brand_row.setContentsMargins(0, 0, 0, 0)
        brand_row.setSpacing(8)

        logo_lbl = QLabel()
        icon_path = os.path.join(ICONS_DIR, "app_icon_64.png")
        if os.path.isfile(icon_path):
            pm = QPixmap(icon_path).scaled(
                34, 34, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_lbl.setPixmap(pm)
        logo_lbl.setFixedSize(34, 34)
        brand_row.addWidget(logo_lbl)

        title_white = QLabel("DEEP LIVE")
        title_white.setObjectName("brandTitle")
        title_gold = QLabel("CAM")
        title_gold.setObjectName("brandGold")
        brand_row.addWidget(title_white)
        brand_row.addWidget(title_gold)

        studio_pill = QLabel(f"GOLD STUDIO v{modules.metadata.version}")
        studio_pill.setObjectName("badgePill")
        brand_row.addWidget(studio_pill)
        brand_row.addStretch(1)

        # Right telemetry & model pills
        self._header_pills = QWidget()
        pills_row = QHBoxLayout(self._header_pills)
        pills_row.setContentsMargins(0, 0, 0, 0)
        pills_row.setSpacing(8)

        accel = accelerator_label()
        self._accel_pill = QLabel(f"⚡ {accel}")
        self._accel_pill.setObjectName("accelPill")
        self._accel_pill.setToolTip(_("Active ONNX hardware execution provider"))
        pills_row.addWidget(self._accel_pill)

        models_ok = _check_core_models_present()
        self._model_pill = FloatingButton(
            _("● Models Ready") if models_ok else _("↓ Models Info"),
            glow_gold=not models_ok,
            base_blur=12,
            hover_blur=18,
        )
        self._model_pill.setObjectName("overlayPill")
        self._model_pill.setToolTip(
            _("Check or pre-download swapper and face-analysis ONNX models")
        )
        self._model_pill.clicked.connect(self._on_check_models)
        pills_row.addWidget(self._model_pill)
        return bar

    # ── media studio card (Source <-> Target) ────────────────────────────

    def _build_media_studio_card(self) -> QFrame:
        card = _make_card_frame("studioCard")
        root = QVBoxLayout(card)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        hdr = QHBoxLayout()
        title = QLabel(_("FACE & TARGET STUDIO"))
        title.setObjectName("sectionTitle")
        hdr.addWidget(title)
        hdr.addStretch(1)
        hint = QLabel(_("Drop media or click frames"))
        hint.setObjectName("mutedHint")
        hdr.addWidget(hint)
        root.addLayout(hdr)

        self._media_row = QGridLayout()
        self._media_row.setHorizontalSpacing(10)
        self._media_row.setVerticalSpacing(10)

        # Source well
        self._src_card = QFrame()
        self._src_card.setObjectName("mediaCard")
        src_lay = QVBoxLayout(self._src_card)
        src_lay.setContentsMargins(10, 10, 10, 10)
        src_lay.setSpacing(8)

        self.source_label = MediaDropZone(_("Drop Source Face\nor Click to Browse"))
        self.source_label.setMinimumSize(140, 165)
        self.source_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.source_label.clicked.connect(self._on_select_source)
        self.source_label.fileDropped.connect(self._load_source_path)
        src_lay.addWidget(self.source_label, 1)

        src_pills = QHBoxLayout()
        src_pills.setSpacing(6)
        self.btn_select_source = FloatingButton(_("Select a face"), glow_gold=True)
        self.btn_select_source.setObjectName("goldPrimary")
        self.btn_select_source.setToolTip(
            _("Choose the source face image to swap onto the target")
        )
        self.btn_select_source.clicked.connect(self._on_select_source)

        self.btn_random_face = FloatingButton("✦ AI")
        self.btn_random_face.setObjectName("miniPill")
        self.btn_random_face.setMinimumWidth(64)
        self.btn_random_face.setToolTip(
            _("Get a random face from thispersondoesnotexist.com")
        )
        self.btn_random_face.clicked.connect(self._on_random_face)

        self.btn_clear_source = FloatingButton("✕")
        self.btn_clear_source.setObjectName("miniPill")
        self.btn_clear_source.setFixedWidth(36)
        self.btn_clear_source.setToolTip(_("Clear selected source face"))
        self.btn_clear_source.clicked.connect(self._on_clear_source)

        src_pills.addWidget(self.btn_select_source, 1)
        src_pills.addWidget(self.btn_random_face)
        src_pills.addWidget(self.btn_clear_source)
        src_lay.addLayout(src_pills)

        # Floating Swap button centered between Source and Target
        self.btn_swap = FloatingButton("⇄", glow_gold=True, base_blur=22, hover_blur=30)
        self.btn_swap.setObjectName("iconCircle")
        self.btn_swap.setFixedSize(44, 44)
        self.btn_swap.setToolTip(_("Swap source and target images"))
        self.btn_swap.clicked.connect(self._on_swap_paths)

        # Target well
        self._tgt_card = QFrame()
        self._tgt_card.setObjectName("mediaCard")
        tgt_lay = QVBoxLayout(self._tgt_card)
        tgt_lay.setContentsMargins(10, 10, 10, 10)
        tgt_lay.setSpacing(8)

        self.target_label = MediaDropZone(_("Drop Target Photo / Video\nor Click to Browse"))
        self.target_label.setMinimumSize(140, 165)
        self.target_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.target_label.clicked.connect(self._on_select_target)
        self.target_label.fileDropped.connect(self._load_target_path)
        tgt_lay.addWidget(self.target_label, 1)

        tgt_pills = QHBoxLayout()
        tgt_pills.setSpacing(6)
        self.btn_select_target = FloatingButton(_("Select a target"))
        self.btn_select_target.setObjectName("floatingPill")
        self.btn_select_target.setToolTip(
            _("Choose the target image or video to apply face swap to")
        )
        self.btn_select_target.clicked.connect(self._on_select_target)

        self.btn_clear_target = FloatingButton("✕")
        self.btn_clear_target.setObjectName("miniPill")
        self.btn_clear_target.setFixedWidth(36)
        self.btn_clear_target.setToolTip(_("Clear selected target media"))
        self.btn_clear_target.clicked.connect(self._on_clear_target)

        tgt_pills.addWidget(self.btn_select_target, 1)
        tgt_pills.addWidget(self.btn_clear_target)
        tgt_lay.addLayout(tgt_pills)

        root.addLayout(self._media_row)
        return card

    # ── sliders card ─────────────────────────────────────────────────────

    def _build_sliders_card(self) -> QFrame:
        card = _make_card_frame("studioCard")
        root = QVBoxLayout(card)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        hdr = QHBoxLayout()
        title = QLabel(_("REFINEMENT"))
        title.setObjectName("sectionTitle")
        hdr.addWidget(title)
        hdr.addStretch(1)
        btn_reset = FloatingButton(_("Reset"), base_blur=10, hover_blur=16)
        btn_reset.setObjectName("overlayPill")
        btn_reset.setToolTip(_("Reset sliders to default values"))
        btn_reset.clicked.connect(self._on_reset_refinements)
        hdr.addWidget(btn_reset)
        root.addLayout(hdr)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        def slider(min_v, max_v, default, denom, on_change):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(int(min_v * denom), int(max_v * denom))
            s.setValue(int(default * denom))
            s.setCursor(Qt.CursorShape.PointingHandCursor)
            s.valueChanged.connect(lambda iv: on_change(iv / denom))
            return s

        # Transparency
        grid.addWidget(QLabel(_("Transparency")), 0, 0)
        self._val_transparency = QLabel("100%")
        self._val_transparency.setObjectName("valuePill")
        self._val_transparency.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.s_transparency = slider(0.0, 1.0, 1.0, 100, self._on_transparency_change)
        self.s_transparency.setToolTip(
            _("Blend between original and swapped face (0% = original, 100% = fully swapped)")
        )
        grid.addWidget(self.s_transparency, 0, 1)
        grid.addWidget(self._val_transparency, 0, 2)

        # Sharpness
        grid.addWidget(QLabel(_("Sharpness")), 1, 0)
        self._val_sharpness = QLabel("0.0")
        self._val_sharpness.setObjectName("valuePill")
        self._val_sharpness.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.s_sharpness = slider(0.0, 5.0, 0.0, 10, self._on_sharpness_change)
        self.s_sharpness.setToolTip(_("Sharpen the enhanced face output"))
        grid.addWidget(self.s_sharpness, 1, 1)
        grid.addWidget(self._val_sharpness, 1, 2)

        # Mouth mask
        grid.addWidget(QLabel(_("Mouth Mask")), 2, 0)
        self._val_mouth = QLabel("0%")
        self._val_mouth.setObjectName("valuePill")
        self._val_mouth.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.s_mouth = slider(0.0, 100.0, 0.0, 1, self._on_mouth_mask_change)
        self.s_mouth.sliderPressed.connect(self._on_mouth_mask_pressed)
        self.s_mouth.sliderReleased.connect(self._on_mouth_mask_released)
        self.s_mouth.setToolTip(
            _("0 = use swapped mouth, 100 = expose original mouth to chin area")
        )
        grid.addWidget(self.s_mouth, 2, 1)
        grid.addWidget(self._val_mouth, 2, 2)

        grid.setColumnStretch(1, 1)
        root.addLayout(grid)
        return card

    # ── camera card ──────────────────────────────────────────────────────

    def _build_camera_card(self) -> QFrame:
        card = _make_card_frame("studioCard")
        root = QVBoxLayout(card)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        hdr = QHBoxLayout()
        title = QLabel(_("LIVE CAMERA & STREAM"))
        title.setObjectName("sectionTitle")
        hdr.addWidget(title)
        hdr.addStretch(1)
        vcam_badge = QLabel(
            _("VCam Ready") if is_virtual_cam_library_installed() else _("Preview Mode")
        )
        vcam_badge.setObjectName("badgePill")
        hdr.addWidget(vcam_badge)
        root.addLayout(hdr)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        # Row 0: camera selector + refresh + phone bridge buttons
        grid.addWidget(QLabel(_("Select Camera:")), 0, 0)
        self.cb_camera = QComboBox()
        self.cb_camera.setToolTip(_("Select which camera or phone stream to use for live mode"))
        grid.addWidget(self.cb_camera, 0, 1)

        cam_btns = QHBoxLayout()
        cam_btns.setSpacing(6)
        self.btn_refresh_cams = FloatingButton("↻", base_blur=10, hover_blur=16)
        self.btn_refresh_cams.setObjectName("miniPill")
        self.btn_refresh_cams.setFixedSize(36, 36)
        self.btn_refresh_cams.setToolTip(_("Rescan connected webcams and virtual cameras"))
        self.btn_refresh_cams.clicked.connect(self._refresh_cameras)

        self.btn_phone_bridge = FloatingButton("+ Bridge", base_blur=10, hover_blur=16)
        self.btn_phone_bridge.setObjectName("miniPill")
        self.btn_phone_bridge.setToolTip(
            _("Connect your Android Deep-Live-Cam app stream (HTTP/MJPEG or RTSP URL)")
        )
        self.btn_phone_bridge.clicked.connect(self._on_add_phone_bridge)

        cam_btns.addWidget(self.btn_refresh_cams)
        cam_btns.addWidget(self.btn_phone_bridge)
        grid.addLayout(cam_btns, 0, 2)

        # Row 1: capture resolution
        grid.addWidget(QLabel(_("Resolution:")), 1, 0)
        self.cb_resolution = QComboBox()
        self._resolution_options = [
            ("640 x 480", (640, 480)),
            ("960 x 540 (qHD)", (960, 540)),
            ("1280 x 720 (HD)", (1280, 720)),
            ("1920 x 1080 (FHD)", (1920, 1080)),
        ]
        for label, _wh in self._resolution_options:
            self.cb_resolution.addItem(label)
        cur = tuple(modules.globals.capture_resolution)
        idx = next((i for i, (_, wh) in enumerate(self._resolution_options) if wh == cur), 0)
        self.cb_resolution.setCurrentIndex(idx)
        self.cb_resolution.currentIndexChanged.connect(self._on_resolution_change)
        self.cb_resolution.setToolTip(_(
            "Requested webcam resolution. Camera may negotiate to its "
            "nearest supported size — actual size printed in console as "
            "'[VideoCapturer] WxH @ FPS'. Applies on next Live start.\n\n"
            "640x480 is native on virtually every webcam and the safest "
            "30fps choice. qHD often works too. HD/FHD typically drop to "
            "~10fps on USB 2.0 webcams due to isochronous bandwidth limits."
        ))
        grid.addWidget(self.cb_resolution, 1, 1, 1, 2)

        # Row 2: face detection size
        grid.addWidget(QLabel(_("Det size:")), 2, 0)
        self.cb_det_size = QComboBox()
        self._det_size_options = [160, 320, 640]
        for v in self._det_size_options:
            self.cb_det_size.addItem(f"{v} x {v}")
        cur_det = int(getattr(modules.globals, "det_size", modules.globals.DEFAULT_DET_SIZE))
        if cur_det not in self._det_size_options:
            cur_det = (
                modules.globals.DEFAULT_DET_SIZE
                if modules.globals.DEFAULT_DET_SIZE in self._det_size_options
                else self._det_size_options[-1]
            )
        self.cb_det_size.setCurrentIndex(self._det_size_options.index(cur_det))
        self.cb_det_size.currentIndexChanged.connect(self._on_det_size_change)
        self.cb_det_size.setToolTip(_(
            "Face detection input resolution. Lower = faster, less accurate at "
            "distance. Changes take effect on next face analyser init."
        ))
        grid.addWidget(self.cb_det_size, 2, 1, 1, 2)

        # Live switches
        self.sw_async_swap = StudioSwitch(
            _("Async swap"),
            modules.globals.live_async_swap,
            _(
                "Run the swap model in the background so the live preview stays "
                "responsive when the model is slow (CPU, Apple Silicon, older "
                "GPUs). Single-face mode only."
            ),
        )
        self.sw_async_swap.toggled.connect(
            lambda v: (
                setattr(modules.globals, "live_async_swap", v),
                save_switch_states(),
            )
        )
        grid.addWidget(self.sw_async_swap, 3, 0, 1, 3)

        self.sw_live_mirror = StudioSwitch(
            _("Live mirror"),
            modules.globals.live_mirror,
            _("Flip the live webcam horizontally like a front-facing mirror"),
        )
        self.sw_live_mirror.toggled.connect(
            lambda v: (
                setattr(modules.globals, "live_mirror", v),
                save_switch_states(),
            )
        )
        grid.addWidget(self.sw_live_mirror, 4, 0, 1, 3)

        self.sw_virtual_cam = StudioSwitch(
            _("Virtual Cam Output"),
            modules.globals.virtual_cam,
            _(virtual_cam_setup_hint()),
        )
        self.sw_virtual_cam.toggled.connect(self._on_virtual_cam_toggled)
        grid.addWidget(self.sw_virtual_cam, 5, 0, 1, 3)

        grid.setColumnStretch(1, 1)
        root.addLayout(grid)
        self._populate_camera_list(initial=True)
        return card

    def _populate_camera_list(self, initial: bool = False) -> None:
        indices, names = get_available_cameras()
        self._camera_indices = list(indices)
        self._camera_names = list(names) if names and names[0] != "No cameras found" else []
        for url, label in getattr(self, "_custom_streams", []):
            self._camera_indices.append(url)
            self._camera_names.append(label)

        self.cb_camera.clear()
        if not self._camera_names:
            self.cb_camera.addItem(_("No cameras found"))
            self.cb_camera.setEnabled(False)
            cam_ok = False
        else:
            self.cb_camera.addItems(self._camera_names)
            self.cb_camera.setEnabled(True)
            cam_ok = True
        if not initial and hasattr(self, "btn_live"):
            self.btn_live.setEnabled(cam_ok)

    def _on_add_phone_bridge(self) -> None:
        url, ok = QInputDialog.getText(
            self,
            _("Connect Android Phone Bridge"),
            _("Enter Android Bridge stream URL (e.g. http://192.168.1.50:8765/stream.mjpg):"),
            text="http://192.168.1.50:8765/stream.mjpg",
        )
        if not ok or not url.strip():
            return
        clean_url = url.strip()
        label = f"Phone Bridge ({clean_url})"
        if not any(u == clean_url for u, _ in self._custom_streams):
            self._custom_streams.append((clean_url, label))
        self._populate_camera_list(initial=False)
        self.cb_camera.setCurrentText(label)
        update_status(f"Phone Bridge added: {clean_url}")

    def _refresh_cameras(self) -> None:
        self._populate_camera_list(initial=False)
        count = len(self._camera_indices)
        update_status(f"Camera scan complete: {count} device(s) detected.")

    # ── options card ─────────────────────────────────────────────────────

    def _build_options_card(self) -> QFrame:
        card = _make_card_frame("studioCard")
        root = QVBoxLayout(card)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QLabel(_("PROCESSING OPTIONS"))
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)

        def make(field, label, tip):
            sw = StudioSwitch(_(label), getattr(modules.globals, field), _(tip))
            sw.toggled.connect(
                lambda v, f=field: (
                    setattr(modules.globals, f, v),
                    save_switch_states(),
                )
            )
            return sw

        self.sw_keep_fps = make(
            "keep_fps", "Keep fps", "Output video keeps the original frame rate"
        )
        self.sw_keep_audio = make(
            "keep_audio", "Keep audio", "Copy audio track from the source video to output"
        )
        self.sw_keep_frames = make(
            "keep_frames", "Keep frames", "Keep extracted frames on disk after processing"
        )
        self.sw_many_faces = make(
            "many_faces", "Many faces", "Swap every detected face, not just the primary one"
        )
        self.sw_poisson = make(
            "poisson_blend", "Poisson Blend", "Blend face edges smoothly using Poisson blending"
        )
        self.sw_color_fix = make(
            "color_correction", "Fix Blueish Cam", "Fix blue/green color cast from some webcams"
        )
        self.sw_show_fps = make(
            "show_fps", "Show FPS", "Display frames-per-second counter on the live preview"
        )
        self.sw_map_faces = StudioSwitch(
            _("Map faces"),
            modules.globals.map_faces,
            _("Manually assign which source face maps to which target face"),
        )
        self.sw_map_faces.toggled.connect(self._on_map_faces_toggled)

        self._option_switches = [
            self.sw_keep_fps,
            self.sw_keep_audio,
            self.sw_keep_frames,
            self.sw_many_faces,
            self.sw_map_faces,
            self.sw_show_fps,
            self.sw_poisson,
            self.sw_color_fix,
        ]
        self._opts_grid = grid
        self._enhancer_label = QLabel(_("Face Enhancer:"))

        self.cb_enhancer = QComboBox()
        self.cb_enhancer.addItems(["None", "GFPGAN", "GPEN-512", "GPEN-256"])
        initial = "None"
        if modules.globals.fp_ui.get("face_enhancer", False):
            initial = "GFPGAN"
        elif modules.globals.fp_ui.get("face_enhancer_gpen512", False):
            initial = "GPEN-512"
        elif modules.globals.fp_ui.get("face_enhancer_gpen256", False):
            initial = "GPEN-256"
        self.cb_enhancer.setCurrentText(initial)
        self.cb_enhancer.currentTextChanged.connect(self._on_enhancer_change)
        self.cb_enhancer.setToolTip(_("Select a face enhancement model (None = no enhancement)"))

        root.addLayout(grid)
        return card

    # ── floating action dock ─────────────────────────────────────────────

    def _build_floating_dock(self) -> FloatingDock:
        dock = FloatingDock()
        self._dock_grid = QGridLayout(dock)
        self._dock_grid.setContentsMargins(16, 10, 16, 10)
        self._dock_grid.setHorizontalSpacing(10)
        self._dock_grid.setVerticalSpacing(8)

        self.btn_start = FloatingButton("▶  " + _("Start"), glow_gold=True)
        self.btn_start.setToolTip(_("Begin processing the target image/video with selected face"))
        self.btn_start.clicked.connect(self._on_start)

        cam_ok = bool(self._camera_names and self._camera_names[0] != "No cameras found")
        self.btn_live = FloatingButton("●  " + _("Live"))
        self.btn_live.setObjectName("liveAccent")
        self.btn_live.setEnabled(cam_ok)
        self.btn_live.setToolTip(_("Start real-time face swap using webcam"))
        self.btn_live.clicked.connect(self._on_live)

        self.btn_preview = FloatingButton("◉  " + _("Preview"))
        self.btn_preview.setObjectName("floatingPill")
        self.btn_preview.setToolTip(_("Show/hide a preview of the processed output"))
        self.btn_preview.clicked.connect(self._on_toggle_preview)

        self.btn_destroy = FloatingButton("■  " + _("Destroy"))
        self.btn_destroy.setObjectName("danger")
        self.btn_destroy.setToolTip(_("Stop processing and close the application"))
        self.btn_destroy.clicked.connect(lambda: self._destroy_cb())

        self._dock_buttons = [
            self.btn_start,
            self.btn_live,
            self.btn_preview,
            self.btn_destroy,
        ]
        for btn in self._dock_buttons:
            btn.setMinimumWidth(118)
        return dock

    # ── responsive layout management ─────────────────────────────────────

    def _apply_responsive_layout(self, width: int) -> None:
        header_compact = width < RESPONSIVE_HEADER_BP
        if header_compact != self._header_compact:
            self._header_compact = header_compact
            self._header_grid.removeWidget(self._header_brand)
            self._header_grid.removeWidget(self._header_pills)
            self._header_grid.addWidget(self._header_brand, 0, 0)
            if header_compact:
                self._header_grid.addWidget(
                    self._header_pills, 1, 0, alignment=Qt.AlignmentFlag.AlignLeft
                )
                self._header_grid.setColumnStretch(0, 1)
                self._header_grid.setColumnStretch(1, 0)
            else:
                self._header_grid.addWidget(
                    self._header_pills, 0, 1, alignment=Qt.AlignmentFlag.AlignRight
                )
                self._header_grid.setColumnStretch(0, 1)
                self._header_grid.setColumnStretch(1, 0)

        two_col = width >= RESPONSIVE_TWO_COL_BP
        if two_col != self._is_two_col:
            self._is_two_col = two_col
            for w in (self._left_col, self._right_col):
                self._workspace_grid.removeWidget(w)
            if two_col:
                self._workspace_grid.addWidget(self._left_col, 0, 0)
                self._workspace_grid.addWidget(self._right_col, 0, 1)
                self._workspace_grid.setColumnStretch(0, 11)
                self._workspace_grid.setColumnStretch(1, 9)
            else:
                self._workspace_grid.addWidget(self._left_col, 0, 0)
                self._workspace_grid.addWidget(self._right_col, 1, 0)
                self._workspace_grid.setColumnStretch(0, 1)
                self._workspace_grid.setColumnStretch(1, 0)

        media_stacked = width < RESPONSIVE_MEDIA_STACK_BP
        if media_stacked != self._media_stacked:
            self._media_stacked = media_stacked
            for w in (self._src_card, self.btn_swap, self._tgt_card):
                self._media_row.removeWidget(w)
            if media_stacked:
                self._media_row.addWidget(self._src_card, 0, 0)
                self._media_row.addWidget(
                    self.btn_swap, 1, 0, alignment=Qt.AlignmentFlag.AlignCenter
                )
                self._media_row.addWidget(self._tgt_card, 2, 0)
                self._media_row.setColumnStretch(0, 1)
                self._media_row.setColumnStretch(1, 0)
                self._media_row.setColumnStretch(2, 0)
            else:
                self._media_row.addWidget(self._src_card, 0, 0)
                self._media_row.addWidget(
                    self.btn_swap, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter
                )
                self._media_row.addWidget(self._tgt_card, 0, 2)
                self._media_row.setColumnStretch(0, 1)
                self._media_row.setColumnStretch(1, 0)
                self._media_row.setColumnStretch(2, 1)

        opts_one_col = width < RESPONSIVE_OPTS_ONE_COL_BP
        if opts_one_col != self._opts_one_col:
            self._opts_one_col = opts_one_col
            for w in (
                *self._option_switches,
                self._enhancer_label,
                self.cb_enhancer,
            ):
                self._opts_grid.removeWidget(w)
            if opts_one_col:
                for idx, w in enumerate(self._option_switches):
                    self._opts_grid.addWidget(w, idx, 0)
                base_row = len(self._option_switches)
                self._opts_grid.addWidget(self._enhancer_label, base_row, 0)
                self._opts_grid.addWidget(self.cb_enhancer, base_row + 1, 0)
                self._opts_grid.setColumnStretch(0, 1)
                self._opts_grid.setColumnStretch(1, 0)
            else:
                for idx, w in enumerate(self._option_switches):
                    self._opts_grid.addWidget(w, idx // 2, idx % 2)
                base_row = len(self._option_switches) // 2
                self._opts_grid.addWidget(self._enhancer_label, base_row, 0)
                self._opts_grid.addWidget(self.cb_enhancer, base_row, 1)
                self._opts_grid.setColumnStretch(0, 1)
                self._opts_grid.setColumnStretch(1, 1)

        compact_dock = width < RESPONSIVE_DOCK_WRAP_BP
        if compact_dock != self._dock_compact:
            self._dock_compact = compact_dock
            for btn in self._dock_buttons:
                self._dock_grid.removeWidget(btn)
            if compact_dock:
                for idx, btn in enumerate(self._dock_buttons):
                    self._dock_grid.addWidget(btn, idx // 2, idx % 2)
            else:
                for idx, btn in enumerate(self._dock_buttons):
                    self._dock_grid.addWidget(btn, 0, idx)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    # ── slot handlers ────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _on_check_models(self) -> None:
        if _check_core_models_present():
            update_status("Core swapper ONNX models are present and ready.")
            self._model_pill.setText(_("● Models Ready"))
        else:
            update_status("Models will auto-download from HuggingFace on first Start / Live.")

    def _on_resolution_change(self, idx: int) -> None:
        if 0 <= idx < len(self._resolution_options):
            _label, wh = self._resolution_options[idx]
            modules.globals.capture_resolution = wh
            save_switch_states()
            update_status(
                _("Capture resolution set (applies on next Live start):")
                + f" {wh[0]}x{wh[1]}"
            )

    def _on_det_size_change(self, idx: int) -> None:
        if 0 <= idx < len(self._det_size_options):
            v = self._det_size_options[idx]
            modules.globals.det_size = v
            reset_face_analyser()
            save_switch_states()
            update_status(
                _("Face detection size set (analyser re-inits on next call):")
                + f" {v}x{v}"
            )

    def _on_virtual_cam_toggled(self, value: bool) -> None:
        modules.globals.virtual_cam = value
        save_switch_states()
        if value and not is_virtual_cam_library_installed():
            update_status(virtual_cam_setup_hint())
        elif value:
            update_status("Virtual Camera output enabled for Live mode.")
        else:
            update_status("Virtual Camera output disabled.")

    def _load_source_path(self, path: str) -> None:
        global _RECENT_SOURCE_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        if path and is_image(path):
            modules.globals.source_path = path
            _RECENT_SOURCE_DIR = os.path.dirname(path)
            self.source_label.setPixmap(
                render_image_preview(path, (SOURCE_TARGET_PREVIEW_SIZE, SOURCE_TARGET_PREVIEW_SIZE))
            )
            self.source_label.setText("")
            update_status(f"Source face loaded: {os.path.basename(path)}")
        else:
            modules.globals.source_path = None
            self.source_label.clear_media()
            update_status("Selected file is not a supported image.")

    def _load_target_path(self, path: str) -> None:
        global _RECENT_TARGET_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        if not path:
            return
        if is_image(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            self.target_label.setPixmap(
                render_image_preview(path, (SOURCE_TARGET_PREVIEW_SIZE, SOURCE_TARGET_PREVIEW_SIZE))
            )
            self.target_label.setText("")
            update_status(f"Target image loaded: {os.path.basename(path)}")
        elif is_video(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            pm = render_video_preview(
                path, (SOURCE_TARGET_PREVIEW_SIZE, SOURCE_TARGET_PREVIEW_SIZE)
            )
            if pm:
                self.target_label.setPixmap(pm)
                self.target_label.setText("")
            update_status(f"Target video loaded: {os.path.basename(path)}")
        else:
            modules.globals.target_path = None
            self.target_label.clear_media()
            update_status("Selected file is not a supported image or video.")

    def _on_select_source(self) -> None:
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self,
            _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        self._load_source_path(path)

    def _on_select_target(self) -> None:
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self,
            _("select an target image or video"),
            _RECENT_TARGET_DIR or "",
            _MEDIA_FILE_FILTER,
        )
        if not path:
            return
        self._load_target_path(path)

    def _on_clear_source(self) -> None:
        modules.globals.source_path = None
        self.source_label.clear_media()
        update_status("Source face cleared.")

    def _on_clear_target(self) -> None:
        modules.globals.target_path = None
        self.target_label.clear_media()
        if _PREVIEW is not None:
            _PREVIEW.hide()
        update_status("Target media cleared.")

    def _on_random_face(self) -> None:
        if _PREVIEW is not None:
            _PREVIEW.hide()
        try:
            response = requests.get(
                "https://thispersondoesnotexist.com/random-person.jpeg",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                raise ValueError(f"expected an image, got {content_type!r}")
            temp_path = os.path.join(tempfile.gettempdir(), "deep_live_cam_random_face.jpg")
            staging_path = f"{temp_path}.part"
            try:
                with open(staging_path, "wb") as f:
                    f.write(response.content)
                pixmap = render_image_preview(staging_path, (200, 200))
                if imread_unicode(staging_path) is None:
                    raise ValueError("downloaded image could not be decoded")
                os.replace(staging_path, temp_path)
            except Exception:
                try:
                    os.remove(staging_path)
                except OSError:
                    pass
                raise
            modules.globals.source_path = temp_path
            self.source_label.setPixmap(pixmap)
            self.source_label.setText("")
        except Exception as exc:
            print(f"Failed to fetch random face: {exc}")

    def _on_swap_paths(self) -> None:
        global _RECENT_SOURCE_DIR, _RECENT_TARGET_DIR
        sp = modules.globals.source_path
        tp = modules.globals.target_path
        if not (sp and tp and is_image(sp) and is_image(tp)):
            return
        modules.globals.source_path, modules.globals.target_path = tp, sp
        _RECENT_SOURCE_DIR = os.path.dirname(tp)
        _RECENT_TARGET_DIR = os.path.dirname(sp)
        if _PREVIEW is not None:
            _PREVIEW.hide()
        self.source_label.setPixmap(
            render_image_preview(tp, (SOURCE_TARGET_PREVIEW_SIZE, SOURCE_TARGET_PREVIEW_SIZE))
        )
        self.target_label.setPixmap(
            render_image_preview(sp, (SOURCE_TARGET_PREVIEW_SIZE, SOURCE_TARGET_PREVIEW_SIZE))
        )
        self.source_label.setText("")
        self.target_label.setText("")

    def _on_reset_refinements(self) -> None:
        self.s_transparency.setValue(100)
        self.s_sharpness.setValue(0)
        self.s_mouth.setValue(0)
        update_status("Refinement sliders reset.")

    def _on_map_faces_toggled(self, value: bool) -> None:
        modules.globals.map_faces = value
        save_switch_states()
        if not value:
            close_mapper_window()

    def _on_enhancer_change(self, choice: str) -> None:
        key_map = {
            "None": None,
            "GFPGAN": "face_enhancer",
            "GPEN-512": "face_enhancer_gpen512",
            "GPEN-256": "face_enhancer_gpen256",
        }
        for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
            _update_tumbler(key, False)
        selected = key_map.get(choice)
        if selected:
            _update_tumbler(selected, True)
        save_switch_states()

    def _on_transparency_change(self, value: float) -> None:
        modules.globals.opacity = value
        pct = int(value * 100)
        self._val_transparency.setText(f"{pct}%")
        if pct == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            update_status("Transparency set to 0% - Face swapping disabled.")
        elif pct == 100:
            modules.globals.face_swapper_enabled = True
            update_status("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"Transparency set to {pct}%")

    def _on_sharpness_change(self, value: float) -> None:
        modules.globals.sharpness = value
        self._val_sharpness.setText(f"{value:.1f}")
        update_status(f"Sharpness set to {value:.1f}")

    def _on_mouth_mask_change(self, value: float) -> None:
        modules.globals.mouth_mask_size = value
        modules.globals.mouth_mask = value > 0
        self._val_mouth.setText(f"{int(value)}%")
        if value <= 0:
            modules.globals.show_mouth_mask_box = False

    def _on_mouth_mask_pressed(self) -> None:
        if modules.globals.mouth_mask_size > 0:
            modules.globals.show_mouth_mask_box = True

    def _on_mouth_mask_released(self) -> None:
        modules.globals.show_mouth_mask_box = False

    def _on_start(self) -> None:
        if _MAPPER is not None and _MAPPER.isVisible():
            update_status("Please complete pop-up or close it.")
            return
        if modules.globals.map_faces:
            modules.globals.source_target_map = []
            if is_image(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_image()
            elif is_video(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_video()
            if modules.globals.source_target_map:
                _open_mapper_dialog(self._start_cb, modules.globals.source_target_map)
            else:
                update_status("No faces found in target")
        else:
            self._select_output_and_start()

    def _select_output_and_start(self) -> None:
        global _RECENT_OUTPUT_DIR
        if is_image(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self,
                _("save image output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.png"),
                _IMAGE_FILE_FILTER,
            )
        elif is_video(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self,
                _("save video output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.mp4"),
                _VIDEO_FILE_FILTER,
            )
        else:
            update_status("Please select a target image or video first")
            return
        if path:
            modules.globals.output_path = path
            _RECENT_OUTPUT_DIR = os.path.dirname(path)
            self._start_cb()

    def _on_toggle_preview(self) -> None:
        if _PREVIEW is None:
            return
        if _PREVIEW.isVisible():
            _PREVIEW.hide()
        elif modules.globals.source_path and modules.globals.target_path:
            _PREVIEW.init_for_target()
            _PREVIEW.refresh_frame(0)
            _PREVIEW.show()
        else:
            update_status("Please select both a source face and target media first")

    def _on_live(self) -> None:
        idx = self.cb_camera.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            update_status("No camera available")
            return
        camera_index = self._camera_indices[idx]
        if _LIVE_MAPPER is not None and _LIVE_MAPPER.isVisible():
            update_status("Source x Target Mapper is already open.")
            _LIVE_MAPPER.raise_()
            return
        if not modules.globals.map_faces:
            if modules.globals.source_path is None:
                update_status("Please select a source image first")
                return
            from modules.face_analyser import get_face_analyser
            from modules.processors.frame.face_swapper import get_face_swapper

            get_face_analyser()
            get_face_swapper()
            _open_webcam_preview(camera_index)
        else:
            modules.globals.source_target_map = []
            _open_live_mapper_dialog(camera_index, modules.globals.source_target_map)

    def closeEvent(self, event):  # noqa: N802
        self._destroy_cb()
        event.accept()


def _update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
        get_frame_processors_modules(modules.globals.frame_processors)


# ─── preview window (still-image / video scrub) ──────────────────────────


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("Preview"))
        icon = load_app_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(
            "background-color: #0F1216; border: 1px solid #2B3139; border-radius: 14px;"
        )
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._image_label, 1)

        # Floating scrubber dock
        self._hud = QFrame()
        self._hud.setObjectName("floatingHud")
        apply_floating_shadow(self._hud, color=QColor(0, 0, 0, 185), blur=24, y_offset=6)
        hud_row = QHBoxLayout(self._hud)
        hud_row.setContentsMargins(14, 8, 14, 8)
        hud_row.setSpacing(10)

        self._frame_pill = QLabel("Frame 0")
        self._frame_pill.setObjectName("valuePill")
        self._frame_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hud_row.addWidget(self._frame_pill)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self.refresh_frame)
        hud_row.addWidget(self._slider, 1)

        btn_refresh = FloatingButton("↻  " + _("Refresh"), base_blur=10, hover_blur=16)
        btn_refresh.setObjectName("overlayPill")
        btn_refresh.clicked.connect(lambda: self.refresh_frame(self._slider.value()))
        hud_row.addWidget(btn_refresh)

        layout.addWidget(self._hud)

    def init_for_target(self) -> None:
        if is_image(modules.globals.target_path):
            self._slider.hide()
            self._frame_pill.setText(_("Image"))
        elif is_video(modules.globals.target_path):
            total = get_video_frame_total(modules.globals.target_path)
            self._slider.setRange(0, max(0, total - 1))
            self._slider.setValue(0)
            self._slider.show()
            self._frame_pill.setText(f"0 / {max(0, total - 1)}")

    def refresh_frame(self, frame_number: int = 0) -> None:
        if not (modules.globals.source_path and modules.globals.target_path):
            return
        if self._slider.isVisible():
            self._frame_pill.setText(f"{frame_number} / {self._slider.maximum()}")
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        source_frame = imread_unicode(modules.globals.source_path)
        if source_frame is None:
            update_status(
                f"Could not read source image: {modules.globals.source_path}"
            )
            return
        from modules.processors.frame.core import get_frame_processors_modules as _gfpm

        for fp in _gfpm(modules.globals.frame_processors):
            temp_frame = fp.process_frame(get_one_face(source_frame), temp_frame)
        h, w = temp_frame.shape[:2]
        bound_w = min(PREVIEW_MAX_WIDTH, max(self.width(), PREVIEW_DEFAULT_WIDTH))
        bound_h = min(PREVIEW_MAX_HEIGHT, max(self.height(), PREVIEW_DEFAULT_HEIGHT))
        ratio = min(bound_w / w, bound_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        temp_frame = cv2.resize(temp_frame, new_size, interpolation=cv2.INTER_LANCZOS4)
        self._image_label.setPixmap(_bgr_to_qpixmap(temp_frame))
        update_status("Processing succeed!")


# ─── webcam preview window ───────────────────────────────────────────────


class _CaptureWorker(QThread):
    """Reads frames from the camera into a bounded queue. Drops on overflow."""

    def __init__(self, cap, capture_queue: queue.Queue, stop_event: threading.Event):
        super().__init__()
        self._cap = cap
        self._queue = capture_queue
        self._stop = stop_event

    def run(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._stop.set()
                break
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass


class _ProcessingWorker(QThread):
    """Pulls raw frames, runs detect/swap/enhance, pushes processed frames."""

    def __init__(self, capture_queue, processed_queue, stop_event, camera_fps: float):
        super().__init__()
        self._cq = capture_queue
        self._pq = processed_queue
        self._stop = stop_event
        self._fps = camera_fps

    def run(self) -> None:
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        async_swapper = None
        vcam_streamer: Optional[VirtualCamStreamer] = None
        reported_vcam_error: Optional[str] = None
        face_lost_at = None
        source_image = None
        last_source_path = None
        reported_source_error = None
        prev_time = time.time()
        fps_update_interval = 0.5
        frame_count = 0
        fps = 0.0
        det_count = 0
        cached_target_face = None
        cached_many_faces = None
        det_interval = max(1, round(self._fps * 0.08))

        while not self._stop.is_set():
            try:
                frame = self._cq.get(timeout=0.05)
            except queue.Empty:
                continue

            temp_frame = frame
            if modules.globals.live_mirror:
                temp_frame = gpu_flip(temp_frame, 1)

            if not modules.globals.map_faces:
                source_path = modules.globals.source_path
                if not source_path:
                    source_image = None
                    last_source_path = None
                    reported_source_error = None
                    if async_swapper is not None:
                        async_swapper.clear()
                elif source_path != last_source_path:
                    source_frame = imread_unicode(source_path)
                    if source_frame is None:
                        source_image = None
                        last_source_path = None
                        if reported_source_error != source_path:
                            reported_source_error = source_path
                            update_status(
                                f"Could not read source image: {source_path}"
                            )
                    else:
                        last_source_path = source_path
                        reported_source_error = None
                        source_image = get_one_face(source_frame)

                det_count += 1
                if det_count % det_interval == 0:
                    if modules.globals.many_faces:
                        cached_target_face = None
                        cached_many_faces = detect_many_faces_fast(temp_frame)
                    else:
                        cached_target_face = detect_one_face_fast(temp_frame)
                        cached_many_faces = None

                cached_faces = None
                if cached_many_faces:
                    cached_faces = cached_many_faces
                elif cached_target_face is not None:
                    cached_faces = [cached_target_face]

                if modules.globals.mouth_mask and cached_faces:
                    ensure_landmarks(temp_frame, cached_faces)

                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN256":
                        if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN512":
                        if modules.globals.fp_ui.get("face_enhancer_gpen512", False):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-SWAPPER":
                        swapped_bboxes = []
                        if modules.globals.many_faces and cached_many_faces:
                            result = temp_frame.copy()
                            for t_face in cached_many_faces:
                                result = fp.swap_face(source_image, t_face, result)
                                if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                    swapped_bboxes.append(t_face.bbox.astype(int))
                            temp_frame = result
                        elif cached_target_face is not None:
                            if modules.globals.live_async_swap:
                                if async_swapper is None:
                                    async_swapper = fp.AsyncLiveSwapper()
                                temp_frame = async_swapper.swap(
                                    source_image, cached_target_face, temp_frame
                                )
                            else:
                                temp_frame = fp.swap_face(
                                    source_image, cached_target_face, temp_frame
                                )
                            if (
                                hasattr(cached_target_face, "bbox")
                                and cached_target_face.bbox is not None
                            ):
                                swapped_bboxes.append(cached_target_face.bbox.astype(int))
                        temp_frame = fp.apply_post_processing(temp_frame, swapped_bboxes)
                    else:
                        temp_frame = fp.process_frame(source_image, temp_frame)

                if cached_target_face is not None:
                    face_lost_at = None
                elif face_lost_at is None:
                    face_lost_at = time.monotonic()
                if async_swapper is not None and (
                    not modules.globals.live_async_swap
                    or (
                        face_lost_at is not None
                        and time.monotonic() - face_lost_at > ASYNC_SWAP_FACE_LOSS_GRACE_S
                    )
                ):
                    async_swapper.clear()
            else:
                modules.globals.target_path = None
                cached_target_face = None
                cached_many_faces = None
                if async_swapper is not None:
                    async_swapper.clear()
                for fp in frame_processors:
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = fp.process_frame_v2(temp_frame)
                    elif fp.NAME in ("DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512"):
                        fp_key = fp.NAME.split(".")[-1].lower().replace("-", "_")
                        if modules.globals.fp_ui.get(fp_key, False):
                            temp_frame = fp.process_frame_v2(temp_frame)
                    else:
                        temp_frame = fp.process_frame_v2(temp_frame)

            # Optional desktop virtual-camera output (pre-HUD overlay)
            if modules.globals.virtual_cam:
                if vcam_streamer is None:
                    vcam_streamer = VirtualCamStreamer(self._fps)
                ok = vcam_streamer.send(temp_frame)
                if not ok and vcam_streamer.last_error and vcam_streamer.last_error != reported_vcam_error:
                    reported_vcam_error = vcam_streamer.last_error
                    update_status(vcam_streamer.last_error)
            elif vcam_streamer is not None:
                vcam_streamer.close()
                vcam_streamer = None
                reported_vcam_error = None

            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            if modules.globals.show_fps:
                cv2.putText(
                    temp_frame,
                    f"FPS: {fps:.1f}",
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (11, 185, 240),
                    2,
                )

            try:
                self._pq.put_nowait(temp_frame)
            except queue.Full:
                try:
                    self._pq.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._pq.put_nowait(temp_frame)
                except queue.Full:
                    pass

        if vcam_streamer is not None:
            vcam_streamer.close()
        if async_swapper is not None:
            async_swapper.close()


class WebcamPreviewWindow(QWidget):
    def __init__(self, camera_index: int):
        super().__init__()
        self.setWindowTitle(_("Deep Live Cam • Live Broadcast"))
        icon = load_app_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background-color: #0B0E11;")
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._image_label, 1)

        self._cap = VideoCapturer(camera_index)
        req_w, req_h = modules.globals.capture_resolution
        if not self._cap.start(req_w, req_h, 60):
            update_status("Failed to start camera")
            QTimer.singleShot(0, self.close)
            return

        camera_fps = self._cap.actual_fps
        print(
            f"[webcam] Camera running at {self._cap.actual_width}x"
            f"{self._cap.actual_height}@{camera_fps:.0f}fps"
        )

        self._capture_queue: queue.Queue = queue.Queue(maxsize=2)
        self._processed_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

        self._capture_worker = _CaptureWorker(
            self._cap, self._capture_queue, self._stop_event
        )
        self._processing_worker = _ProcessingWorker(
            self._capture_queue, self._processed_queue, self._stop_event, camera_fps
        )
        self._capture_worker.start()
        self._processing_worker.start()

        # Floating translucent live control HUD
        self._live_hud = self._build_live_hud()
        self._position_live_hud()

        poll_ms = max(1, min(16, int(500 / max(camera_fps, 1))))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(poll_ms)

    def _build_live_hud(self) -> QFrame:
        hud = QFrame(self)
        hud.setObjectName("floatingHud")
        apply_floating_shadow(hud, color=QColor(0, 0, 0, 200), blur=26, y_offset=6)
        row = QHBoxLayout(hud)
        row.setContentsMargins(12, 6, 12, 6)
        row.setSpacing(8)

        live_dot = QLabel("● LIVE")
        live_dot.setObjectName("accelPill")
        row.addWidget(live_dot)

        btn_mirror = FloatingButton(_("Mirror"), base_blur=10, hover_blur=16)
        btn_mirror.setObjectName("overlayPill")
        btn_mirror.clicked.connect(self._toggle_mirror)
        row.addWidget(btn_mirror)

        btn_async = FloatingButton(_("Async"), base_blur=10, hover_blur=16)
        btn_async.setObjectName("overlayPill")
        btn_async.clicked.connect(self._toggle_async)
        row.addWidget(btn_async)

        btn_stop = FloatingButton("■ " + _("Stop"), base_blur=10, hover_blur=16)
        btn_stop.setObjectName("danger")
        btn_stop.clicked.connect(self.close)
        row.addWidget(btn_stop)

        hud.adjustSize()
        hud.raise_()
        return hud

    def _toggle_mirror(self) -> None:
        modules.globals.live_mirror = not modules.globals.live_mirror
        if _MAIN is not None:
            _MAIN.sw_live_mirror.setChecked(modules.globals.live_mirror)
        save_switch_states()

    def _toggle_async(self) -> None:
        modules.globals.live_async_swap = not modules.globals.live_async_swap
        if _MAIN is not None:
            _MAIN.sw_async_swap.setChecked(modules.globals.live_async_swap)
        save_switch_states()

    def _position_live_hud(self) -> None:
        hud = getattr(self, "_live_hud", None)
        if hud is None:
            return
        hud.adjustSize()
        x = max(8, (self.width() - hud.width()) // 2)
        y = max(8, self.height() - hud.height() - 14)
        hud.move(x, y)
        hud.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._position_live_hud()

    def _tick(self) -> None:
        if self._stop_event.is_set():
            self.close()
            return
        try:
            bgr_frame = self._processed_queue.get_nowait()
        except queue.Empty:
            return
        bgr_frame = fit_image_to_size(bgr_frame, self.width(), self.height())
        self._image_label.setPixmap(_bgr_to_qpixmap(bgr_frame))

    def closeEvent(self, event) -> None:  # noqa: N802
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        timer = getattr(self, "_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        for name in ("_capture_worker", "_processing_worker"):
            worker = getattr(self, name, None)
            if worker is None:
                continue
            if not worker.wait(WORKER_SHUTDOWN_GRACE_MS):
                print(
                    f"[webcam] still waiting for {name[1:]} to stop...",
                    flush=True,
                )
                worker.wait()
        cap = getattr(self, "_cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        global _WEBCAM_PREVIEW
        if _WEBCAM_PREVIEW is self:
            _WEBCAM_PREVIEW = None
        event.accept()


def _open_webcam_preview(camera_index: int) -> None:
    global _WEBCAM_PREVIEW
    if _WEBCAM_PREVIEW is not None:
        _WEBCAM_PREVIEW.close()
    _WEBCAM_PREVIEW = WebcamPreviewWindow(camera_index)
    _WEBCAM_PREVIEW.show()


# ─── mapper dialogs (image/video + live) ────────────────────────────────


def _make_thumb(cv2_img: np.ndarray) -> QPixmap:
    rgb = gpu_cvt_color(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE), Image.LANCZOS
    )
    return _pil_to_qpixmap(image)


class MapperDialog(QDialog):
    """Source × Target mapper for image / video processing."""

    def __init__(self, start_cb: Callable, mapping: list):
        super().__init__(_MAIN)
        self._start_cb = start_cb
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_WIDTH, POPUP_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setObjectName("statusLabel")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_submit = FloatingButton(_("Submit"), glow_gold=True)
        btn_submit.clicked.connect(self._on_submit)
        layout.addWidget(btn_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn = FloatingButton(_("Select source image"))
            btn.setObjectName("floatingPill")
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _c, n=row: self._select_source(n))
            grid.addWidget(btn, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet(
                "background-color: #0F1216; border: 1px dashed #F0B90B; border-radius: 10px;"
            )
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("⇄")
            x_label.setStyleSheet("color: #FCD535; font-size: 14pt; font-weight: 700;")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet(
                "background-color: #0F1216; border: 1px solid #2B3139; border-radius: 10px;"
            )
            grid.addWidget(tgt_label, row, 3)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_source(self, row: int) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self,
            _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row]["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_submit(self) -> None:
        if has_valid_map():
            self.accept()
            _MAIN._select_output_and_start()
        else:
            self.set_status("Atleast 1 source with target is required!")


class LiveMapperDialog(QDialog):
    """Source × Target mapper for live webcam mode."""

    def __init__(self, camera_index: int, mapping: list):
        super().__init__(_MAIN)
        self._camera_index = camera_index
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_LIVE_WIDTH, POPUP_LIVE_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setObjectName("statusLabel")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        dock = FloatingDock()
        btn_row = QHBoxLayout(dock)
        btn_row.setContentsMargins(14, 8, 14, 8)
        btn_row.setSpacing(10)
        for text, slot, obj_name, glow in (
            (_("Add"), self._on_add, "floatingPill", False),
            (_("Clear"), self._on_clear, "danger", False),
            (_("Submit"), self._on_submit, "", True),
        ):
            b = FloatingButton(text, glow_gold=glow)
            if obj_name:
                b.setObjectName(obj_name)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addWidget(dock, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn_s = FloatingButton(_("Select source image"))
            btn_s.setObjectName("floatingPill")
            btn_s.setFixedWidth(200)
            btn_s.clicked.connect(lambda _c, n=row: self._select_face(n, "source"))
            grid.addWidget(btn_s, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet(
                "background-color: #0F1216; border: 1px dashed #F0B90B; border-radius: 10px;"
            )
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("⇄")
            x_label.setStyleSheet("color: #FCD535; font-size: 14pt; font-weight: 700;")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            btn_t = FloatingButton(_("Select target image"))
            btn_t.setObjectName("floatingPill")
            btn_t.setFixedWidth(200)
            btn_t.clicked.connect(lambda _c, n=row: self._select_face(n, "target"))
            grid.addWidget(btn_t, row, 3)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet(
                "background-color: #0F1216; border: 1px solid #2B3139; border-radius: 10px;"
            )
            grid.addWidget(tgt_label, row, 4)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_face(self, row: int, kind: str) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self,
            _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row][kind] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_add(self) -> None:
        add_blank_map()
        self._rebuild()
        self.set_status("Please provide mapping!")

    def _on_clear(self) -> None:
        for item in self._map:
            item.pop("source", None)
            item.pop("target", None)
        self._rebuild()
        self.set_status("All mappings cleared!")

    def _on_submit(self) -> None:
        if has_valid_map():
            simplify_maps()
            self.set_status("Mappings successfully submitted!")
            self.accept()
            _open_webcam_preview(self._camera_index)
        else:
            self.set_status("At least 1 source with target is required!")


def _open_mapper_dialog(start_cb: Callable, mapping: list) -> None:
    global _MAPPER
    close_mapper_window()
    _MAPPER = MapperDialog(start_cb, mapping)
    _MAPPER.show()


def _open_live_mapper_dialog(camera_index: int, mapping: list) -> None:
    global _LIVE_MAPPER
    close_mapper_window()
    _LIVE_MAPPER = LiveMapperDialog(camera_index, mapping)
    _LIVE_MAPPER.show()


def close_mapper_window() -> None:
    global _MAPPER, _LIVE_MAPPER
    if _MAPPER is not None:
        _MAPPER.close()
        _MAPPER = None
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.close()
        _LIVE_MAPPER = None


# ─── entry point ─────────────────────────────────────────────────────────


def _close_live_windows() -> None:
    """Close every window that owns worker threads, so the threads stop."""
    for win in (_WEBCAM_PREVIEW, _LIVE_MAPPER):
        if win is None:
            continue
        try:
            win.close()
        except Exception:
            pass


class _Window:
    """Thin wrapper exposing .mainloop() for core.py compatibility."""

    def __init__(self, app: QApplication, main_window: MainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        self._main.show()
        self._app.exec()


def init(
    start: Callable[[], None], destroy: Callable[[], None], lang: str
) -> _Window:
    global _APP, _MAIN, _PREVIEW, _LANG, _BRIDGE

    _LANG = LanguageManager(lang)
    if QApplication.instance() is None:
        _APP = QApplication(sys.argv)
    else:
        _APP = QApplication.instance()
    _APP.setStyleSheet(BINANCE_QSS)
    icon = load_app_icon()
    if not icon.isNull():
        _APP.setWindowIcon(icon)

    _BRIDGE = _UIBridge()

    def _destroy_with_cleanup(*args, **kwargs):
        _close_live_windows()
        return destroy(*args, **kwargs)

    _MAIN = MainWindow(start, _destroy_with_cleanup)
    _PREVIEW = PreviewWindow()

    _BRIDGE.statusChanged.connect(_MAIN.set_status)

    return _Window(_APP, _MAIN)
