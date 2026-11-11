"""Binance-inspired Obsidian Black & Gold theme, icons, and floating widgets."""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPropertyAnimation,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QIcon, QMouseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from modules.paths import ICONS_DIR


# ─── Binance Obsidian & Gold Design Tokens ────────────────────────────────

PALETTE = {
    "bg_obsidian": "#0B0E11",
    "bg_surface": "#14171D",
    "bg_card": "#181A20",
    "bg_elevated": "#1E2329",
    "bg_well": "#0F1216",
    "border_subtle": "#2B3139",
    "border_hover": "#474D57",
    "gold_primary": "#F0B90B",
    "gold_bright": "#FCD535",
    "gold_deep": "#C99400",
    "text_primary": "#EAECEF",
    "text_secondary": "#B7BDC6",
    "text_muted": "#848E9C",
    "emerald": "#0ECB81",
    "crimson": "#F6465D",
}


BINANCE_QSS = """
QMainWindow, QDialog {
    background-color: #0B0E11;
    color: #EAECEF;
}
QWidget {
    color: #EAECEF;
    font-family: "Inter", "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    font-size: 10.5pt;
}

/* Header & Surface Panels */
QFrame#headerBar {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #14171D, stop:0.65 #181A20, stop:1 #1B1810
    );
    border: 1px solid #2B3139;
    border-bottom: 1px solid rgba(240, 185, 11, 0.32);
    border-radius: 16px;
}

QFrame#studioCard, QGroupBox {
    background-color: #181A20;
    border: 1px solid #2B3139;
    border-radius: 16px;
}
QGroupBox {
    margin-top: 14px;
    padding: 18px 14px 14px 14px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 2px 10px;
    background-color: #1E2329;
    border: 1px solid rgba(240, 185, 11, 0.35);
    border-radius: 8px;
    color: #FCD535;
    font-size: 9.5pt;
    letter-spacing: 0.6px;
}

QLabel#sectionTitle {
    color: #FCD535;
    font-size: 10pt;
    font-weight: 700;
    letter-spacing: 0.8px;
    background: transparent;
}
QLabel#brandTitle {
    color: #EAECEF;
    font-size: 14pt;
    font-weight: 800;
    letter-spacing: 1.0px;
    background: transparent;
}
QLabel#brandGold {
    color: #F0B90B;
    font-size: 14pt;
    font-weight: 800;
    letter-spacing: 1.0px;
    background: transparent;
}
QLabel#badgePill {
    background-color: rgba(240, 185, 11, 0.14);
    border: 1px solid rgba(240, 185, 11, 0.48);
    border-radius: 10px;
    padding: 3px 10px;
    color: #FCD535;
    font-size: 8.5pt;
    font-weight: 700;
}
QLabel#accelPill {
    background-color: rgba(14, 203, 129, 0.12);
    border: 1px solid rgba(14, 203, 129, 0.42);
    border-radius: 10px;
    padding: 3px 10px;
    color: #0ECB81;
    font-size: 8.5pt;
    font-weight: 700;
}
QLabel#valuePill {
    background-color: #0F1216;
    border: 1px solid #2B3139;
    border-radius: 8px;
    padding: 2px 8px;
    color: #FCD535;
    font-size: 9pt;
    font-weight: 700;
    min-width: 44px;
}
QLabel#mutedHint {
    color: #848E9C;
    font-size: 9pt;
    background: transparent;
}

/* Floating Bottom Dock & HUD */
QFrame#floatingDock {
    background-color: rgba(20, 23, 29, 0.96);
    border: 1px solid rgba(240, 185, 11, 0.40);
    border-radius: 26px;
}
QFrame#floatingHud {
    background-color: rgba(11, 14, 17, 0.88);
    border: 1px solid rgba(240, 185, 11, 0.42);
    border-radius: 22px;
}

/* Floating Buttons */
QPushButton, QPushButton#goldPrimary {
    background-color: #F0B90B;
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #FCD535, stop:0.55 #F0B90B, stop:1 #D49B00
    );
    color: #0B0E11;
    border: 1px solid #FCD535;
    border-radius: 18px;
    padding: 9px 18px;
    font-weight: 700;
    font-size: 10pt;
}
QPushButton:hover, QPushButton#goldPrimary:hover {
    background-color: #FCD535;
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #FFE46B, stop:0.55 #FCD535, stop:1 #F0B90B
    );
    border-color: #FFF0A5;
}
QPushButton:pressed {
    background-color: #C99400;
    border-color: #F0B90B;
}
QPushButton:disabled {
    background: #23272F;
    color: #5E6673;
    border: 1px solid #2B3139;
}

QPushButton#secondary, QPushButton#floatingPill {
    background-color: #1E2329;
    color: #FCD535;
    border: 1px solid rgba(240, 185, 11, 0.42);
    border-radius: 18px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton#secondary:hover, QPushButton#floatingPill:hover {
    background-color: #2B3139;
    border: 1px solid #FCD535;
    color: #FFF0A5;
}
QPushButton#secondary:pressed, QPushButton#floatingPill:pressed {
    background-color: #14171D;
}

QPushButton#liveAccent {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #143328, stop:1 #0D231C
    );
    color: #0ECB81;
    border: 1px solid rgba(14, 203, 129, 0.72);
    border-radius: 18px;
    padding: 9px 18px;
    font-weight: 700;
}
QPushButton#liveAccent:hover {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #184737, stop:1 #113026
    );
    border-color: #2EE59D;
    color: #70F3BE;
}
QPushButton#liveAccent:disabled {
    background: #1B1F26;
    color: #5E6673;
    border-color: #2B3139;
}

QPushButton#danger {
    background-color: #261419;
    color: #FF7588;
    border: 1px solid rgba(246, 70, 93, 0.55);
    border-radius: 18px;
    padding: 9px 16px;
    font-weight: 700;
}
QPushButton#danger:hover {
    background-color: #F6465D;
    color: #FFFFFF;
    border-color: #FF7588;
}

QPushButton#overlayPill {
    background-color: rgba(11, 14, 17, 0.90);
    color: #EAECEF;
    border: 1px solid rgba(240, 185, 11, 0.48);
    border-radius: 15px;
    padding: 6px 12px;
    font-size: 9pt;
    font-weight: 600;
}
QPushButton#overlayPill:hover {
    background-color: #2B3139;
    border-color: #FCD535;
    color: #FCD535;
}
QPushButton#miniPill {
    background-color: #1E2329;
    color: #FCD535;
    border: 1px solid rgba(240, 185, 11, 0.45);
    border-radius: 16px;
    padding: 7px 10px;
    font-size: 9.2pt;
    font-weight: 700;
}
QPushButton#miniPill:hover {
    background-color: #2B3139;
    border-color: #FCD535;
    color: #FFF0A5;
}

QPushButton#iconCircle {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #2B3139, stop:1 #181A20
    );
    color: #FCD535;
    border: 1.5px solid #F0B90B;
    border-radius: 22px;
    padding: 0px;
    font-size: 13pt;
    font-weight: 700;
}
QPushButton#iconCircle:hover {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #FCD535, stop:1 #F0B90B
    );
    color: #0B0E11;
}

/* Inputs & Dropdowns */
QLineEdit {
    background-color: #0F1216;
    color: #EAECEF;
    border: 1px solid #2B3139;
    border-radius: 10px;
    padding: 7px 12px;
    selection-background-color: #F0B90B;
    selection-color: #0B0E11;
}
QLineEdit:focus {
    border-color: #F0B90B;
}
QComboBox {
    background-color: #0F1216;
    color: #EAECEF;
    border: 1px solid #2B3139;
    border-radius: 10px;
    padding: 7px 12px;
    min-height: 24px;
    font-weight: 500;
}
QComboBox:hover, QComboBox:focus {
    border-color: #F0B90B;
}
QComboBox::drop-down {
    border: none;
    width: 26px;
}
QComboBox QAbstractItemView {
    background-color: #181A20;
    color: #EAECEF;
    selection-background-color: #F0B90B;
    selection-color: #0B0E11;
    border: 1px solid #F0B90B;
    border-radius: 8px;
    padding: 4px;
}

/* Switches */
QCheckBox {
    spacing: 10px;
    padding: 4px 2px;
    color: #EAECEF;
    font-size: 9.8pt;
}
QCheckBox::indicator {
    width: 38px;
    height: 20px;
    border-radius: 10px;
    background-color: #2B3139;
    border: 1px solid #3C434D;
}
QCheckBox::indicator:hover {
    border-color: #F0B90B;
}
QCheckBox::indicator:checked {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #D49B00, stop:1 #FCD535
    );
    border: 1px solid #FCD535;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 6px;
    background: #2B3139;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #C99400, stop:1 #FCD535
    );
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #FCD535;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
    border: 2px solid #0B0E11;
}
QSlider::handle:horizontal:hover {
    background: #FFF0A5;
    border-color: #F0B90B;
}

/* Media Drop Zone */
QFrame#mediaCard {
    background-color: #14171D;
    border: 1px solid #2B3139;
    border-radius: 16px;
}
QLabel#imageDrop {
    background-color: #0F1216;
    border: 1.5px dashed #3C434D;
    border-radius: 12px;
    color: #848E9C;
    font-weight: 600;
    padding: 8px;
}
QLabel#imageDrop[hasMedia="true"] {
    border: 1.5px solid rgba(240, 185, 11, 0.55);
    background-color: #0B0E11;
    padding: 2px;
}
QLabel#imageDrop[dragActive="true"] {
    border: 2px dashed #FCD535;
    background-color: rgba(240, 185, 11, 0.08);
    color: #FCD535;
}

/* Status & Links */
QLabel#statusLabel {
    color: #B7BDC6;
    font-size: 9.5pt;
    background-color: #14171D;
    border: 1px solid #2B3139;
    border-radius: 12px;
    padding: 6px 14px;
}
QLabel#linkLabel {
    color: #F0B90B;
    font-size: 9pt;
    font-weight: 600;
    background: transparent;
}
QLabel#linkLabel:hover {
    color: #FCD535;
    text-decoration: underline;
}

/* ScrollArea & Scrollbars */
QScrollArea, QWidget#studioWorkspace {
    border: none;
    background-color: #0B0E11;
}
QScrollBar:vertical {
    background: #0B0E11;
    width: 8px;
    margin: 4px 0;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #2B3139;
    min-height: 28px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #F0B90B;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background: #0B0E11;
    height: 8px;
    margin: 0 4px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #2B3139;
    min-width: 28px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover {
    background: #F0B90B;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

QToolTip {
    background-color: #181A20;
    color: #EAECEF;
    border: 1px solid #F0B90B;
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 9.5pt;
}
"""


def load_app_icon() -> QIcon:
    """Load the multi-resolution Binance Black & Gold application icon."""
    icon = QIcon()
    candidates = [
        "app_icon_16.png",
        "app_icon_32.png",
        "app_icon_48.png",
        "app_icon_64.png",
        "app_icon_128.png",
        "app_icon_256.png",
        "app_icon_512.png",
    ]
    for filename in candidates:
        path = os.path.join(ICONS_DIR, filename)
        if os.path.isfile(path):
            icon.addFile(path)
    if icon.isNull():
        ico_path = os.path.join(ICONS_DIR, "win", "app_icon.ico")
        if os.path.isfile(ico_path):
            icon.addFile(ico_path)
    return icon


def apply_floating_shadow(
    widget: QWidget,
    color: Optional[QColor] = None,
    blur: int = 24,
    y_offset: int = 6,
) -> QGraphicsDropShadowEffect:
    """Attach an elevated drop shadow so cards and buttons visibly float."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, y_offset)
    shadow.setColor(color or QColor(0, 0, 0, 175))
    widget.setGraphicsEffect(shadow)
    return shadow


class FloatingButton(QPushButton):
    """Pill push button with elevated drop shadow and smooth hover lift."""

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        glow_gold: bool = False,
        base_blur: int = 18,
        hover_blur: int = 28,
    ) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._base_blur = base_blur
        self._hover_blur = hover_blur
        self._base_offset = 4.0
        self._hover_offset = 7.0
        shadow_color = QColor(240, 185, 11, 95) if glow_gold else QColor(0, 0, 0, 180)
        self._shadow = apply_floating_shadow(
            self, color=shadow_color, blur=base_blur, y_offset=int(self._base_offset)
        )
        self._anim = QPropertyAnimation(self._shadow, b"blurRadius", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def enterEvent(self, event: QEvent) -> None:
        super().enterEvent(event)
        if not self.isEnabled():
            return
        self._shadow.setOffset(0, self._hover_offset)
        self._anim.stop()
        self._anim.setStartValue(self._shadow.blurRadius())
        self._anim.setEndValue(self._hover_blur)
        self._anim.start()

    def leaveEvent(self, event: QEvent) -> None:
        super().leaveEvent(event)
        self._shadow.setOffset(0, self._base_offset)
        self._anim.stop()
        self._anim.setStartValue(self._shadow.blurRadius())
        self._anim.setEndValue(self._base_blur)
        self._anim.start()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._shadow.setOffset(0, 2.0)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._shadow.setOffset(0, self._hover_offset if self.underMouse() else self._base_offset)
        super().mouseReleaseEvent(event)


class FloatingDock(QFrame):
    """Elevated obsidian-and-gold pill dock for floating primary actions."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("floatingDock")
        apply_floating_shadow(
            self,
            color=QColor(0, 0, 0, 210),
            blur=32,
            y_offset=10,
        )


class MediaDropZone(QLabel):
    """Interactive drag-and-drop image/video preview frame with click-to-browse."""

    fileDropped = Signal(str)
    clicked = Signal()

    def __init__(self, placeholder: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(placeholder, parent)
        self._placeholder = placeholder
        self.setObjectName("imageDrop")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("hasMedia", "false")
        self.setProperty("dragActive", "false")

    def _repolish(self) -> None:
        self.style().unpolish(self)
        self.style().polish(self)

    def set_media_active(self, active: bool) -> None:
        val = "true" if active else "false"
        if self.property("hasMedia") != val:
            self.setProperty("hasMedia", val)
            self._repolish()

    def setPixmap(self, pixmap) -> None:  # noqa: N802
        super().setPixmap(pixmap)
        self.set_media_active(pixmap is not None and not pixmap.isNull())

    def clear_media(self) -> None:
        self.clear()
        self.setText(self._placeholder)
        self.set_media_active(False)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    self.setProperty("dragActive", "true")
                    self._repolish()
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragLeaveEvent(self, event: QEvent) -> None:
        self.setProperty("dragActive", "false")
        self._repolish()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        self.setProperty("dragActive", "false")
        self._repolish()
        urls = event.mimeData().urls()
        if urls:
            local = urls[0].toLocalFile()
            if local:
                self.fileDropped.emit(local)
                event.acceptProposedAction()
                return
        event.ignore()


class StudioSwitch(QWidget):
    """Compact toggle switch with label and hover tooltip."""

    toggled = Signal(bool)

    def __init__(self, text: str, initial: bool, tooltip: str = "") -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        self._checkbox = QCheckBox(text)
        self._checkbox.setChecked(initial)
        self._checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
        self._checkbox.toggled.connect(self.toggled.emit)
        if tooltip:
            self._checkbox.setToolTip(tooltip)
        layout.addWidget(self._checkbox)
        layout.addStretch(1)

    def isChecked(self) -> bool:  # noqa: N802
        return self._checkbox.isChecked()

    def setChecked(self, value: bool) -> None:  # noqa: N802
        self._checkbox.setChecked(value)
