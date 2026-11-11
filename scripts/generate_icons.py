#!/usr/bin/env python3
"""Generate Deep-Live-Cam Binance Black & Gold brand logos and platform icons.

Produces:
  - assets/icons/app_icon.svg & logo.svg
  - assets/icons/app_icon.png (1024x1024) + 512/256/128/64/32 px PNGs
  - assets/icons/win/app_icon.ico (multi-size Windows x64 icon)
  - assets/icons/mac/app_icon.icns (macOS x64 icon)
  - assets/icons/linux/app_icon.png & app_icon.svg (Linux x64 icon)
  - assets/icons/android/mipmap-*/ic_launcher*.png (Android adaptive & legacy icons)
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ICONS_DIR = ROOT / "assets" / "icons"

# Binance Black & Gold palette
OBSIDIAN_TOP = np.array([28, 32, 40], dtype=np.float32)
OBSIDIAN_BOT = np.array([9, 12, 15], dtype=np.float32)
GOLD_LIGHT = np.array([254, 224, 92], dtype=np.float32)
GOLD_MID = np.array([240, 185, 11], dtype=np.float32)
GOLD_DEEP = np.array([175, 122, 0], dtype=np.float32)
EMERALD = np.array([14, 203, 129], dtype=np.float32)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _sd_squircle(px: np.ndarray, py: np.ndarray, radius: float, power: float = 4.6) -> np.ndarray:
    """Superellipse signed-distance approximation centered at origin."""
    ax = np.abs(px) / radius
    ay = np.abs(py) / radius
    d = (ax ** power + ay ** power) ** (1.0 / power)
    return (d - 1.0) * radius


def _gold_gradient(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Metallic Binance gold sheen across normalized coordinates [-1, 1]."""
    t = np.clip(0.5 + 0.42 * (u * 0.7 + v * 0.7), 0.0, 1.0)
    # Two-stop blend: GOLD_LIGHT -> GOLD_MID -> GOLD_DEEP
    low = t < 0.55
    w1 = np.clip(t / 0.55, 0.0, 1.0)[..., None]
    w2 = np.clip((t - 0.55) / 0.45, 0.0, 1.0)[..., None]
    c1 = GOLD_LIGHT * (1.0 - w1) + GOLD_MID * w1
    c2 = GOLD_MID * (1.0 - w2) + GOLD_DEEP * w2
    col = np.where(low[..., None], c1, c2)
    # Subtle diagonal specular highlight band
    band = np.exp(-((u + v + 0.25) ** 2) / 0.09) * 0.22
    col = np.clip(col + band[..., None] * 255.0, 0.0, 255.0)
    return col


def render_app_icon(size: int = 1024, mask_squircle: bool = True) -> Image.Image:
    """Render the high-DPI Binance Black & Gold Deep-Live-Cam icon."""
    ss = 2048  # Supersample canvas for razor-sharp edges
    coords = np.linspace(-1.0, 1.0, ss, dtype=np.float32)
    xv, yv = np.meshgrid(coords, coords)
    r = np.sqrt(xv * xv + yv * yv)

    canvas = np.zeros((ss, ss, 4), dtype=np.float32)

    # 1. Background squircle or full bleed (for Android adaptive foreground/full)
    if mask_squircle:
        sd_bg = _sd_squircle(xv, yv, 0.88, power=4.8)
        bg_alpha = 1.0 - _smoothstep(-0.004, 0.004, sd_bg)
    else:
        sd_bg = np.full_like(xv, -1.0)
        bg_alpha = np.ones_like(xv)

    # Rich obsidian radial + vertical gradient with subtle gold ambient glow
    grad_t = np.clip((yv + 0.9) / 1.8, 0.0, 1.0)[..., None]
    bg_rgb = OBSIDIAN_TOP * (1.0 - grad_t) + OBSIDIAN_BOT * grad_t
    radial_glow = np.exp(-(r * r) / 0.28)[..., None]
    bg_rgb = np.clip(bg_rgb + radial_glow * np.array([42.0, 31.0, 8.0]), 0.0, 255.0)

    canvas[..., :3] = bg_rgb
    canvas[..., 3] = bg_alpha * 255.0

    # Gold outer chamfered rim inside the squircle
    gold_rgb = _gold_gradient(xv, yv)
    if mask_squircle:
        rim_mask = (
            _smoothstep(-0.036, -0.029, sd_bg)
            * (1.0 - _smoothstep(-0.009, -0.002, sd_bg))
        )
        canvas[..., :3] = canvas[..., :3] * (1.0 - rim_mask[..., None]) + gold_rgb * rim_mask[..., None]

        # Subtle inner hairline accent
        hair_mask = (
            _smoothstep(-0.068, -0.064, sd_bg)
            * (1.0 - _smoothstep(-0.059, -0.055, sd_bg))
            * 0.26
        )
        canvas[..., :3] = canvas[..., :3] * (1.0 - hair_mask[..., None]) + gold_rgb * hair_mask[..., None]

    # 2. Rotated Binance-style diamond aperture frame
    diamond_d = np.abs(xv) + np.abs(yv)
    diamond_ring = (
        _smoothstep(0.60, 0.612, diamond_d)
        * (1.0 - _smoothstep(0.655, 0.667, diamond_d))
        * 0.38
    )
    canvas[..., :3] = canvas[..., :3] * (1.0 - diamond_ring[..., None]) + gold_rgb * diamond_ring[..., None]

    # Corner diamond nodes (top, bottom, left, right)
    for cx, cy in ((0.0, -0.635), (0.0, 0.635), (-0.635, 0.0), (0.635, 0.0)):
        nd = np.abs(xv - cx) + np.abs(yv - cy)
        node_m = 1.0 - _smoothstep(0.055, 0.064, nd)
        canvas[..., :3] = canvas[..., :3] * (1.0 - node_m[..., None]) + gold_rgb * node_m[..., None]

    # 3. Precision camera shutter / lens ring
    outer_ring = (
        _smoothstep(0.41, 0.42, r)
        * (1.0 - _smoothstep(0.468, 0.478, r))
    )
    # Break the ring subtly at 45-degree diagonals for an aperture/tech look
    angle = np.arctan2(yv, xv)
    gap = (
        (np.abs(np.sin(2.0 * angle)) > 0.965)
        & (r > 0.40)
        & (r < 0.49)
    ).astype(np.float32)
    outer_ring = outer_ring * (1.0 - 0.85 * gap)
    canvas[..., :3] = canvas[..., :3] * (1.0 - outer_ring[..., None]) + gold_rgb * outer_ring[..., None]

    # 4. Dual interlocking AI face-swap emblem inside the lens
    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGBA")
    overlay = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    c = ss // 2
    # Smooth curved portrait contour scaled comfortably inside the aperture ring
    # Left solid golden face half
    left_poly = [
        (c - int(ss * 0.012), c - int(ss * 0.165)),
        (c - int(ss * 0.068), c - int(ss * 0.158)),
        (c - int(ss * 0.114), c - int(ss * 0.125)),
        (c - int(ss * 0.138), c - int(ss * 0.068)),
        (c - int(ss * 0.140), c + int(ss * 0.005)),
        (c - int(ss * 0.126), c + int(ss * 0.075)),
        (c - int(ss * 0.095), c + int(ss * 0.126)),
        (c - int(ss * 0.048), c + int(ss * 0.158)),
        (c - int(ss * 0.012), c + int(ss * 0.165)),
    ]
    draw.polygon(left_poly, fill=(246, 200, 40, 248))

    # Carve left eye visor slit in deep obsidian
    eye_l = [
        (c - int(ss * 0.102), c - int(ss * 0.036)),
        (c - int(ss * 0.038), c - int(ss * 0.018)),
        (c - int(ss * 0.044), c + int(ss * 0.006)),
        (c - int(ss * 0.096), c - int(ss * 0.006)),
    ]
    draw.polygon(eye_l, fill=(14, 17, 22, 255))

    # Right half: Swapped wireframe/scan contour
    right_pts = [
        (c + int(ss * 0.012), c - int(ss * 0.165)),
        (c + int(ss * 0.068), c - int(ss * 0.158)),
        (c + int(ss * 0.114), c - int(ss * 0.125)),
        (c + int(ss * 0.138), c - int(ss * 0.068)),
        (c + int(ss * 0.140), c + int(ss * 0.005)),
        (c + int(ss * 0.126), c + int(ss * 0.075)),
        (c + int(ss * 0.095), c + int(ss * 0.126)),
        (c + int(ss * 0.048), c + int(ss * 0.158)),
        (c + int(ss * 0.012), c + int(ss * 0.165)),
    ]
    draw.line(right_pts + [right_pts[0]], fill=(252, 213, 53, 255), width=int(ss * 0.019), joint="curve")

    # Right eye luminous gold accent
    eye_r = [
        (c + int(ss * 0.102), c - int(ss * 0.036)),
        (c + int(ss * 0.038), c - int(ss * 0.018)),
        (c + int(ss * 0.044), c + int(ss * 0.006)),
        (c + int(ss * 0.096), c - int(ss * 0.006)),
    ]
    draw.polygon(eye_r, fill=(252, 213, 53, 255))

    # Neural scan bars on right cheek
    for y_off, w_bar in ((0.042, 0.074), (0.077, 0.056), (0.112, 0.034)):
        y0 = c + int(ss * y_off)
        draw.rounded_rectangle(
            [c + int(ss * 0.030), y0, c + int(ss * (0.030 + w_bar)), y0 + int(ss * 0.015)],
            radius=int(ss * 0.0075),
            fill=(240, 185, 11, 225),
        )

    # Center vertical laser split-line
    draw.rounded_rectangle(
        [c - int(ss * 0.007), c - int(ss * 0.205), c + int(ss * 0.007), c + int(ss * 0.205)],
        radius=int(ss * 0.007),
        fill=(255, 240, 165, 255),
    )

    # Live status emerald beacon inside top-right rim
    bx = c + int(ss * 0.245)
    by = c - int(ss * 0.245)
    br_outer = int(ss * 0.040)
    br_inner = int(ss * 0.024)
    draw.ellipse([bx - br_outer, by - br_outer, bx + br_outer, by + br_outer], fill=(11, 14, 17, 255))
    draw.ellipse([bx - br_inner, by - br_inner, bx + br_inner, by + br_inner], fill=(14, 203, 129, 255))

    # Soft golden glow behind the emblem
    glow_layer = overlay.filter(ImageFilter.GaussianBlur(radius=int(ss * 0.014)))
    glow_np = np.asarray(glow_layer, dtype=np.float32)
    glow_np[..., 3] *= 0.35
    glow_layer = Image.fromarray(np.clip(glow_np, 0, 255).astype(np.uint8), mode="RGBA")

    img = Image.alpha_composite(img, glow_layer)
    img = Image.alpha_composite(img, overlay)

    if mask_squircle:
        # Re-apply clean squircle alpha mask so glow never spills outside icon bounds
        alpha_arr = np.asarray(img, dtype=np.float32)[..., 3]
        alpha_arr = np.minimum(alpha_arr, bg_alpha * 255.0)
        rgba = np.asarray(img).copy()
        rgba[..., 3] = np.clip(alpha_arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(rgba, mode="RGBA")

    if size != ss:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def render_wide_logo(width: int = 1200, height: int = 320) -> Image.Image:
    """Render a wide horizontal brand banner for README / Splash / Header."""
    bg = Image.new("RGBA", (width, height), (11, 14, 17, 255))
    draw = ImageDraw.Draw(bg)
    # Subtle golden border frame
    draw.rounded_rectangle(
        [4, 4, width - 5, height - 5],
        radius=28,
        fill=(18, 21, 26, 255),
        outline=(240, 185, 11, 120),
        width=3,
    )
    icon = render_app_icon(240)
    bg.alpha_composite(icon, (44, (height - 240) // 2))
    return bg


def write_svg_files() -> None:
    app_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="obsidian" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#1C2028"/>
      <stop offset="100%" stop-color="#090C0F"/>
    </linearGradient>
    <linearGradient id="gold" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FEE05C"/>
      <stop offset="55%" stop-color="#F0B90B"/>
      <stop offset="100%" stop-color="#AF7A00"/>
    </linearGradient>
    <radialGradient id="amberGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#F0B90B" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="#0B0E11" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect x="28" y="28" width="456" height="456" rx="108" fill="url(#obsidian)" stroke="url(#gold)" stroke-width="10"/>
  <rect x="44" y="44" width="424" height="424" rx="94" fill="url(#amberGlow)" stroke="#F0B90B" stroke-opacity="0.28" stroke-width="2"/>
  <polygon points="256,94 418,256 256,418 94,256" fill="none" stroke="url(#gold)" stroke-opacity="0.45" stroke-width="8"/>
  <polygon points="256,78 272,94 256,110 240,94" fill="url(#gold)"/>
  <polygon points="256,402 272,418 256,434 240,418" fill="url(#gold)"/>
  <polygon points="78,256 94,240 110,256 94,272" fill="url(#gold)"/>
  <polygon points="402,256 418,240 434,256 418,272" fill="url(#gold)"/>
  <circle cx="256" cy="256" r="114" fill="none" stroke="url(#gold)" stroke-width="14" stroke-dasharray="146 33"/>
  <path d="M250,146 L200,156 L169,210 L168,268 L190,329 L235,366 L250,366 Z" fill="url(#gold)"/>
  <polygon points="192,235 232,245 228,260 196,253" fill="#0E1116"/>
  <path d="M262,146 L312,156 L343,210 L344,268 L322,329 L277,366 L262,366 Z" fill="none" stroke="url(#gold)" stroke-width="12" stroke-linejoin="round"/>
  <polygon points="320,235 280,245 284,260 316,253" fill="url(#gold)"/>
  <rect x="274" y="284" width="48" height="9" rx="4.5" fill="url(#gold)"/>
  <rect x="274" y="306" width="36" height="9" rx="4.5" fill="url(#gold)" opacity="0.88"/>
  <rect x="274" y="328" width="22" height="9" rx="4.5" fill="url(#gold)" opacity="0.75"/>
  <rect x="251.5" y="126" width="9" height="260" rx="4.5" fill="#FFF0A5"/>
  <circle cx="381" cy="131" r="21" fill="#0B0E11"/>
  <circle cx="381" cy="131" r="12" fill="#0ECB81"/>
</svg>
"""
    logo_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 260" width="1120" height="260">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#14171D"/>
      <stop offset="100%" stop-color="#0B0E11"/>
    </linearGradient>
    <linearGradient id="goldText" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#FCD535"/>
      <stop offset="60%" stop-color="#F0B90B"/>
      <stop offset="100%" stop-color="#D49B00"/>
    </linearGradient>
  </defs>
  <rect x="4" y="4" width="1112" height="252" rx="34" fill="url(#bg)" stroke="#F0B90B" stroke-opacity="0.45" stroke-width="3"/>
  <g transform="translate(26, 20) scale(0.43)">
    <rect x="28" y="28" width="456" height="456" rx="108" fill="#0B0E11" stroke="url(#goldText)" stroke-width="12"/>
    <polygon points="256,94 418,256 256,418 94,256" fill="none" stroke="url(#goldText)" stroke-opacity="0.45" stroke-width="8"/>
    <circle cx="256" cy="256" r="114" fill="none" stroke="url(#goldText)" stroke-width="14"/>
    <path d="M250,146 L200,156 L169,210 L168,268 L190,329 L235,366 L250,366 Z" fill="url(#goldText)"/>
    <path d="M262,146 L312,156 L343,210 L344,268 L322,329 L277,366 L262,366 Z" fill="none" stroke="url(#goldText)" stroke-width="12" stroke-linejoin="round"/>
    <rect x="251.5" y="126" width="9" height="260" rx="4.5" fill="#FFF0A5"/>
    <circle cx="381" cy="131" r="14" fill="#0ECB81"/>
  </g>
  <text x="275" y="124" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="68" font-weight="800" letter-spacing="4" fill="#EAECEF">DEEP LIVE <tspan fill="url(#goldText)">CAM</tspan></text>
  <rect x="278" y="150" width="235" height="42" rx="21" fill="#F0B90B" fill-opacity="0.14" stroke="#F0B90B" stroke-opacity="0.6" stroke-width="2"/>
  <circle cx="304" cy="171" r="6" fill="#0ECB81"/>
  <text x="322" y="178" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="20" font-weight="700" letter-spacing="2" fill="#FCD535">GOLD STUDIO EDITION</text>
</svg>
"""
    for dest in (ICONS_DIR / "app_icon.svg", ICONS_DIR / "linux" / "app_icon.svg"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(app_svg, encoding="utf-8")
    (ICONS_DIR / "logo.svg").write_text(logo_svg, encoding="utf-8")


def write_icns(master: Image.Image, dest: Path) -> None:
    """Build a multi-resolution Apple ICNS file containing PNG icon chunks."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Standard macOS ICNS PNG types
    types = [
        (b"icp4", 16),
        (b"icp5", 32),
        (b"icp6", 64),
        (b"ic07", 128),
        (b"ic08", 256),
        (b"ic09", 512),
        (b"ic10", 1024),
    ]
    import io

    chunks = []
    for ostype, px in types:
        resized = master.resize((px, px), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        data = buf.getvalue()
        chunks.append(ostype + struct.pack(">I", 8 + len(data)) + data)
    body = b"".join(chunks)
    header = b"icns" + struct.pack(">I", 8 + len(body))
    dest.write_bytes(header + body)


def write_android_icons(master_full: Image.Image, master_round_src: Image.Image) -> None:
    densities = {
        "mdpi": 48,
        "hdpi": 72,
        "xhdpi": 96,
        "xxhdpi": 144,
        "xxxhdpi": 192,
    }
    android_dir = ICONS_DIR / "android"
    for name, px in densities.items():
        d = android_dir / f"mipmap-{name}"
        d.mkdir(parents=True, exist_ok=True)
        # Legacy square squircle icon
        master_round_src.resize((px, px), Image.Resampling.LANCZOS).save(d / "ic_launcher.png")
        # Circular launcher icon (4x supersampled mask for crisp anti-aliasing)
        circ_hi = master_full.resize((px * 4, px * 4), Image.Resampling.LANCZOS).copy()
        mask_hi = Image.new("L", (px * 4, px * 4), 0)
        ImageDraw.Draw(mask_hi).ellipse([2, 2, px * 4 - 3, px * 4 - 3], fill=255)
        circ_hi.putalpha(mask_hi)
        circ_hi.resize((px, px), Image.Resampling.LANCZOS).save(d / "ic_launcher_round.png")
        # Adaptive foreground (108dp canvas with safe 72dp inset)
        fg_px = int(round(px * 108 / 48))
        inset = int(round(fg_px * 0.66))
        fg = Image.new("RGBA", (fg_px, fg_px), (0, 0, 0, 0))
        emblem = master_round_src.resize((inset, inset), Image.Resampling.LANCZOS)
        off = (fg_px - inset) // 2
        fg.alpha_composite(emblem, (off, off))
        fg.save(d / "ic_launcher_foreground.png")

    anydpi = android_dir / "mipmap-anydpi-v26"
    anydpi.mkdir(parents=True, exist_ok=True)
    adaptive_xml = """<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@color/ic_launcher_background" />
    <foreground android:drawable="@mipmap/ic_launcher_foreground" />
</adaptive-icon>
"""
    (anydpi / "ic_launcher.xml").write_text(adaptive_xml, encoding="utf-8")
    (anydpi / "ic_launcher_round.xml").write_text(adaptive_xml, encoding="utf-8")
    values = android_dir / "values"
    values.mkdir(parents=True, exist_ok=True)
    (values / "ic_launcher_background.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<resources>\n    <color name="ic_launcher_background">#0B0E11</color>\n</resources>\n',
        encoding="utf-8",
    )


def main() -> None:
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    (ICONS_DIR / "win").mkdir(parents=True, exist_ok=True)
    (ICONS_DIR / "mac").mkdir(parents=True, exist_ok=True)
    (ICONS_DIR / "linux").mkdir(parents=True, exist_ok=True)

    master = render_app_icon(1024, mask_squircle=True)
    full_bleed = render_app_icon(1024, mask_squircle=False)

    master.save(ICONS_DIR / "app_icon.png")
    for s in (512, 256, 128, 64, 48, 32, 16):
        master.resize((s, s), Image.Resampling.LANCZOS).save(ICONS_DIR / f"app_icon_{s}.png")

    # Windows x64 .ico (multi-size)
    master.save(
        ICONS_DIR / "win" / "app_icon.ico",
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )

    # macOS x64 .icns
    write_icns(master, ICONS_DIR / "mac" / "app_icon.icns")

    # Linux x64 PNG (512 & 256)
    master.resize((512, 512), Image.Resampling.LANCZOS).save(ICONS_DIR / "linux" / "app_icon.png")
    master.resize((256, 256), Image.Resampling.LANCZOS).save(ICONS_DIR / "linux" / "app_icon_256.png")

    # Android mipmap + adaptive XMLs
    write_android_icons(full_bleed, master)

    # SVG master & horizontal logo
    write_svg_files()

    print("Generated icon suite in", ICONS_DIR)


if __name__ == "__main__":
    main()
