"""
color.py — deterministic color engine.

Two layers:
  1. Pure color math (conversions, WCAG luminance/contrast). This is exact,
     standards-based, and needs no tuning.
  2. ThemeResolver: seed color -> full semantic palette + the 12 PowerPoint
     theme slots, with contrast GUARANTEED (not merely attempted).

Notes on honesty:
  - We work in HSL for hue rotation and lightness ramps. HSL is NOT
    perceptually uniform: equal steps in L do not look equally spaced,
    and rotating hue changes perceived lightness. For production polish,
    swap the HSL functions for OKLCh/HSLuv (see `coloraide`). The public
    API here stays the same, so that swap is localized.
  - The WCAG 2.x contrast math below IS exact and is the legal standard.
"""
from __future__ import annotations
import colorsys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. Pure color math
# ---------------------------------------------------------------------------

RGB = tuple[int, int, int]


def hex_to_rgb(h: str) -> RGB:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hex(rgb: RGB) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, round(c))) for c in rgb))


def rgb_to_hsl(rgb: RGB) -> tuple[float, float, float]:
    r, g, b = (c / 255 for c in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)  # note: colorsys is HLS order
    return (h * 360.0, s, l)


def hsl_to_rgb(h: float, s: float, l: float) -> RGB:
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360.0, l, s)
    return (round(r * 255), round(g * 255), round(b * 255))


def rotate_hue(rgb: RGB, degrees: float) -> RGB:
    h, s, l = rgb_to_hsl(rgb)
    return hsl_to_rgb(h + degrees, s, l)


def with_lightness(rgb: RGB, l: float) -> RGB:
    h, s, _ = rgb_to_hsl(rgb)
    return hsl_to_rgb(h, s, max(0.0, min(1.0, l)))


def _linearize(c: float) -> float:
    """sRGB channel (0..1) -> linear. Threshold per W3C WCAG relative luminance."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: RGB) -> float:
    r, g, b = (_linearize(c / 255) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: RGB, b: RGB) -> float:
    """WCAG 2.x contrast ratio, 1.0 .. 21.0."""
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# WCAG 2.1 SC 1.4.3 thresholds
AA_NORMAL = 4.5
AA_LARGE = 3.0
AAA_NORMAL = 7.0

_NEAR_BLACK: RGB = (17, 17, 17)
_WHITE: RGB = (255, 255, 255)


def best_text_color(bg: RGB, candidates: list[RGB] | None = None) -> tuple[RGB, float]:
    """Pick the candidate text color with the highest contrast against `bg`."""
    cands = candidates or [_WHITE, _NEAR_BLACK]
    scored = [(c, contrast_ratio(c, bg)) for c in cands]
    return max(scored, key=lambda t: t[1])


def ensure_on_contrast(fill: RGB, target: float = AA_NORMAL) -> RGB:
    """
    Nudge a FILL color's lightness until some grayscale text (white or near-black)
    clears `target` against it. This makes the contrast guarantee real instead of
    best-effort. We move away from the crossover point of the two contrast curves:
    if white currently wins we darken further; else we lighten.
    """
    rgb = fill
    for _ in range(60):
        c_white = contrast_ratio(_WHITE, rgb)
        c_black = contrast_ratio(_NEAR_BLACK, rgb)
        if max(c_white, c_black) >= target:
            return rgb
        h, s, l = rgb_to_hsl(rgb)
        l = max(0.0, l - 0.02) if c_white >= c_black else min(1.0, l + 0.02)
        rgb = hsl_to_rgb(h, s, l)
    return rgb


# ---------------------------------------------------------------------------
# 2. ThemeResolver
# ---------------------------------------------------------------------------

@dataclass
class Theme:
    # Semantic tokens (what the layout engine references, never raw hex)
    tokens: dict[str, str] = field(default_factory=dict)
    # 12 PowerPoint theme slots (what gets written into theme1.xml)
    ppt_slots: dict[str, str] = field(default_factory=dict)
    # Diagnostics: every enforced pairing and its measured ratio
    contrast_report: list[tuple[str, str, float]] = field(default_factory=list)

    def hex(self, token: str) -> str:
        return self.tokens[token]


class ThemeResolver:
    """
    seed hex -> Theme. Deterministic. Every text/surface pairing is verified
    >= AA and repaired if needed.

    Scheme controls how accents are derived from the seed hue:
      'analogous'  -> primary, +30, -30, +60, -60, +15  (calm, cohesive)
      'triadic'    -> primary, +120, -120, +150, +90, +30
      'complement' -> primary, +180, +30, -30, +150, +210
    """

    def __init__(self, seed: str = "#1D4ED8", scheme: str = "analogous",
                 dark: bool = False):
        self.seed = hex_to_rgb(seed)
        self.scheme = scheme
        self.dark = dark

    def _accent_hues(self) -> list[float]:
        return {
            "analogous": [0, 30, -30, 60, -60, 15],
            "triadic": [0, 120, -120, 150, 90, 30],
            "complement": [0, 180, 30, -30, 150, 210],
        }[self.scheme]

    def _neutral(self, l: float) -> RGB:
        """Tinted gray: seed hue at very low saturation, given lightness."""
        h, _, _ = rgb_to_hsl(self.seed)
        return hsl_to_rgb(h, 0.05, l)

    def resolve(self) -> Theme:
        primary = ensure_on_contrast(self.seed)
        accents = [ensure_on_contrast(rotate_hue(primary, d)) for d in self._accent_hues()]
        accents[0] = primary

        if not self.dark:
            background = self._neutral(0.985)
            surface = self._neutral(0.955)
            text = self._neutral(0.14)
            muted = self._neutral(0.42)
            border = self._neutral(0.85)
        else:
            background = self._neutral(0.11)
            surface = self._neutral(0.16)
            text = self._neutral(0.95)
            muted = self._neutral(0.68)
            border = self._neutral(0.30)

        tokens = {
            "primary": rgb_to_hex(primary),
            "secondary": rgb_to_hex(accents[1]),
            "accent": rgb_to_hex(accents[3]),
            "background": rgb_to_hex(background),
            "surface": rgb_to_hex(surface),
            "text": rgb_to_hex(text),
            "muted": rgb_to_hex(muted),
            "border": rgb_to_hex(border),
        }
        for i, a in enumerate(accents, 1):
            tokens[f"accent{i}"] = rgb_to_hex(a)

        # Resolve every "on-" role by measured contrast against its surface.
        report: list[tuple[str, str, float]] = []

        def on(surface_key: str) -> str:
            bg = hex_to_rgb(tokens[surface_key])
            col, ratio = best_text_color(bg, [_WHITE, _NEAR_BLACK,
                                              hex_to_rgb(tokens["text"])])
            report.append((f"on-{surface_key}", tokens[surface_key], round(ratio, 2)))
            return rgb_to_hex(col)

        for key in ["primary", "secondary", "accent", "background", "surface"]:
            tokens[f"on_{key}"] = on(key)

        # Map semantic tokens -> the 12 OOXML theme slots.
        ppt_slots = {
            "dk1": tokens["text"],       # primary text/dark
            "lt1": tokens["background"], # primary background/light
            "dk2": tokens["primary"],    # secondary dark
            "lt2": tokens["surface"],    # secondary light
            "accent1": tokens["accent1"],
            "accent2": tokens["accent2"],
            "accent3": tokens["accent3"],
            "accent4": tokens["accent4"],
            "accent5": tokens["accent5"],
            "accent6": tokens["accent6"],
            "hlink": tokens["secondary"],
            "folHlink": tokens["accent"],
        }
        return Theme(tokens=tokens, ppt_slots=ppt_slots, contrast_report=report)