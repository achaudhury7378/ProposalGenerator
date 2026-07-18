"""
layout.py — the layout engine (the heuristic part; be honest about it).

What is solid here:
  - The 12-column grid math is exact geometry in EMU.
  - Intent -> template mapping is a deterministic dispatch.

What is APPROXIMATE here (flagged in code):
  - Overflow detection estimates text height from an average glyph advance
    (~0.5 * font size). Real fitting needs actual font metrics (fontTools /
    PIL.ImageFont.getlength). The estimator is deliberately conservative so it
    errs toward shrinking/splitting rather than clipping. Swap `_est_line_count`
    for a metrics-based version when you want pixel-accurate fitting.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math

from .ir import SlideContent
from .typography import Typography

EMU_PER_INCH = 914400

# 16:9 slide
SLIDE_W = round(13.333 * EMU_PER_INCH)
SLIDE_H = round(7.5 * EMU_PER_INCH)

MARGIN = round(0.6 * EMU_PER_INCH)
GUTTER = round(0.25 * EMU_PER_INCH)
COLS = 12

MIN_FONT_PT = 12


@dataclass
class Grid:
    slide_w: int = SLIDE_W
    slide_h: int = SLIDE_H
    margin: int = MARGIN
    gutter: int = GUTTER
    cols: int = COLS

    @property
    def usable_w(self) -> int:
        return self.slide_w - 2 * self.margin

    @property
    def col_w(self) -> int:
        return (self.usable_w - (self.cols - 1) * self.gutter) // self.cols

    def x(self, col: int) -> int:
        """Left EMU of a 0-indexed column start."""
        return self.margin + col * (self.col_w + self.gutter)

    def span_w(self, ncols: int) -> int:
        return ncols * self.col_w + (ncols - 1) * self.gutter


@dataclass
class Element:
    kind: str                      # "text" | "rect" | "picture" | "chart" | "table"
    left: int
    top: int
    width: int
    height: int
    role: str | None = None        # typography role: h1/body/...
    fill_token: str | None = None  # semantic color token for fill
    text_token: str | None = None  # semantic color token for text
    text: str | None = None
    align: str = "left"
    meta: dict = field(default_factory=dict)


@dataclass
class SlideSpec:
    layout_intent: str
    background_token: str
    elements: list[Element]
    notes: str | None = None
    overflow: bool = False         # True if content had to be shrunk/split


# ---------------------------------------------------------------------------
# Overflow estimation (APPROXIMATE — see module docstring)
# ---------------------------------------------------------------------------

def _est_line_count(text: str, font_pt: int, box_w_emu: int) -> int:
    if not text:
        return 0
    avg_char_in = 0.5 * font_pt / 72.0          # rough advance width
    box_w_in = box_w_emu / EMU_PER_INCH
    cpl = max(1, int(box_w_in / avg_char_in))
    # Each hard newline (bullet) is its own paragraph and wraps independently.
    return sum(max(1, math.ceil(len(line) / cpl)) for line in text.split("\n"))


def _fits(text: str, font_pt: int, box_w: int, box_h: int, lh: float) -> bool:
    lines = _est_line_count(text, font_pt, box_w)
    needed_in = lines * font_pt * lh / 72.0
    return needed_in <= box_h / EMU_PER_INCH


def _fit_font(text: str, role_pt: int, box_w: int, box_h: int, lh: float) -> tuple[int, bool]:
    """Return (font_pt, overflowed). Shrink toward MIN_FONT_PT; flag if still over."""
    pt = role_pt
    while pt > MIN_FONT_PT and not _fits(text, pt, box_w, box_h, lh):
        pt -= 1
    return pt, (not _fits(text, pt, box_w, box_h, lh))


# ---------------------------------------------------------------------------
# LayoutEngine
# ---------------------------------------------------------------------------

class LayoutEngine:
    def __init__(self, typography: Typography, grid: Grid | None = None):
        self.t = typography
        self.g = grid or Grid()

    def build(self, sc: SlideContent) -> SlideSpec:
        fn = getattr(self, f"_layout_{sc.layout_intent}", self._layout_bullets)
        spec = fn(sc)
        spec.notes = sc.speaker_notes
        return spec

    # -- individual templates ------------------------------------------------

    def _layout_title(self, sc: SlideContent) -> SlideSpec:
        g = self.g
        els = [
            Element("text", g.x(0), round(2.8 * EMU_PER_INCH), g.span_w(12),
                    round(1.6 * EMU_PER_INCH), role="display",
                    text_token="text", text=sc.title, align="left"),
        ]
        if sc.subtitle:
            els.append(Element("text", g.x(0), round(4.5 * EMU_PER_INCH), g.span_w(10),
                               round(0.8 * EMU_PER_INCH), role="h3",
                               text_token="muted", text=sc.subtitle))
        # A brand accent bar for visual anchor.
        els.append(Element("rect", g.x(0), round(2.55 * EMU_PER_INCH),
                           round(0.9 * EMU_PER_INCH), round(0.12 * EMU_PER_INCH),
                           fill_token="primary"))
        return SlideSpec("title", "background", els)

    def _layout_section(self, sc: SlideContent) -> SlideSpec:
        g = self.g
        els = [
            Element("rect", 0, 0, g.slide_w, g.slide_h, fill_token="primary"),
            Element("text", g.x(0), round(3.0 * EMU_PER_INCH), g.span_w(11),
                    round(1.6 * EMU_PER_INCH), role="h1",
                    text_token="on_primary", text=sc.title),
        ]
        return SlideSpec("section", "primary", els)

    def _layout_bullets(self, sc: SlideContent) -> SlideSpec:
        g = self.g
        els = [self._title_el(sc.title)]
        body_top = round(2.0 * EMU_PER_INCH)
        body_h = g.slide_h - body_top - g.margin
        box_w = g.span_w(11)
        text = "\n".join("  " * b.level + "• " + b.text for b in sc.bullets)
        role_pt = self.t.style("body").size_pt
        lh = self.t.style("body").line_height
        pt, over = _fit_font(text, role_pt, box_w, body_h, lh)
        els.append(Element("text", g.x(0), body_top, box_w, body_h,
                           role="body", text_token="text", text=text,
                           meta={"font_pt_override": pt}))
        return SlideSpec("bullets", "background", els, overflow=over)

    def _layout_two_column(self, sc: SlideContent) -> SlideSpec:
        g = self.g
        els = [self._title_el(sc.title)]
        cols = sc.columns or [sc.bullets[: len(sc.bullets) // 2],
                              sc.bullets[len(sc.bullets) // 2:]]
        top = round(2.0 * EMU_PER_INCH)
        h = g.slide_h - top - g.margin
        for i, col in enumerate(cols[:2]):
            text = "\n".join("• " + b.text for b in col)
            els.append(Element("text", g.x(i * 6), top, g.span_w(5), h,
                               role="body", text_token="text", text=text))
        return SlideSpec("two_column", "background", els)

    def _layout_stat(self, sc: SlideContent) -> SlideSpec:
        g = self.g
        els = [
            Element("text", g.x(0), round(2.4 * EMU_PER_INCH), g.span_w(12),
                    round(2.2 * EMU_PER_INCH), role="display",
                    text_token="primary", text=sc.stat_value or "", align="left"),
        ]
        if sc.stat_caption:
            els.append(Element("text", g.x(0), round(4.7 * EMU_PER_INCH), g.span_w(10),
                               round(1.0 * EMU_PER_INCH), role="h3",
                               text_token="muted", text=sc.stat_caption))
        return SlideSpec("stat", "background", els)

    def _layout_quote(self, sc: SlideContent) -> SlideSpec:
        g = self.g
        els = [
            Element("text", g.x(1), round(2.4 * EMU_PER_INCH), g.span_w(10),
                    round(2.4 * EMU_PER_INCH), role="h1",
                    text_token="text", text=f"\u201c{sc.quote}\u201d"),
        ]
        if sc.attribution:
            els.append(Element("text", g.x(1), round(5.0 * EMU_PER_INCH), g.span_w(8),
                               round(0.7 * EMU_PER_INCH), role="body",
                               text_token="muted", text=f"\u2014 {sc.attribution}"))
        return SlideSpec("quote", "surface", els)

    # Stubs that fall back to a sensible layout; extend these next.
    def _layout_image_text(self, sc: SlideContent) -> SlideSpec:
        return self._layout_two_column(sc)

    def _layout_comparison(self, sc: SlideContent) -> SlideSpec:
        return self._layout_two_column(sc)

    # -- helpers -------------------------------------------------------------

    def _title_el(self, title: str | None) -> Element:
        g = self.g
        return Element("text", g.x(0), g.margin, g.span_w(11),
                       round(1.1 * EMU_PER_INCH), role="h2",
                       text_token="text", text=title or "")