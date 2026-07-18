"""
ir.py — the Slide Intermediate Representation.

This is the CONTRACT between the LLM and the design engine. The LLM fills this
(via Ollama structured output / `instructor` / Outlines). It contains CONTENT
and coarse INTENT only — never colors, fonts, or coordinates. If a design word
shows up in this file, it's in the wrong layer.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

LayoutIntent = Literal[
    "title", "section", "bullets", "two_column",
    "stat", "quote", "image_text", "chart", "table", "comparison",
]


class Bullet(BaseModel):
    text: str
    level: int = 0


class SlideContent(BaseModel):
    layout_intent: LayoutIntent
    title: str | None = None
    subtitle: str | None = None
    bullets: list[Bullet] = Field(default_factory=list)
    columns: list[list[Bullet]] = Field(default_factory=list)
    stat_value: str | None = None      # e.g. "37%"
    stat_caption: str | None = None
    quote: str | None = None
    attribution: str | None = None
    image_query: str | None = None     # asset hint only, not a URL
    speaker_notes: str | None = None


class Deck(BaseModel):
    title: str
    subtitle: str | None = None
    seed_color: str | None = None      # brand hex; None -> engine default
    pairing: str = "corporate"
    slides: list[SlideContent] = Field(default_factory=list)