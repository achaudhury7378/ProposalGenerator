"""
typography.py — deterministic type system.

Two decisions, cleanly separated:
  1. The SCALE is arithmetic: size(n) = base * ratio**n. Not a taste question.
  2. The PAIRING (which two families) IS taste, so it's a curated lookup table
     of pairings that are known-safe and embeddable (Google Fonts / OFL).

Font pairing rule of thumb encoded here: pair by CLASSIFICATION CONTRAST
(serif heading + sans body, or a shared-skeleton superfamily). Two families
max, ~2 weights each.
"""
from __future__ import annotations
from dataclasses import dataclass

# Curated, embeddable pairings. (heading_family, body_family, note)
# All are OFL/Apache — safe to bundle .ttf and embed in the .pptx.
PAIRINGS: dict[str, tuple[str, str, str]] = {
    "editorial":   ("Playfair Display", "Inter",            "high-contrast serif display + neutral sans"),
    "corporate":   ("Inter",            "Inter",            "single neutral sans, weight-based hierarchy"),
    "warm":        ("Lora",             "Raleway",          "friendly serif + geometric sans"),
    "technical":   ("IBM Plex Sans",    "IBM Plex Sans",    "superfamily, cannot clash"),
    "classic":     ("Merriweather",     "Source Sans 3",    "readable serif + humanist sans"),
}

# Named ratios for the modular scale.
RATIOS = {
    "minor_third": 1.20,
    "major_third": 1.25,
    "perfect_fourth": 1.333,
    "perfect_fifth": 1.5,
    "golden": 1.618,
}

# role -> (scale step n, weight, is_bold). Step feeds size = base * ratio**n.
_ROLE_STEPS = {
    "display": (4, "Bold", True),
    "h1":      (3, "Bold", True),
    "h2":      (2, "SemiBold", True),
    "h3":      (1, "SemiBold", True),
    "body":    (0, "Regular", False),
    "caption": (-1, "Regular", False),
}


@dataclass
class RoleStyle:
    family: str
    size_pt: int
    bold: bool
    line_height: float  # multiplier


@dataclass
class Typography:
    roles: dict[str, RoleStyle]
    heading_family: str
    body_family: str

    def style(self, role: str) -> RoleStyle:
        return self.roles[role]


class TypographyResolver:
    def __init__(self, pairing: str = "corporate", base_pt: int = 18,
                 ratio: str = "major_third"):
        self.heading, self.body, _ = PAIRINGS[pairing]
        self.base = base_pt
        self.ratio = RATIOS[ratio]

    def resolve(self) -> Typography:
        roles: dict[str, RoleStyle] = {}
        for role, (step, _weight, is_bold) in _ROLE_STEPS.items():
            size = round(self.base * (self.ratio ** step))
            # Big headings get tight leading; body/caption get airy leading.
            lh = 1.12 if size >= self.base * self.ratio else 1.5
            family = self.heading if role in ("display", "h1", "h2", "h3") else self.body
            roles[role] = RoleStyle(family=family, size_pt=size, bold=is_bold, line_height=lh)
        return Typography(roles=roles, heading_family=self.heading, body_family=self.body)