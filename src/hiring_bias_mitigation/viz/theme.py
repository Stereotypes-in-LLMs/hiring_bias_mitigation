"""One Altair theme for every figure in the paper.

Centralised so a reviewer never sees two figures with different type sizes, and so the
Ukrainian set is identical to the English one in everything but its text.

Font choice is load-bearing here. vl-convert renders text with the fonts it can find, and a
family without Cyrillic silently drops to tofu boxes in the Ukrainian figures -- which looks
like a rendering bug in the PDF rather than a missing font. `FONT_STACK` therefore lists
Cyrillic-complete families first, and `check_font` reports what is actually available before
a figure run commits to it.
"""

from __future__ import annotations

#: Families that carry a complete Cyrillic set, most preferred first.
FONT_STACK = "DejaVu Sans, Noto Sans, Liberation Sans, Arial, sans-serif"

#: Okabe–Ito: distinguishable under the common forms of colour blindness and in greyscale
#: print, which is where a fairness figure usually ends up.
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky
    "#F0E442",  # yellow
    "#000000",  # black
]

#: Diverging scale for "did this get better or worse", anchored at zero.
DIVERGING = ["#D55E00", "#F5F5F5", "#0072B2"]

BASE_FONT_SIZE = 11


def paper_theme() -> dict:
    """Altair theme config: print-sized type, no chartjunk, horizontal gridlines only."""
    return {
        "config": {
            "font": FONT_STACK,
            "background": "white",
            "view": {"stroke": "transparent", "continuousWidth": 320, "continuousHeight": 220},
            "axis": {
                "labelFont": FONT_STACK,
                "titleFont": FONT_STACK,
                "labelFontSize": BASE_FONT_SIZE,
                "titleFontSize": BASE_FONT_SIZE + 1,
                "titleFontWeight": "normal",
                "labelColor": "#222222",
                "titleColor": "#222222",
                "domainColor": "#888888",
                "tickColor": "#888888",
                "grid": False,
            },
            # Vertical position is what a reader compares in these figures; a horizontal rule
            # helps that and a vertical one only adds ink.
            "axisY": {"grid": True, "gridColor": "#E8E8E8", "gridDash": [1, 2], "domain": False},
            "legend": {
                "labelFont": FONT_STACK,
                "titleFont": FONT_STACK,
                "labelFontSize": BASE_FONT_SIZE,
                "titleFontSize": BASE_FONT_SIZE,
                "titleFontWeight": "normal",
                "orient": "top",
                "direction": "horizontal",
                "symbolType": "circle",
            },
            "title": {
                "font": FONT_STACK,
                "fontSize": BASE_FONT_SIZE + 2,
                "fontWeight": "bold",
                "anchor": "start",
                "color": "#111111",
                "subtitleFont": FONT_STACK,
                "subtitleFontSize": BASE_FONT_SIZE,
                "subtitleColor": "#555555",
            },
            "header": {
                "labelFont": FONT_STACK,
                "titleFont": FONT_STACK,
                "labelFontSize": BASE_FONT_SIZE,
                "titleFontSize": BASE_FONT_SIZE,
                "titleFontWeight": "normal",
            },
            "range": {"category": PALETTE, "diverging": DIVERGING},
            "point": {"filled": True, "size": 70},
            "bar": {"discreteBandSize": {"band": 0.7}},
            "rule": {"color": "#777777"},
        }
    }


THEME_NAME = "hbm_paper"


def register(enable: bool = True) -> None:
    """Registers the theme with Altair, across both the 5.0 and 5.5 theme APIs."""
    import altair as alt

    try:  # Altair >= 5.5
        alt.theme.register(THEME_NAME, enable=enable)(paper_theme)
    except AttributeError:  # Altair 5.0-5.4
        alt.themes.register(THEME_NAME, paper_theme)
        if enable:
            alt.themes.enable(THEME_NAME)


def check_font() -> tuple[str, list[str]]:
    """The first Cyrillic-capable family in `FONT_STACK` that the system actually has.

    Asks fontconfig, not vl-convert: vl-convert exposes no font-listing API (only
    `register_font_directory`), and an earlier version of this function called one that does
    not exist, caught the AttributeError and reported "no fonts" on a machine with 27 Cyrillic
    families installed. A font check that fails open is worse than none -- it trains you to
    ignore the warning.

    Returns ("", []) when fontconfig is unavailable, which is a genuine "cannot tell" rather
    than a claim that nothing is installed.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["fc-list", ":lang=uk", "family"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "", []

    available = set()
    for line in out.splitlines():
        # fc-list prints comma-separated aliases per family.
        available.update(part.strip() for part in line.split(",") if part.strip())

    for family in (f.strip() for f in FONT_STACK.split(",")):
        if family in available:
            return family, sorted(available)
    return "", sorted(available)


def register_font_dir(path: str) -> None:
    """Points vl-convert at a directory of fonts it would not otherwise find.

    Needed on machines where the fonts are not installed system-wide -- a container, or a
    cluster node with a local font bundle.
    """
    import vl_convert as vlc

    vlc.register_font_directory(path)
