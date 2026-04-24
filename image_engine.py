"""
image_engine.py
===============
Provides two image sources for AgniAI's browser frontend:

  1. Web image search  — DuckDuckGo image search (no API key needed)
  2. Generated charts  — matplotlib charts built from hardcoded Agniveer data

Public API
----------
  get_images(query, answer_text) -> List[ImageResult]

Each ImageResult has:
  kind       : "web" | "chart"
  url        : absolute URL (web) or "/static/images/<file>" (chart)
  caption    : short human-readable label
  alt        : alt text for accessibility
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests

# ── Static assets dir (charts saved here) ─────────────────────────────────
STATIC_IMAGES_DIR = Path(__file__).parent / "static" / "images"
STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ── How many images to return per response ─────────────────────────────────
MAX_WEB_IMAGES   = 2
MAX_CHART_IMAGES = 1   # at most 1 generated chart per answer

# ── DuckDuckGo search timeout ──────────────────────────────────────────────
DDG_TIMEOUT = 8   # seconds


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class ImageResult:
    kind: str          # "web" or "chart"
    url: str           # full URL or /static/images/...
    caption: str
    alt: str
    width: int = 0
    height: int = 0


# =============================================================================
# TOPIC DETECTION
# =============================================================================

# Maps topic-slug → (web search query,  chart generator function name or None)
_TOPIC_MAP: List[tuple] = [
    # (keyword_patterns,  ddg_query,  chart_fn)
    (
        r"\b(salary|stipend|pay|package|income|wages?|compensation|seva nidhi|corpus)\b",
        "Agniveer salary package chart India",
        "chart_salary",
    ),
    (
        r"\b(selection|process|stages?|pipeline|steps?|filter|shortlist)\b",
        "Agniveer selection process flowchart India",
        "chart_selection_funnel",
    ),
    (
        r"\b(physical|fitness|pft|run|pull.?up|beam|ditch|zig.?zag|endurance)\b",
        "Agniveer physical fitness test India army",
        "chart_pft",
    ),
    (
        r"\b(eligibility|age|qualification|education|criteria|requirement)\b",
        "Agniveer eligibility age qualification India",
        None,
    ),
    (
        r"\b(timeline|schedule|calendar|date|notification|registration|exam)\b",
        "Agniveer recruitment timeline 2024 India",
        "chart_timeline",
    ),
    (
        r"\b(insurance|death|disability|compensation|cover|benefit)\b",
        "Agniveer insurance benefits India",
        None,
    ),
    (
        r"\b(training|regiment|centre|drill|induction)\b",
        "Agniveer military training India army",
        None,
    ),
    (
        r"\b(agnipath|scheme|policy|overview|what is|agni)\b",
        "Agnipath scheme India overview",
        None,
    ),
]


def _detect_topic(query: str, answer: str) -> Optional[tuple]:
    """Return first matching (ddg_query, chart_fn) or None."""
    combined = (query + " " + answer).lower()
    for pattern, ddg_query, chart_fn in _TOPIC_MAP:
        if re.search(pattern, combined, re.IGNORECASE):
            return ddg_query, chart_fn
    return None


# =============================================================================
# WEB IMAGE SEARCH  (DuckDuckGo — no API key)
# =============================================================================

def _ddg_image_search(query: str, max_results: int = 2) -> List[ImageResult]:
    """
    Fetch image results from DuckDuckGo's unofficial image endpoint.
    Returns up to max_results ImageResult objects of kind="web".
    Falls back to empty list on any error.
    """
    try:
        # Step 1: get vqd token
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0 Safari/537.36"
            )
        }
        r = requests.get(
            "https://duckduckgo.com/",
            params={"q": query},
            headers=headers,
            timeout=DDG_TIMEOUT,
        )
        vqd_match = re.search(r'vqd="([^"]+)"', r.text) or \
                    re.search(r"vqd=([\d-]+)[&']", r.text)
        if not vqd_match:
            return []
        vqd = vqd_match.group(1)

        # Step 2: query image endpoint
        time.sleep(0.3)
        resp = requests.get(
            "https://duckduckgo.com/i.js",
            params={"l": "in-en", "o": "json", "q": query, "vqd": vqd, "f": ",,,,,", "p": "1"},
            headers=headers,
            timeout=DDG_TIMEOUT,
        )
        data = resp.json()
        results: List[ImageResult] = []
        for item in data.get("results", [])[:max_results]:
            img_url = item.get("image", "")
            if not img_url or not img_url.startswith("http"):
                continue
            results.append(ImageResult(
                kind="web",
                url=img_url,
                caption=item.get("title", query)[:80],
                alt=f"Image related to: {query}",
                width=item.get("width", 0),
                height=item.get("height", 0),
            ))
        return results

    except Exception:
        return []


# =============================================================================
# CHART GENERATORS  (matplotlib — fully offline)
# =============================================================================

def _save_chart(fig, name: str) -> str:
    """Save a matplotlib figure and return the /static/images/... URL."""
    import matplotlib
    matplotlib.use("Agg")

    path = STATIC_IMAGES_DIR / f"{name}.png"
    fig.savefig(str(path), dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f"/static/images/{name}.png"


def chart_salary() -> Optional[ImageResult]:
    """Bar chart — Agniveer monthly package by year."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        years      = ["Year 1", "Year 2", "Year 3", "Year 4"]
        total      = [30_000, 33_000, 36_500, 40_000]
        in_hand    = [21_000, 23_100, 25_580, 28_000]
        corpus     = [9_000,  9_900, 10_950, 12_000]

        x = np.arange(len(years))
        w = 0.35

        fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="#0d1117")
        ax.set_facecolor("#161b22")

        bars1 = ax.bar(x - w/2, in_hand, w, label="In-Hand (70%)",
                       color="#238636", zorder=3)
        bars2 = ax.bar(x + w/2, corpus,  w, label="Seva Nidhi (30%)",
                       color="#1f6feb", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(years, color="#c9d1d9", fontsize=11)
        ax.set_ylabel("₹ / month", color="#8b949e", fontsize=10)
        ax.set_title("Agniveer Monthly Package Breakdown", color="#f0f6fc",
                     fontsize=13, fontweight="bold", pad=12)
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"₹{int(v):,}")
        )
        ax.set_ylim(0, 35_000)
        ax.grid(axis="y", color="#30363d", linewidth=0.7, zorder=0)
        ax.spines[:].set_color("#30363d")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 400,
                    f"₹{int(bar.get_height()):,}",
                    ha="center", va="bottom", color="#c9d1d9", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 400,
                    f"₹{int(bar.get_height()):,}",
                    ha="center", va="bottom", color="#c9d1d9", fontsize=8)

        ax.legend(facecolor="#21262d", edgecolor="#30363d",
                  labelcolor="#c9d1d9", fontsize=9)

        note = "Seva Nidhi exit corpus ≈ ₹10–12 lakh (tax-free) after 4 years"
        fig.text(0.5, -0.04, note, ha="center", fontsize=8.5,
                 color="#8b949e", style="italic")

        url = _save_chart(fig, "agniveer_salary")
        return ImageResult(kind="chart", url=url,
                           caption="Agniveer Monthly Package (Year-wise)",
                           alt="Bar chart showing Agniveer in-hand pay and Seva Nidhi corpus per year")
    except Exception:
        return None


def chart_selection_funnel() -> Optional[ImageResult]:
    """Horizontal funnel — Agniveer selection elimination stages."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as FancyArrowPatch

        stages = [
            ("Applicants",          "~10,00,000", "#e05252"),
            ("Clear Written (CEE)", "~3–4 Lakh",  "#e07d52"),
            ("Clear Physical",      "~1.5–2 Lakh","#d4a017"),
            ("Clear Medical",       "~1–1.2 Lakh","#52a0e0"),
            ("Final Selected",      "~40–50K",    "#52c05a"),
        ]

        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(stages) + 0.5)

        ax.set_title("Agniveer Selection Funnel", color="#f0f6fc",
                     fontsize=13, fontweight="bold", pad=10)

        for i, (label, count, color) in enumerate(stages):
            y   = len(stages) - i - 0.5
            pad = i * 0.55          # funnel narrows downward
            rect = plt.Rectangle((pad, y - 0.35), 10 - 2*pad, 0.7,
                                  color=color, alpha=0.82, zorder=3)
            ax.add_patch(rect)
            ax.text(5, y, f"{label}  —  {count}",
                    ha="center", va="center", fontsize=10,
                    color="white", fontweight="bold", zorder=4)

        fig.text(0.5, 0.01,
                 "Overall ratio: ~1 selected per 20–30 applicants",
                 ha="center", fontsize=8.5, color="#8b949e", style="italic")

        url = _save_chart(fig, "agniveer_selection_funnel")
        return ImageResult(kind="chart", url=url,
                           caption="Agniveer Selection Funnel (Stage-wise Elimination)",
                           alt="Funnel diagram showing candidate elimination at each stage of Agniveer recruitment")
    except Exception:
        return None


def chart_pft() -> Optional[ImageResult]:
    """Table chart — Physical Fitness Test groups and marks."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        col_labels = ["Group", "1.6 km Run Time", "Run Marks", "Pull-Ups", "Pull-Up Marks"]
        rows = [
            ["Group I",   "≤ 5:30 min",        "60", "10", "40"],
            ["Group II",  "5:31 – 5:45 min",   "48", "9",  "33"],
            ["Group III", "5:46 – 6:00 min",   "36", "8",  "27"],
            ["Group IV",  "6:01 – 6:15 min",   "24", "7",  "21"],
        ]

        fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        ax.axis("off")
        ax.set_title("Physical Fitness Test (PFT) — Marks Breakdown",
                     color="#f0f6fc", fontsize=12, fontweight="bold", pad=10)

        colors_row = [
            ["#1a3a1a"] * 5,
            ["#1a2d3a"] * 5,
            ["#2a2a1a"] * 5,
            ["#2a1a1a"] * 5,
        ]

        tbl = ax.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            cellColours=colors_row,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1, 2)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#30363d")
            if r == 0:
                cell.set_facecolor("#21262d")
                cell.set_text_props(color="#f0f6fc", fontweight="bold")
            else:
                cell.set_text_props(color="#c9d1d9")

        url = _save_chart(fig, "agniveer_pft")
        return ImageResult(kind="chart", url=url,
                           caption="Physical Fitness Test (PFT) Groups & Marks",
                           alt="Table showing Agniveer physical fitness test groups, run times and marks")
    except Exception:
        return None


def chart_timeline() -> Optional[ImageResult]:
    """Horizontal timeline — Agniveer recruitment schedule."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        events = [
            ("Notification",      "Feb",      0),
            ("Registration",      "Feb–Mar",  1),
            ("Online CEE",        "May–Jun",  2),
            ("CEE Results",       "Jun–Jul",  3),
            ("Rally & Medical",   "Aug–Oct",  4),
            ("Rally Results",     "Nov",      5),
            ("Doc Verification",  "Dec",      6),
            ("Training Starts",   "01 Jan",   7),
        ]

        fig, ax = plt.subplots(figsize=(9, 2.8), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        ax.axis("off")
        ax.set_title("Agniveer Recruitment Timeline (Phase-I)",
                     color="#f0f6fc", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(-0.5, len(events) - 0.5)
        ax.set_ylim(-0.8, 1.2)

        # Line
        ax.plot([0, len(events)-1], [0, 0], color="#30363d", lw=2, zorder=1)

        palette = ["#1f6feb", "#238636", "#e05252", "#d4a017",
                   "#c05acb", "#52a0e0", "#e07d52", "#52c05a"]

        for label, month, x in events:
            color = palette[x % len(palette)]
            ax.scatter(x, 0, s=120, color=color, zorder=3)
            # Alternate above/below
            y_text = 0.45 if x % 2 == 0 else -0.45
            ax.text(x, y_text, label, ha="center", va="center",
                    fontsize=7.5, color="#c9d1d9", fontweight="bold")
            ax.text(x, y_text + (0.28 if x % 2 == 0 else -0.28),
                    month, ha="center", va="center",
                    fontsize=7, color="#8b949e")
            # Connector
            ax.plot([x, x], [0, y_text * 0.7], color=color,
                    lw=1.2, alpha=0.6, zorder=2)

        url = _save_chart(fig, "agniveer_timeline")
        return ImageResult(kind="chart", url=url,
                           caption="Agniveer Recruitment Timeline (Phase-I)",
                           alt="Horizontal timeline of Agniveer recruitment milestones")
    except Exception:
        return None


# Chart function registry
_CHART_FN_MAP = {
    "chart_salary":           chart_salary,
    "chart_selection_funnel": chart_selection_funnel,
    "chart_pft":              chart_pft,
    "chart_timeline":         chart_timeline,
}


# =============================================================================
# PUBLIC API
# =============================================================================

def get_images(query: str, answer_text: str = "") -> List[ImageResult]:
    """
    Detect topic from query + answer, then:
      1. Fetch up to MAX_WEB_IMAGES from DuckDuckGo
      2. Generate up to MAX_CHART_IMAGES chart (if a generator exists for topic)

    Returns a combined list (charts first, web images after).
    Always returns [] on any failure — never raises.
    """
    topic = _detect_topic(query, answer_text)
    if not topic:
        return []

    ddg_query, chart_fn_name = topic
    results: List[ImageResult] = []

    # ── Generated chart (offline, always available) ────────────────────────
    if chart_fn_name and chart_fn_name in _CHART_FN_MAP:
        try:
            chart = _CHART_FN_MAP[chart_fn_name]()
            if chart:
                results.append(chart)
        except Exception:
            pass

    # ── Web images (requires internet) ────────────────────────────────────
    web_imgs = _ddg_image_search(ddg_query, max_results=MAX_WEB_IMAGES)
    results.extend(web_imgs)

    return results
