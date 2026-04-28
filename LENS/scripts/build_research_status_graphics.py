#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "docs" / "research-status" / "assets"

CARD_BORDER = "#d9dfeb"
TEXT = "#0f172a"
MUTED = "#334155"
SOFT = "#64748b"
BG = "#f4f7fb"
WHITE = "#ffffff"
ALLY = "#0d9488"
SCOUT = "#f97316"
DATUM = "#4f46e5"
ACCENT = "#0f172a"
ALLY_SOFT = "#99f6e4"
SCOUT_SOFT = "#fdba74"
DATUM_SOFT = "#c7d2fe"
GREEN = "#22c55e"
RED = "#ef4444"
GOLD = "#f59e0b"
BLUE = "#3b82f6"


AGENTS = [
    {
        "name": "Ally",
        "slug": "ally",
        "color": ALLY,
        "summary": ["Dense completion corridor", "Hesitation happens inside the same safe zone"],
    },
    {
        "name": "Scout",
        "slug": "scout",
        "color": SCOUT,
        "summary": ["Sharp refusal corner", "Best current example of a clear reject boundary"],
    },
    {
        "name": "Datum",
        "slug": "datum",
        "color": DATUM,
        "summary": ["Sparse single basin", "Useful baseline for a low-density field"],
    },
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            ]
        )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_H1 = font(54, bold=True)
FONT_H2 = font(30, bold=True)
FONT_H3 = font(24, bold=True)
FONT_BODY = font(21)
FONT_SMALL = font(17)
FONT_TINY = font(15)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, fill: str, outline: str | None = None, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], max_width: int, font_obj, fill: str, line_gap: int = 8) -> int:
    x, y = xy
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = word if not current else f"{current} {word}"
        if draw.textlength(test, font=font_obj) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    bbox = draw.textbbox((0, 0), "Ag", font=font_obj)
    line_height = bbox[3] - bbox[1]
    for line in lines:
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += line_height + line_gap
    return y


def pill(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fill: str, text_fill: str = WHITE) -> int:
    width = int(draw.textlength(text, font=FONT_SMALL)) + 32
    rounded(draw, (x, y, x + width, y + 38), 19, fill)
    draw.text((x + 16, y + 9), text, font=FONT_SMALL, fill=text_fill)
    return width


def glow_circle(base: Image.Image, cx: int, cy: int, r: int, color: str, blur: int = 24, alpha: int = 190) -> None:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*rgb, alpha))
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    base.alpha_composite(layer)


def glow_polyline(base: Image.Image, points: list[tuple[int, int]], color: str, width: int = 18, blur: int = 18, alpha: int = 170) -> None:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
    draw.line(points, fill=(*rgb, alpha), width=width, joint="curve")
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    base.alpha_composite(layer)


def field_card(canvas: Image.Image, box: tuple[int, int, int, int], agent: dict) -> None:
    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = box
    rounded(draw, box, 28, WHITE, CARD_BORDER, 2)
    draw.rounded_rectangle((x1, y1, x2, y1 + 10), radius=28, fill=agent["color"])
    draw.text((x1 + 26, y1 + 30), agent["name"], font=FONT_H2, fill=TEXT)

    plot = (x1 + 24, y1 + 92, x2 - 24, y1 + 420)
    rounded(draw, plot, 22, "#fbfdff", "#e2e8f0", 2)

    for i in range(1, 4):
        px = plot[0] + i * (plot[2] - plot[0]) // 4
        py = plot[1] + i * (plot[3] - plot[1]) // 4
        draw.line((px, plot[1] + 18, px, plot[3] - 18), fill="#e5e7eb", width=2)
        draw.line((plot[0] + 18, py, plot[2] - 18, py), fill="#e5e7eb", width=2)

    draw.text((plot[0] + 10, plot[1] - 24), "Lower risk", font=FONT_TINY, fill=SOFT)
    draw.text((plot[2] - 88, plot[3] + 8), "More coherent", font=FONT_TINY, fill=SOFT)

    if agent["name"] == "Ally":
        glow_polyline(canvas, [(plot[0] + 70, plot[3] - 70), (plot[0] + 180, plot[1] + 190), (plot[2] - 110, plot[1] + 80)], ALLY, width=20, blur=22, alpha=170)
        glow_polyline(canvas, [(plot[0] + 90, plot[3] - 88), (plot[0] + 210, plot[1] + 200), (plot[2] - 90, plot[1] + 72)], ALLY_SOFT, width=30, blur=28, alpha=130)
        glow_circle(canvas, plot[2] - 105, plot[1] + 88, 52, GREEN, blur=26, alpha=200)
        glow_circle(canvas, plot[0] + 200, plot[1] + 196, 42, GOLD, blur=24, alpha=180)
        draw.text((plot[2] - 138, plot[1] + 146), "completion basin", font=FONT_TINY, fill=TEXT)
        draw.text((plot[0] + 124, plot[1] + 235), "hesitation zone", font=FONT_TINY, fill=TEXT)
    elif agent["name"] == "Scout":
        glow_polyline(canvas, [(plot[0] + 130, plot[3] - 90), (plot[0] + 210, plot[1] + 170), (plot[2] - 140, plot[1] + 92)], SCOUT, width=16, blur=18, alpha=160)
        glow_circle(canvas, plot[2] - 132, plot[1] + 92, 44, GREEN, blur=24, alpha=190)
        glow_circle(canvas, plot[0] + 70, plot[3] - 54, 52, RED, blur=24, alpha=205)
        draw.line((plot[0] + 110, plot[3] - 90, plot[0] + 82, plot[3] - 64), fill=RED, width=6)
        draw.text((plot[0] + 24, plot[3] - 118), "refusal corner", font=FONT_TINY, fill=TEXT)
        draw.text((plot[2] - 130, plot[1] + 148), "completion", font=FONT_TINY, fill=TEXT)
    else:
        glow_polyline(canvas, [(plot[0] + 70, plot[3] - 92), (plot[2] - 120, plot[1] + 120)], DATUM_SOFT, width=16, blur=18, alpha=130)
        glow_circle(canvas, plot[2] - 120, plot[1] + 118, 58, DATUM, blur=28, alpha=200)
        draw.text((plot[2] - 158, plot[1] + 182), "small stable basin", font=FONT_TINY, fill=TEXT)

    y = y1 + 458
    for item in agent["summary"]:
        draw.ellipse((x1 + 28, y + 8, x1 + 40, y + 20), fill=agent["color"])
        y = draw_wrapped(draw, item, (x1 + 52, y), x2 - x1 - 80, FONT_BODY, TEXT, line_gap=6) + 8

    footer = (x1 + 24, y2 - 108, x2 - 24, y2 - 24)
    rounded(draw, footer, 18, "#f8fafc", "#e2e8f0", 1)
    if agent["name"] == "Ally":
        title = "What people can see"
        body = "A strong safe corridor. The agent pauses inside it, then usually finishes."
    elif agent["name"] == "Scout":
        title = "What people can see"
        body = "A visible red corner where refusals collect instead of blending in."
    else:
        title = "What people can see"
        body = "A light-traffic field with one main destination and very little spread."
    draw.text((footer[0] + 16, footer[1] + 14), title, font=FONT_H3, fill=ACCENT)
    draw_wrapped(draw, body, (footer[0] + 16, footer[1] + 48), footer[2] - footer[0] - 32, FONT_SMALL, MUTED, line_gap=5)


def build_comparison() -> None:
    canvas = Image.new("RGBA", (1800, 1260), BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((90, 70), "What trace collection reveals already", font=FONT_H1, fill=TEXT)
    draw_wrapped(
        draw,
        "These are simplified public illustrations of the current trace corpus. "
        "The point is that crowd-sourced traces make stable shapes visible: safe corridors, hesitation zones, and refusal boundaries.",
        (90, 148),
        1620,
        FONT_BODY,
        MUTED,
        line_gap=7,
    )

    card_w = 520
    card_h = 860
    top = 280
    lefts = [90, 640, 1190]

    for idx, agent in enumerate(AGENTS):
        x = lefts[idx]
        field_card(canvas, (x, top, x + card_w, top + card_h), agent)

    draw_wrapped(
        draw,
        "Current evidence supports a public story about observability: collect enough traces and you stop talking only about principles. "
        "You can point to where agents complete, where they hesitate, and where they refuse.",
        (90, 1175),
        1620,
        FONT_SMALL,
        SOFT,
        line_gap=4,
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(OUTDIR / "trace-attractor-comparison.png", quality=95)


def arrow(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, color: str) -> None:
    draw.line((x1, y1, x2, y2), fill=color, width=8)
    size = 16
    draw.polygon([(x2, y2), (x2 - size, y2 - size // 2), (x2 - size, y2 + size // 2)], fill=color)


def flow_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: str, accent: str) -> None:
    rounded(draw, box, 24, WHITE, CARD_BORDER, 2)
    rounded(draw, (box[0] + 16, box[1] + 16, box[0] + 140, box[1] + 54), 14, accent)
    draw.text((box[0] + 30, box[1] + 24), title, font=FONT_SMALL, fill=WHITE)
    draw_wrapped(draw, body, (box[0] + 24, box[1] + 84), box[2] - box[0] - 48, FONT_BODY, TEXT, line_gap=6)


def build_flywheel() -> None:
    canvas = Image.new("RGB", (1800, 1080), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((90, 72), "How the free app turns traces into alignment research", font=FONT_H1, fill=TEXT)
    draw_wrapped(
        draw,
        "The public case is simple: open-source agents and a free app can crowd-source the dataset that alignment research is missing. "
        "Each trace is a small behavioral measurement. At scale, those measurements become interpretable maps.",
        (90, 150),
        1600,
        FONT_BODY,
        MUTED,
        line_gap=7,
    )

    boxes = [
        ((90, 300, 410, 590), "1. Capture", "Users run CIRIS in the free app or in open-source agent deployments. The runtime records behavior through real tasks.", ALLY),
        ((500, 300, 820, 590), "2. Contribute", "With explicit permission, selected traces can join a shared research corpus. The power comes from many real traces, not a staged demo.", BLUE),
        ((910, 300, 1230, 590), "3. Aggregate", "Across agents and tasks, stable shapes appear. You can see completion corridors, hesitation zones, and refusal boundaries.", SCOUT),
        ((1320, 300, 1640, 590), "4. Improve", "Those maps improve operator tools now, and richer schema upgrades push the work toward proper collapse and recovery analysis.", DATUM),
    ]
    for box, title, body, accent in boxes:
        flow_box(draw, box, title, body, accent)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0][2] + 18
        y1 = (boxes[i][0][1] + boxes[i][0][3]) // 2
        x2 = boxes[i + 1][0][0] - 18
        y2 = (boxes[i + 1][0][1] + boxes[i + 1][0][3]) // 2
        arrow(draw, x1, y1, x2, y2, "#94a3b8")

    rounded(draw, (90, 690, 855, 955), 28, WHITE, "#bfdbfe", 2)
    draw.text((120, 730), "What the current data already supports", font=FONT_H2, fill=TEXT)
    current = [
        "Behavioral attractors are visible in aggregate traces.",
        "Different agents occupy different regions and boundary conditions.",
        "Deferral and refusal can be mapped as operational states, not just anecdotes.",
        "Operators can inspect trajectory shape instead of relying on marketing claims.",
    ]
    y = 790
    for line in current:
        draw.ellipse((122, y + 8, 132, y + 18), fill="#2563eb")
        y = draw_wrapped(draw, line, (146, y), 660, FONT_BODY, TEXT, line_gap=6) + 10

    rounded(draw, (945, 690, 1710, 955), 28, WHITE, "#ddd6fe", 2)
    draw.text((975, 730), "What the next schema upgrade unlocks", font=FONT_H2, fill=TEXT)
    next_up = [
        "Real CCA needs raw source counts, source provenance, and correlation structure.",
        "IDMA should emit raw k, raw rho, module structure, and intervention markers.",
        "That upgrade turns behavioral overlays into actual collapse and recovery analysis.",
        "The page should say the dataset is growing toward that standard, not pretend it is already there.",
    ]
    y = 790
    for line in next_up:
        draw.ellipse((977, y + 8, 987, y + 18), fill="#7c3aed")
        y = draw_wrapped(draw, line, (1001, y), 660, FONT_BODY, TEXT, line_gap=6) + 10

    canvas.save(OUTDIR / "crowdsourced-alignment-loop.png", quality=95)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    build_comparison()
    build_flywheel()


if __name__ == "__main__":
    main()
