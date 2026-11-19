"""STEP 4: Generate 200 document-like images with PIL into negatives/documents.

Categories (50 each):
  1. Rendered text pages (articles, lab reports, multi-paragraph documents)
  2. Data tables, spreadsheets, and invoices (grid cells, headers, numbers)
  3. Code-like IDE screenshots (dark/light themes, line gutters, syntax tokens)
  4. UI / dashboard screenshots (navbars, form inputs, metric cards, charts)

- Uses varied TrueType system fonts (Arial, Times, Consolas, Calibri, Courier)
- Varied backgrounds (white, off-white, gray, dark slate, dark IDE)
- ~35% converted to grayscale (saved as 3-channel RGB)
- Saved as JPEG with seeded random quality 60-95
- Safe to re-run (skips existing valid files)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent

SAMPLE_WORDS = [
    "patient", "clinical", "radiology", "report", "assessment", "protocol",
    "table", "summary", "invoice", "department", "analysis", "system",
    "configuration", "parameter", "function", "return", "import", "dataset",
    "validation", "threshold", "confidence", "interval", "schedule", "status",
    "approved", "pending", "reference", "document", "section", "page",
]

CODE_LINES = [
    "import os, sys, json",
    "from pathlib import Path",
    "from typing import Dict, List, Optional",
    "",
    "class DocumentProcessor:",
    "    def __init__(self, config_path: Path, strict: bool = True):",
    "        self.config_path = Path(config_path)",
    "        self.strict = strict",
    "        self.records: List[Dict[str, float]] = []",
    "",
    "    def compute_summary(self, values: List[float]) -> Dict[str, float]:",
    "        if not values:",
    "            raise ValueError('Empty value list provided')",
    "        mean_val = sum(values) / len(values)",
    "        return {'count': len(values), 'mean': round(mean_val, 4)}",
    "",
    "def main() -> int:",
    "    proc = DocumentProcessor(Path('configs/settings.yaml'))",
    "    res = proc.compute_summary([12.4, 15.8, 9.2, 18.1, 14.0])",
    "    print(json.dumps(res, indent=2))",
    "    return 0",
]


def load_fonts() -> List[Tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ImageFont.FreeTypeFont | ImageFont.ImageFont]]:
    font_names = ["arial.ttf", "times.ttf", "consola.ttf", "calibri.ttf", "cour.ttf", "georgia.ttf"]
    win_fonts = Path("C:/Windows/Fonts")
    loaded = []
    for name in font_names:
        fp = win_fonts / name
        if fp.exists():
            try:
                f_body = ImageFont.truetype(str(fp), size=14)
                f_head = ImageFont.truetype(str(fp), size=20)
                loaded.append((f_body, f_head))
            except Exception:
                pass
    if not loaded:
        default_f = ImageFont.load_default()
        loaded.append((default_f, default_f))
    return loaded


def is_valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def render_text_page(rng: random.Random, fonts) -> Image.Image:
    w, h = rng.choice([(600, 800), (640, 840), (700, 900), (640, 640)])
    bg = rng.choice([(255, 255, 255), (250, 248, 242), (242, 244, 247), (235, 235, 235), (32, 34, 38)])
    dark_mode = bg[0] < 80
    fg = (230, 230, 230) if dark_mode else (25, 25, 30)
    accent = (100, 170, 240) if dark_mode else (30, 70, 140)

    img = Image.new("RGB", (w, h), color=bg)
    draw = ImageDraw.Draw(img)
    f_body, f_head = rng.choice(fonts)

    # Header bar & title
    margin = rng.randint(36, 56)
    title = " ".join(rng.sample(SAMPLE_WORDS, 4)).upper()
    draw.text((margin, 32), title, fill=accent, font=f_head)
    draw.line((margin, 62, w - margin, 62), fill=accent, width=2)

    y = 82
    while y < h - 60:
        # Paragraph of 4-8 lines
        n_lines = rng.randint(4, 8)
        for _ in range(n_lines):
            if y >= h - 60:
                break
            words = [rng.choice(SAMPLE_WORDS) for _ in range(rng.randint(7, 12))]
            line = " ".join(words).capitalize() + "."
            draw.text((margin, y), line, fill=fg, font=f_body)
            y += rng.randint(18, 22)
        y += rng.randint(12, 24)

    # Footer
    draw.line((margin, h - 42, w - margin, h - 42), fill=(160, 160, 160), width=1)
    draw.text((margin, h - 34), f"Document ID: DOC-{rng.randint(10000, 99999)}   |   Page {rng.randint(1, 24)}", fill=fg, font=f_body)
    return img


def render_table_page(rng: random.Random, fonts) -> Image.Image:
    w, h = rng.choice([(760, 600), (680, 780), (720, 720)])
    bg = rng.choice([(255, 255, 255), (247, 249, 252), (28, 30, 36)])
    dark = bg[0] < 80
    fg = (225, 225, 230) if dark else (30, 35, 45)
    grid_col = (75, 80, 95) if dark else (180, 188, 200)
    hdr_bg = (50, 70, 110) if dark else (215, 226, 242)
    alt_bg = (38, 42, 50) if dark else (240, 244, 249)

    img = Image.new("RGB", (w, h), color=bg)
    draw = ImageDraw.Draw(img)
    f_body, f_head = rng.choice(fonts)

    draw.text((36, 24), f"TABULAR SUMMARY REPORT #{rng.randint(100, 999)}", fill=fg, font=f_head)

    top = 68
    left = 36
    right = w - 36
    n_cols = rng.randint(4, 6)
    n_rows = rng.randint(12, 18)
    col_w = (right - left) // n_cols
    row_h = min(32, (h - top - 50) // (n_rows + 1))

    # Header row
    draw.rectangle((left, top, left + col_w * n_cols, top + row_h), fill=hdr_bg, outline=grid_col)
    for c in range(n_cols):
        cx = left + c * col_w
        draw.line((cx, top, cx, top + row_h * (n_rows + 1)), fill=grid_col, width=1)
        draw.text((cx + 8, top + 6), f"COL_{c+1}", fill=fg, font=f_body)

    # Data rows
    for r in range(1, n_rows + 1):
        ry = top + r * row_h
        if r % 2 == 0:
            draw.rectangle((left, ry, left + col_w * n_cols, ry + row_h), fill=alt_bg)
        draw.line((left, ry, left + col_w * n_cols, ry), fill=grid_col, width=1)
        for c in range(n_cols):
            cx = left + c * col_w
            cell_txt = f"{rng.uniform(1.0, 999.9):.2f}" if c > 0 else f"ID-{1000 + r}"
            draw.text((cx + 8, ry + 6), cell_txt, fill=fg, font=f_body)

    draw.rectangle((left, top, left + col_w * n_cols, top + row_h * (n_rows + 1)), outline=grid_col, width=2)
    return img


def render_code_screenshot(rng: random.Random, fonts) -> Image.Image:
    w, h = rng.choice([(800, 600), (740, 560), (680, 680)])
    bg = rng.choice([(30, 30, 30), (22, 27, 34), (40, 44, 52), (248, 248, 248)])
    dark = bg[0] < 100
    sidebar_bg = (37, 37, 38) if dark else (232, 234, 237)
    gutter_fg = (110, 118, 129) if dark else (140, 140, 140)
    kw_col = (86, 156, 214) if dark else (0, 51, 179)
    str_col = (206, 145, 120) if dark else (6, 125, 23)
    txt_col = (212, 212, 212) if dark else (30, 30, 30)

    img = Image.new("RGB", (w, h), color=bg)
    draw = ImageDraw.Draw(img)
    f_body, _ = rng.choice(fonts)

    # Sidebar & top tab bar
    sidebar_w = rng.randint(130, 170)
    draw.rectangle((0, 0, sidebar_w, h), fill=sidebar_bg)
    draw.rectangle((sidebar_w, 0, w, 32), fill=sidebar_bg)

    for i in range(10):
        draw.text((14, 42 + i * 24), f"module_{i+1}.py", fill=gutter_fg, font=f_body)

    y = 48
    lines = list(CODE_LINES)
    rng.shuffle(lines)
    for line_no, code in enumerate(CODE_LINES + lines[:8], 1):
        if y >= h - 24:
            break
        draw.text((sidebar_w + 12, y), f"{line_no:2d}", fill=gutter_fg, font=f_body)
        color = kw_col if code.strip().startswith(("def ", "class ", "import ", "from ", "return")) else (
            str_col if "'" in code else txt_col
        )
        draw.text((sidebar_w + 48, y), code, fill=color, font=f_body)
        y += 20

    return img


def render_ui_screenshot(rng: random.Random, fonts) -> Image.Image:
    w, h = rng.choice([(800, 600), (760, 640), (720, 540)])
    bg = rng.choice([(241, 245, 249), (250, 250, 250), (18, 22, 29)])
    dark = bg[0] < 80
    card_bg = (30, 36, 46) if dark else (255, 255, 255)
    border = (60, 70, 85) if dark else (203, 213, 225)
    fg = (235, 240, 245) if dark else (30, 41, 59)
    nav_bg = (15, 23, 42) if not dark else (28, 38, 56)

    img = Image.new("RGB", (w, h), color=bg)
    draw = ImageDraw.Draw(img)
    f_body, f_head = rng.choice(fonts)

    # Top navbar
    draw.rectangle((0, 0, w, 48), fill=nav_bg)
    draw.text((24, 14), "Application Portal Dashboard", fill=(255, 255, 255), font=f_head)

    # 3 metric cards
    card_w = (w - 80) // 3
    for i in range(3):
        cx = 24 + i * (card_w + 16)
        draw.rectangle((cx, 68, cx + card_w, 156), fill=card_bg, outline=border, width=1)
        draw.text((cx + 14, 80), f"METRIC {i+1}", fill=fg, font=f_body)
        draw.text((cx + 14, 110), f"{rng.randint(120, 9850)}", fill=fg, font=f_head)

    # Main chart / form panel
    draw.rectangle((24, 176, w - 24, h - 24), fill=card_bg, outline=border, width=1)
    draw.text((44, 194), "Activity Overview & Input Controls", fill=fg, font=f_head)

    # Bar chart inside panel
    n_bars = rng.randint(8, 14)
    bw = (w - 140) // n_bars
    base_y = h - 56
    for b in range(n_bars):
        bx = 52 + b * bw
        bh = rng.randint(40, max(60, h - 300))
        bar_col = rng.choice([(59, 130, 246), (16, 185, 129), (99, 102, 241), (120, 120, 120)])
        draw.rectangle((bx, base_y - bh, bx + bw - 8, base_y), fill=bar_col)
    draw.line((44, base_y, w - 44, base_y), fill=fg, width=2)
    return img


def main() -> None:
    cfg_path = ROOT_DIR / "configs" / "data.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    neg_dir = Path(cfg.get("validator_negatives_dir", "C:/bonetest/data/validator_data/negatives"))
    docs_dir = neg_dir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    fonts = load_fonts()
    rng = random.Random(4004)
    renderers = [render_text_page, render_table_page, render_code_screenshot, render_ui_screenshot]

    written = 0
    skipped = 0
    gray_count = 0
    target_total = 200

    for idx in range(1, target_total + 1):
        out_path = docs_dir / f"doc_{idx:04d}.jpg"
        make_gray = (idx % 3 == 0)
        quality = rng.randint(60, 95)

        if is_valid_image(out_path):
            skipped += 1
            if make_gray:
                gray_count += 1
            continue

        renderer = renderers[(idx - 1) % len(renderers)]
        img = renderer(rng, fonts)
        if make_gray:
            img = img.convert("L").convert("RGB")
            gray_count += 1
        else:
            img = img.convert("RGB")

        img.save(out_path, format="JPEG", quality=quality)
        written += 1

    total_files = len([p for p in docs_dir.iterdir() if p.is_file()])
    print("=" * 90)
    print("STEP 4: DOCUMENT & SCREENSHOT NEGATIVE GENERATOR")
    print("=" * 90)
    print(f"Output folder          : {docs_dir}")
    print(f"Newly written          : {written}, Skipped existing: {skipped}")
    print(f"Grayscale-RGB count    : {gray_count}/{target_total}")
    print(f"Total files in folder  : {total_files}")
    print("=" * 90)


if __name__ == "__main__":
    main()
