#!/usr/bin/env python
"""Render paper/paper.md to paper/paper.pdf with figures embedded.

markdown -> HTML -> PDF via xhtml2pdf (pisa). DejaVuSans is embedded so
Unicode (°, ×, ⁻¹⁶⁰, ≈, ≥, →) renders instead of turning into boxes.
"""
import os
import re
from pathlib import Path

import markdown
from matplotlib import get_data_path
from xhtml2pdf import pisa
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "paper.md"
PAPER_DIR = PAPER.parent
OUT = ROOT / "paper" / "paper.pdf"

FONT_DIR = Path(get_data_path()) / "fonts" / "ttf"
FONTS = {
    "normal": FONT_DIR / "DejaVuSans.ttf",
    "bold": FONT_DIR / "DejaVuSans-Bold.ttf",
    "italic": FONT_DIR / "DejaVuSans-Oblique.ttf",
    "bolditalic": FONT_DIR / "DejaVuSans-BoldOblique.ttf",
}

# Per-figure display width (pt). Wide panel gets full width; others narrower.
FIG_WIDTH = {
    "panels_2024.png": 500,
    "change_map.png": 300,
    "roc_cv.png": 320,
    "lst_by_class_year.png": 360,
    "day_vs_night_effect.png": 400,
    "core_vs_fringe_night.png": 340,
}
DEFAULT_W = 360


def as_uri(p: Path) -> str:
    return p.resolve().as_uri()


def register_fonts():
    pdfmetrics.registerFont(TTFont("DejaVu", str(FONTS["normal"])))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONTS["bold"])))
    pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(FONTS["italic"])))
    pdfmetrics.registerFont(TTFont("DejaVu-BoldOblique", str(FONTS["bolditalic"])))
    registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                       italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique")


CSS = """
@page { size: letter; margin: 2.0cm 2.2cm; }
body { font-family: "DejaVu"; font-size: 10.5pt; line-height: 1.4; color: #1a1a1a; }
h1 { font-size: 17pt; color: #111; }
h2 { font-size: 13pt; color: #222; margin-top: 14pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; color: #333; margin-top: 10pt; }
p { margin: 5pt 0; }
blockquote { background: #f4f6f8; border-left: 3px solid #4C78A8; margin: 8pt 0;
  padding: 6pt 10pt; font-size: 9.5pt; }
table { -pdf-keep-with-next: false; margin: 8pt 0; border: 0.5pt solid #999; }
th { background: #eee; border: 0.5pt solid #999; padding: 3pt 5pt; font-size: 9pt; font-weight: bold; }
td { border: 0.5pt solid #999; padding: 3pt 5pt; font-size: 9pt; }
code { font-family: "Courier"; font-size: 9pt; background: #f0f0f0; }
.figure { -pdf-keep-with-next: true; margin-top: 8pt; }
.caption { font-size: 8.5pt; color: #555; margin: 2pt 0 10pt 0; }
hr { border: 0; border-top: 0.5pt solid #ccc; }
"""


def build_html() -> str:
    md_text = PAPER.read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "sane_lists"])

    # Center images, size them, and turn the following <em> paragraph into a caption.
    def img_repl(m):
        src = m.group("src")
        name = src.split("/")[-1]
        w = FIG_WIDTH.get(name, DEFAULT_W)
        return (f'<div class="figure" style="text-align:center">'
                f'<img src="{src}" style="width:{w}pt"/></div>')

    # Unicode superscripts (⁻¹⁶⁰ etc.) -> <sup>-160</sup>; DejaVu lacks the
    # superscript glyphs (they render as boxes) but has normal digits/minus.
    sup = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
           "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+", "ⁿ": "n"}
    body = re.sub("[" + "".join(sup) + "]+",
                  lambda m: "<sup>" + "".join(sup[c] for c in m.group(0)) + "</sup>",
                  body)

    body = re.sub(r'<p>\s*<img[^>]*src="(?P<src>[^"]+)"[^>]*>\s*</p>', img_repl, body)
    # caption paragraph = a <p> that is entirely italic
    body = re.sub(r'<p><em>(.*?)</em></p>',
                  r'<p class="caption"><em>\1</em></p>', body, flags=re.S)

    return f"<html><head><style>{CSS}</style></head><body>{body}</body></html>"


def link_callback(uri, rel):
    if uri.startswith("file:"):
        return uri[len("file:///"):] if os.name == "nt" else uri[len("file://"):]
    if uri.startswith("figures/"):
        return str((PAPER_DIR / uri).resolve())
    return uri


def main():
    register_fonts()
    html = build_html()
    with open(OUT, "wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, link_callback=link_callback,
                                encoding="utf-8")
    if result.err:
        raise SystemExit(f"PDF generation had {result.err} error(s).")
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
