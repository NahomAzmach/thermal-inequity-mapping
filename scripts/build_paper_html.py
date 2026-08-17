#!/usr/bin/env python
"""Render paper/paper.md to a self-contained styled HTML (images embedded as
base64 data URIs) so it opens anywhere and prints cleanly to PDF."""
import base64, re, mimetypes
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "paper.md"
OUT = ROOT / "paper" / "paper.html"

md_text = PAPER.read_text(encoding="utf-8")
html_body = markdown.markdown(md_text, extensions=["tables", "attr_list", "sane_lists"])

# inline <img src="figures/x.png"> as base64 data URIs
def embed(m):
    src = m.group(1)
    p = (PAPER.parent / src).resolve()
    if not p.exists():
        return m.group(0)
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'src="data:{mime};base64,{b64}"'

html_body = re.sub(r'src="([^"]+)"', embed, html_body)

CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:820px;margin:40px auto;
padding:0 24px;line-height:1.55;color:#1a1a1a;}
h1{font-size:1.7em;line-height:1.25;border-bottom:2px solid #333;padding-bottom:.3em;}
h2{font-size:1.3em;margin-top:1.6em;border-bottom:1px solid #ccc;padding-bottom:.2em;}
h3{font-size:1.1em;margin-top:1.3em;color:#333;}
img{max-width:100%;height:auto;display:block;margin:1.2em auto .3em;
border:1px solid #e0e0e0;border-radius:4px;}
em{color:#444;}
p em:only-child{display:block;font-size:.9em;text-align:center;margin:0 auto 1.4em;
max-width:90%;color:#555;}
table{border-collapse:collapse;width:100%;margin:1.2em 0;font-family:Helvetica,Arial,sans-serif;
font-size:.9em;}
th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;}
th{background:#f2f2f2;}
code{background:#f4f4f4;padding:1px 4px;border-radius:3px;font-size:.9em;}
hr{border:none;border-top:1px solid #ddd;margin:2em 0;}
a{color:#1a5fb4;}
@media print{body{margin:0;max-width:100%;} h2{page-break-after:avoid;} img{page-break-inside:avoid;}}
"""

html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Thermal Inequity Mapping — Addis Ababa</title>
<style>{CSS}</style></head><body>{html_body}</body></html>"""

OUT.write_text(html, encoding="utf-8")
kb = OUT.stat().st_size / 1024
print(f"Wrote {OUT}  ({kb:.0f} KB, images embedded)")
