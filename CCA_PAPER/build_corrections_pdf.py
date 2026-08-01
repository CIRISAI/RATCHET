#!/usr/bin/env python3
"""
Build CORRECTIONS_v3.pdf from CORRECTIONS_v3.md for the Zenodo deposit.

The markdown file is canonical; this script only renders it. Run:
    python3 build_corrections_pdf.py
"""

import re
import sys
from pathlib import Path

import markdown
from weasyprint import HTML

HERE = Path(__file__).parent
SRC = HERE / "CORRECTIONS_v3.md"
OUT = HERE / "CORRECTIONS_v3.pdf"

CSS = """
@page {
    size: A4;
    margin: 2cm 2cm 2.2cm 2cm;
    @bottom-center {
        content: "Corrections to Coherence Collapse Analysis v3 \\2014 page " counter(page);
        font-family: 'DejaVu Serif', Georgia, serif;
        font-size: 8pt;
        color: #666;
    }
}
body {
    font-family: 'DejaVu Serif', Georgia, serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #111;
}
h1 { font-size: 18pt; margin: 0 0 0.4em 0; line-height: 1.2; }
h1:not(:first-of-type) { margin-top: 1.4em; page-break-before: always; }
h2 { font-size: 13pt; margin: 1.5em 0 0.4em 0; border-bottom: 1px solid #ccc;
     padding-bottom: 0.15em; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 1.1em 0 0.3em 0; page-break-after: avoid; }
p { margin: 0.45em 0; orphans: 3; widows: 3; }
code {
    font-family: 'DejaVu Sans Mono', monospace;
    font-size: 8.5pt;
    background: #f4f4f4;
    padding: 0.08em 0.28em;
    border-radius: 2px;
}
pre {
    background: #f6f6f6;
    border: 1px solid #ddd;
    border-left: 3px solid #888;
    padding: 0.6em 0.8em;
    font-size: 8pt;
    line-height: 1.35;
    overflow-x: auto;
    page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 8pt; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.7em 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
}
th, td { border: 1px solid #bbb; padding: 0.32em 0.5em; text-align: left;
         vertical-align: top; }
th { background: #eee; font-weight: bold; }
blockquote {
    margin: 0.7em 0;
    padding: 0.4em 0.9em;
    border-left: 3px solid #999;
    background: #fafafa;
    page-break-inside: avoid;
}
blockquote p { margin: 0.25em 0; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.4em 0; }
a { color: #14507a; text-decoration: none; }
strong { font-weight: bold; }
ul, ol { margin: 0.45em 0; padding-left: 1.4em; }
li { margin: 0.18em 0; }
"""


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found", file=sys.stderr)
        return 1

    text = SRC.read_text(encoding="utf-8")

    # Strip the leading H1 duplicate handling: keep as-is, the CSS handles it.
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )

    # Markdown leaves bare ~~strike~~ alone; render it.
    html_body = re.sub(r"~~(.+?)~~", r"<s>\1</s>", html_body, flags=re.S)

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{html_body}</body></html>"
    )

    HTML(string=html, base_url=str(HERE)).write_pdf(OUT)
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.name} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
