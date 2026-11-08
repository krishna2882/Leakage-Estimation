"""Renders report.md (written by 09_report.py) to report.pdf: Markdown -> HTML -> headless Chrome/Chromium print.
Run on a machine with Chrome or Chromium (not needed on Ada): python3 12_report_pdf.py"""
import os
import re
import shutil
import subprocess
import tempfile
import markdown

ROOT = os.path.dirname(os.path.abspath(__file__))

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 10pt; line-height: 1.38; color: #111; }
h1 { font-size: 17pt; margin: 0 0 8pt; border-bottom: 1.5pt solid #333; padding-bottom: 4pt; }
h2 { font-size: 13pt; margin: 16pt 0 5pt; border-bottom: .6pt solid #999; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 11pt 0 4pt; break-after: avoid; }
p, li { text-align: justify; }
ul { padding-left: 16pt; margin: 4pt 0; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.4pt; }
pre { background: #f4f4f4; border: .5pt solid #ccc; padding: 5pt 6pt; font-size: 7.6pt; line-height: 1.3;
      white-space: pre-wrap; word-break: break-all; break-inside: avoid; }
pre code { font-size: inherit; }
table { border-collapse: collapse; margin: 5pt 0 8pt; font-size: 8pt; break-inside: auto; }
th, td { border: .5pt solid #aaa; padding: 2pt 5pt; text-align: right; }
th { background: #e8e8e8; }
td:first-child, th:first-child { text-align: left; }
tr { break-inside: avoid; }
img { display: block; max-width: 66%; max-height: 95mm; margin: 6pt auto; break-inside: avoid; }
"""


def blank_before_blocks(md):
    """Python-Markdown needs an empty line before a list or table that follows a paragraph line, and after a table
    (otherwise the next line becomes a table row)"""
    out, prev = [], ""
    for line in md.split("\n"):
        starts = re.match(r"\s*(- |\* |\d+\. |\|)", line)
        prev_same = re.match(r"\s*(- |\* |\d+\. |\|)", prev) or prev.startswith("  ")
        if starts and prev.strip() and not prev_same and not line.startswith(" "):
            out.append("")
        elif prev.startswith("|") and line.strip() and not line.startswith("|"):
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def browser():
    for b in ("google-chrome", "chromium", "chromium-browser", "google-chrome-stable"):
        if shutil.which(b):
            return shutil.which(b)
    raise SystemExit("no Chrome/Chromium found: install one or print report.html from a browser")


if __name__ == "__main__":
    md = open(os.path.join(ROOT, "report.md")).read()
    body = markdown.markdown(blank_before_blocks(md), extensions=["tables", "fenced_code", "sane_lists"])
    title = re.search(r"^# (.+)$", md, re.M).group(1)
    html = f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>'
    # the HTML sits next to report.md so the relative plot paths resolve
    fh = os.path.join(ROOT, "report.html")
    open(fh, "w").write(html)
    out = os.path.join(ROOT, "report.pdf")
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([browser(), "--headless", "--disable-gpu", "--no-sandbox", f"--user-data-dir={prof}",
                        "--no-pdf-header-footer", f"--print-to-pdf={out}", "file://" + fh],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    os.remove(fh)
    print("wrote", out, f"({os.path.getsize(out) / 1e6:.1f} MB)")
