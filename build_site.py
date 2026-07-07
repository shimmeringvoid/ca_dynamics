#!/usr/bin/env python3
"""Build a static HTML site from the executed chapter notebooks.

Pure Python: uses nbconvert (a standard Jupyter dependency). No Node,
no MyST, no BASE_URL configuration headaches. Output lands in ``site/``,
ready to publish with GitHub Pages.

Usage:
    python build_site.py            # build into ./site
    python build_site.py --serve    # build, then serve locally for preview
"""
from __future__ import annotations

import argparse
import pathlib
import re

from nbconvert import HTMLExporter

REPO = pathlib.Path(__file__).resolve().parent
NB_DIR = REPO / "notebooks"
OUT = REPO / "site"

# (filename stem, chapter number, human title) in reading order
CHAPTERS = [
    ("01_what_are_cellular_automata", 1, "What Are Cellular Automata?"),
    ("02_zero_dimensional_ca",        2, "Zero-Dimensional CA"),
    ("03_one_dimensional_ca",         3, "One-Dimensional CA"),
    ("04_two_dimensional_ca",         4, "Two-Dimensional CA"),
    ("05_ecosystem_modeling",         5, "Ecosystem Modeling"),
    ("06_wolfram_classification",     6, "Qualitative Classification"),
    ("07_evolutionary_dynamics",      7, "Evolutionary Dynamics"),
]

# A small, dependency-free stylesheet shared by every page.
CSS = """
:root { --ink:#1a1a2e; --link:#2a4d8f; --rule:#e2e2ea; --bg:#ffffff; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font: 16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
a { color:var(--link); text-decoration:none; }
a:hover { text-decoration:underline; }
.wrap { max-width: 62rem; margin: 0 auto; padding: 1.5rem 1.25rem 4rem; }
header.site { border-bottom:1px solid var(--rule); background:#fafafe; }
header.site .wrap { padding: 1rem 1.25rem; display:flex; flex-wrap:wrap;
                    gap:.5rem 1.25rem; align-items:baseline; }
header.site .home { font-weight:700; }
header.site nav a { margin-right: .9rem; white-space: nowrap; font-size: .92rem; }
h1.title { margin:.2rem 0 .1rem; }
.subtitle { color:#555; font-style:italic; margin:0 0 .4rem; }
.byline { color:#555; margin:.2rem 0 1.4rem; }
.chapter-nav { display:flex; justify-content:space-between; gap:1rem;
               border-top:1px solid var(--rule); margin-top:2.5rem;
               padding-top:1rem; font-size:.95rem; }
.card-list { list-style:none; padding:0; margin:1.5rem 0; }
.card-list li { border:1px solid var(--rule); border-radius:10px;
                padding:.9rem 1.1rem; margin:.6rem 0; }
.card-list .num { color:#999; font-variant-numeric:tabular-nums; margin-right:.6rem; }
.notebook-body { margin-top: 1.5rem; }
/* keep nbconvert's own output readable and constrained */
.notebook-body img { max-width:100%; height:auto; }
.notebook-body pre { overflow:auto; }
footer.site { border-top:1px solid var(--rule); color:#777;
              font-size:.9rem; margin-top:3rem; }
footer.site .wrap { padding: 1rem 1.25rem; }
"""

GH = "https://github.com/shimmeringvoid/ca_dynamics"


def header_html(active_stem: str | None) -> str:
    links = []
    for stem, num, _title in CHAPTERS:
        cls = ' style="font-weight:700"' if stem == active_stem else ""
        links.append(f'<a href="{stem}.html"{cls}>{num}</a>')
    nav = "".join(links)
    return f"""<header class="site"><div class="wrap">
      <a class="home" href="index.html">Cellular Automata Dynamics</a>
      <nav>Chapters: {nav}</nav>
      <span style="margin-left:auto"><a href="{GH}">GitHub</a></span>
    </div></header>"""


def footer_html() -> str:
    return f"""<footer class="site"><div class="wrap">
      Rafael Espericueta &middot; Professor of Mathematics Emeritus, Bakersfield College.
      Code MIT, text CC BY 4.0. &middot; <a href="{GH}">source on GitHub</a>
    </div></footer>"""


def chapter_nav(idx: int) -> str:
    prev_html = next_html = "<span></span>"
    if idx > 0:
        s, n, t = CHAPTERS[idx - 1]
        prev_html = f'<a href="{s}.html">&larr; {n}. {t}</a>'
    if idx < len(CHAPTERS) - 1:
        s, n, t = CHAPTERS[idx + 1]
        next_html = f'<a href="{s}.html">{n}. {t} &rarr;</a>'
    return f'<div class="chapter-nav">{prev_html}{next_html}</div>'


def build():
    OUT.mkdir(exist_ok=True)
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True

    for idx, (stem, num, title) in enumerate(CHAPTERS):
        nb_path = NB_DIR / f"{stem}.ipynb"
        body, _ = exporter.from_filename(str(nb_path))
        # pull just the rendered notebook body out of nbconvert's full doc
        m = re.search(r"<body[^>]*>(.*)</body>", body, re.DOTALL)
        inner = m.group(1) if m else body
        # nbconvert emits its own <style>; keep it, it styles code/outputs
        styles = "".join(re.findall(r"<style.*?</style>", body, re.DOTALL))

        page = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{num}. {title} — Cellular Automata Dynamics</title>
<style>{CSS}</style>
{styles}
</head><body>
{header_html(stem)}
<main class="wrap"><div class="notebook-body">{inner}</div>
{chapter_nav(idx)}</main>
{footer_html()}
</body></html>"""
        (OUT / f"{stem}.html").write_text(page, encoding="utf-8")
        print(f"  wrote {stem}.html")

    # landing page
    cards = "\n".join(
        f'<li><a href="{stem}.html"><span class="num">{num}.</span>{title}</a></li>'
        for stem, num, title in CHAPTERS
    )
    index = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cellular Automata Dynamics</title>
<style>{CSS}</style></head><body>
{header_html(None)}
<main class="wrap">
  <h1 class="title">Cellular Automata Dynamics</h1>
  <p class="subtitle">Explorations in Parallel Processing &mdash; Python Edition</p>
  <p class="byline">Rafael Espericueta, Professor of Mathematics Emeritus, Bakersfield College</p>
  <p>An interactive companion to the book. Each chapter is a runnable
     notebook &mdash; read the explanation, then run and modify the code.
     Everything is powered by the <a href="{GH}"><code>cadyn</code></a>
     Python package.</p>
  <p style="margin:1.5rem 0;">
    <a href="cellular_automata_dynamics.pdf"
       style="display:inline-block; background:var(--link); color:#fff;
              padding:.6rem 1.1rem; border-radius:8px; font-weight:600;">
      &#128214; Download the full book (PDF)</a>
    <span style="color:#777; margin-left:.6rem;">complete text, appendices,
      and bibliography</span>
  </p>
  <ul class="card-list">
  {cards}
  </ul>
  <p>The complete typeset book (PDF), the source code, and instructions
     for running the notebooks yourself are on
     <a href="{GH}">GitHub</a>.</p>
</main>
{footer_html()}
</body></html>"""
    (OUT / "index.html").write_text(index, encoding="utf-8")
    print("  wrote index.html")

    # copy the full typeset book into the site so it can be downloaded
    import shutil
    pdf_src = REPO / "latex" / "book.pdf"
    if pdf_src.exists():
        shutil.copy2(pdf_src, OUT / "cellular_automata_dynamics.pdf")
        print("  copied book.pdf -> site/cellular_automata_dynamics.pdf")

    # .nojekyll tells GitHub Pages to serve the files as-is (no Jekyll pass)
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    print("  wrote index.html")


def serve():
    import http.server, socketserver, functools
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(OUT))
    with socketserver.TCPServer(("", 8000), handler) as httpd:
        print("Serving at http://localhost:8000  (Ctrl-C to stop)")
        httpd.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true",
                    help="serve the built site locally after building")
    args = ap.parse_args()
    build()
    if args.serve:
        serve()
