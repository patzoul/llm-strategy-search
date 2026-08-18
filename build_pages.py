"""Generate docs/index.html for GitHub Pages from report.html.

report.html is written as an Artifact *fragment*: it starts at <title> and has
no doctype, no <head>, no charset and no viewport, because the Artifact host
supplies that wrapper. Served directly by Pages it would render as tag soup --
the em dashes and minus signs would mojibake without a charset, and it would be
unusable on a phone without a viewport tag.

This wraps the same content in a real document so there is one source of truth:
edit report.html, re-run this, and both the Artifact and the Pages site match.

    python build_pages.py
"""

from __future__ import annotations

import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "report.html")
OUT = os.path.join(HERE, "docs", "index.html")

REPO = "https://github.com/patzoul/llm-strategy-search"
DESC = ("Eight LLM-generated trading strategies tested against surrogate data "
        "containing nothing. None established an edge.")

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<meta name="color-scheme" content="light dark">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:type" content="article">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='14' font-size='14'>&#128207;</text></svg>">
<style>
/* Minimal reset. The Artifact host supplies one; Pages does not. */
*,*::before,*::after{{box-sizing:border-box}}
body{{margin:0}}
img,svg{{max-width:100%}}
</style>
</head>
<body>
"""

FOOT = """
<p style="max-width:78ch;margin:3rem auto 4rem;padding:0 1.4rem;
          font-family:ui-monospace,'SF Mono','Cascadia Mono',Menlo,Consolas,monospace;
          font-size:0.72rem;line-height:1.7;opacity:0.75">
  Code, run logs, fitted parameters and the full surrogate score arrays:
  <a href="{repo}" style="color:inherit">{repo}</a><br>
  Corrections and methodological objections are welcome as
  <a href="{repo}/issues" style="color:inherit">issues</a>.
</p>
</body>
</html>
"""


def main() -> None:
    src = io.open(SRC, encoding="utf-8").read()
    m = re.search(r"<title>(.*?)</title>", src, re.S)
    if not m:
        raise SystemExit("report.html has no <title> to build from")
    title = m.group(1).strip()
    body = src.replace(m.group(0), "", 1).lstrip("\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    page = HEAD.format(title=title, desc=DESC) + body + FOOT.format(repo=REPO)
    tmp = OUT + ".new"
    io.open(tmp, "w", encoding="utf-8", newline="").write(page)
    os.replace(tmp, OUT)
    print(f"wrote {OUT}  ({len(page):,} chars, title={title!r})")


if __name__ == "__main__":
    main()
