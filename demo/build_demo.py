"""Build a controlled demo article from a few Concadia validation images.

The page exercises the functional test matrix: existing / empty / missing alt,
an image inside a link, a lazy-loaded image and a broken image. Images come
from Wikimedia Commons via Concadia; attribution URLs are listed on the page.
The generated folder (demo/site) is git-ignored.

Usage: python demo/build_demo.py   then open http://127.0.0.1:8765/demo/ with the backend running
"""
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from altcontext.data import image_path, load_subset  # noqa: E402

SITE = ROOT / "demo" / "site"
N = 6


def main():
    rows = [r for r in load_subset("val")[100:] if len(r["context"].split()) > 30 and len(r["caption"].split()) > 4][:N]
    manifest = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data" / "processed" / "manifest.jsonl") if any(r["id"] in l for r in rows)}
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "img").mkdir(parents=True)
    cases = ["existing", "missing", "empty", "link", "lazy", "missing"]
    sections = []
    for i, (r, case) in enumerate(zip(rows, cases)):
        name = f"{i + 1}.jpg"
        shutil.copy(image_path(r), SITE / "img" / name)
        cap = html.escape(r["caption"])
        para = html.escape(" ".join(r["context"].split()[:120]))
        alt_attr = {"existing": f' alt="{html.escape(r["target_description"])}"', "empty": ' alt=""', "missing": "", "link": "", "lazy": ""}[case]
        lazy = ' loading="lazy"' if case == "lazy" else ""
        img = f'<img src="img/{name}"{alt_attr}{lazy} width="256" height="256">'
        if case == "link":
            img = f'<a href="#top" aria-label="Back to top">{img}</a>'
        sections.append(f"""
  <section>
    <h2>{html.escape(r['page_title'])}</h2>
    <div class="case">Test case: {case} alt</div>
    <figure>{img}<figcaption>{cap}</figcaption></figure>
    <p>{para}</p>
  </section>""")
    attributions = "\n".join(f'<li>Image {i + 1}: <a href="{html.escape(manifest[r["id"]].get("source_url") or "")}">{html.escape(manifest[r["id"]].get("source_url") or "")}</a> (via Concadia, Wikipedia article “{html.escape(r["page_title"])}”)</li>' for i, r in enumerate(rows))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>AltContext demo article</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font:16px/1.6 Georgia,serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#222}} figure{{margin:1rem 0}} figcaption{{font-size:.9rem;color:#555}} .case{{font:12px monospace;color:#8a3b00}} nav{{background:#eee;padding:.5rem}}</style>
</head><body id="top">
<nav aria-label="Site"><a href="#top">Home</a> · <a href="#top">Articles</a> · <a href="#top">About</a> — navigation text is excluded from context</nav>
<h1>AltContext demo article</h1>
<p>This page is a controlled test fixture. Each section places one image with a caption and a nearby paragraph, and uses a different alt-text case.</p>
{''.join(sections)}
  <section>
    <h2>Broken image</h2>
    <div class="case">Test case: broken image (should show a clear error)</div>
    <figure><img src="img/does-not-exist.jpg"><figcaption>This image file does not exist.</figcaption></figure>
    <p>The extension should report that the image could not be loaded and remain responsive.</p>
  </section>
<footer><h2>Image sources</h2><ol>{attributions}</ol></footer>
</body></html>"""
    (SITE / "index.html").write_text(page)
    print(f"wrote {SITE / 'index.html'} with {len(rows)} images")


if __name__ == "__main__":
    main()
