// AltContext content script. Injected on request (activeTab + scripting) into the
// isolated world. It only reads the page; it never modifies alt attributes.
(() => {
  if (window.__altcontext) return;

  const MIN_SIDE = 24;
  const MAX_TEXT = 1500;
  const EXCLUDED = "nav, header, footer, aside, form, [role=navigation], [role=banner], [role=contentinfo], [role=complementary], [aria-hidden=true], .ad, .ads, .advert, .advertisement, [id*=sidebar], [class*=sidebar]";

  const clean = (t) => (t || "").replace(/\s+/g, " ").trim().slice(0, MAX_TEXT);
  const precedes = (a, b) => !!(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);

  function captionFor(img) {
    const fig = img.closest("figure");
    const fc = fig && fig.querySelector("figcaption");
    if (fc) return clean(fc.innerText);
    const ids = (img.getAttribute("aria-describedby") || "").split(/\s+/).filter(Boolean);
    const described = ids.map((id) => document.getElementById(id)).filter(Boolean).map((el) => el.innerText).join(" ");
    if (described) return clean(described);
    // Common non-figure caption patterns (e.g. MediaWiki thumbs).
    const thumb = img.closest(".thumb, .thumbinner, .wp-caption, .image-caption-wrapper");
    const tc = thumb && thumb.querySelector(".thumbcaption, .wp-caption-text, figcaption, .caption");
    return tc ? clean(tc.innerText) : "";
  }

  function headingFor(img) {
    let best = null;
    for (const h of document.querySelectorAll("h1, h2, h3, h4, h5, h6, [role=heading]")) {
      if (h.closest(EXCLUDED)) continue;
      if (precedes(h, img)) best = h; else break;
    }
    return best ? clean(best.innerText) : "";
  }

  function paragraphFor(img, caption) {
    // Search outward from the image for the closest substantial paragraph.
    const candidates = [...document.querySelectorAll("p")].filter((p) => {
      if (p.closest(EXCLUDED) || p.closest("figure")) return false;
      const t = clean(p.innerText);
      return t.length >= 40 && t !== caption;
    });
    if (!candidates.length) return "";
    const rect = img.getBoundingClientRect();
    const cy = rect.top + rect.height / 2;
    let best = null, bestScore = Infinity;
    for (const p of candidates) {
      const r = p.getBoundingClientRect();
      const dist = r.bottom < cy ? cy - r.bottom : r.top > cy ? r.top - cy : 0;
      // Prefer paragraphs in the same section as the image.
      const sameSection = p.closest("section, article, main") === img.closest("section, article, main");
      const score = dist + (sameSection ? 0 : 2000);
      if (score < bestScore) { best = p; bestScore = score; }
    }
    return best ? clean(best.innerText) : "";
  }

  function altInfo(img) {
    const role = (img.getAttribute("role") || "").toLowerCase();
    if (!img.hasAttribute("alt")) return { status: "missing", text: "" };
    const alt = img.getAttribute("alt");
    if (alt.trim() === "" || role === "presentation" || role === "none") return { status: "empty", text: "" };
    return { status: "existing", text: clean(alt) };
  }

  function functionalInfo(img) {
    const ctl = img.closest("a[href], button, [role=button], [role=link]");
    if (!ctl) return null;
    const label = clean(ctl.getAttribute("aria-label") || ctl.innerText || img.getAttribute("alt") || "");
    return { kind: ctl.tagName === "A" ? "link" : "button", label };
  }

  function candidates() {
    return [...document.images].filter((img) => {
      if (img.closest("[aria-hidden=true]")) return false;
      const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
      const src = img.currentSrc || img.src;
      if (!src) return false;
      if (img.complete && img.naturalWidth && (w < MIN_SIDE || h < MIN_SIDE)) return false; // icons / trackers
      return true;
    });
  }

  function describeElement(img, index, total) {
    const caption = captionFor(img);
    const alt = altInfo(img);
    return {
      index,
      total,
      src: img.currentSrc || img.src,
      width: img.naturalWidth || 0,
      height: img.naturalHeight || 0,
      loaded: img.complete && img.naturalWidth > 0,
      alt,
      functional: functionalInfo(img),
      context: {
        caption,
        heading: headingFor(img),
        paragraph: paragraphFor(img, caption),
        page_title: clean(document.title),
      },
    };
  }

  let registry = [];

  function scan() {
    registry = candidates();
    registry.forEach((img, i) => img.setAttribute("data-altcontext-index", String(i + 1)));
    return {
      url: location.href,
      title: document.title,
      images: registry.map((img, i) => describeElement(img, i + 1, registry.length)),
    };
  }

  // Re-read the element at request time so the request uses current src and context.
  function collect(index) {
    const img = registry[index - 1];
    if (!img || !img.isConnected) return { error: "stale", message: "That image is no longer on the page. Scan the page again." };
    return describeElement(img, index, registry.length);
  }

  function highlight(index) {
    const img = registry[index - 1];
    if (!img || !img.isConnected) return false;
    img.scrollIntoView({ block: "center", behavior: "smooth" });
    const prev = img.style.outline;
    img.style.outline = "4px solid #d9480f";
    img.style.outlineOffset = "2px";
    setTimeout(() => { img.style.outline = prev; img.style.outlineOffset = ""; }, 2500);
    return true;
  }

  // Fallback byte access for blob:/same-origin/CORS-clean images the worker cannot fetch.
  async function bytes(index) {
    const img = registry[index - 1];
    if (!img) return { error: "stale" };
    const src = img.currentSrc || img.src;
    try {
      const res = await fetch(src);
      if (res.ok) {
        const blob = await res.blob();
        return await new Promise((resolve) => {
          const r = new FileReader();
          r.onload = () => resolve({ dataUrl: r.result });
          r.onerror = () => resolve({ error: "read_failed" });
          r.readAsDataURL(blob);
        });
      }
    } catch (_) { /* fall through to canvas */ }
    try {
      if (img.complete && !img.naturalWidth) return { error: "broken" };
      if (!img.complete) return { error: "not_loaded" };
      const c = document.createElement("canvas");
      const scale = Math.min(1, 1024 / Math.max(img.naturalWidth, img.naturalHeight));
      c.width = Math.round(img.naturalWidth * scale);
      c.height = Math.round(img.naturalHeight * scale);
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      return { dataUrl: c.toDataURL("image/png") };
    } catch (_) {
      return { error: "tainted" };
    }
  }

  window.__altcontext = { scan, collect, highlight, bytes };
})();
