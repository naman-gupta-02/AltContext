// AltContext side panel. All page text and model output are inserted with
// textContent, never as HTML.

const $ = (sel) => document.querySelector(sel);
const statusEl = $("#status");
const serviceEl = $("#service");
const list = $("#images");
let current = { tabId: null, pageOrigin: null, images: [] };
let grantedOrigins = new Set();

const send = (msg) => chrome.runtime.sendMessage(msg);
const announce = (text) => { statusEl.textContent = ""; setTimeout(() => { statusEl.textContent = text; }, 50); };
const mode = () => document.querySelector("input[name=mode]:checked").value;
const useContext = () => $("#use-context").checked;

async function loadPrefs() {
  const s = await chrome.storage.local.get(["mode", "useContext"]);
  if (s.mode) document.querySelector(`input[name=mode][value=${s.mode}]`).checked = true;
  if (typeof s.useContext === "boolean") $("#use-context").checked = s.useContext;
  document.querySelectorAll("input[name=mode]").forEach((r) => r.addEventListener("change", () => chrome.storage.local.set({ mode: mode() })));
  $("#use-context").addEventListener("change", () => chrome.storage.local.set({ useContext: useContext() }));
}

async function refreshHealth() {
  const r = await send({ type: "health" });
  if (!r.ok) {
    serviceEl.dataset.state = "error";
    serviceEl.textContent = r.message;
    return;
  }
  const h = r.health;
  serviceEl.dataset.state = h.status === "ready" ? "ready" : h.status === "loading" ? "loading" : "error";
  if (h.status === "ready") {
    serviceEl.textContent = `Local model ready: ${h.model_version}` + (r.hasToken ? "" : " — add the service token in Settings.");
  } else if (h.status === "loading") {
    serviceEl.textContent = "Local model is loading…";
    setTimeout(refreshHealth, 3000);
  } else {
    serviceEl.textContent = `Local model unavailable: ${h.error || "unknown error"}`;
  }
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

function originOf(url) {
  try { const u = new URL(url); return u.protocol.startsWith("http") ? u.origin : null; } catch { return null; }
}

async function refreshGrants(origins) {
  grantedOrigins = new Set();
  for (const o of origins) {
    if (await chrome.permissions.contains({ origins: [`${o}/*`] })) grantedOrigins.add(o);
  }
}

async function doScan() {
  const btn = $("#scan");
  btn.disabled = true;
  announce("Scanning the page for images…");
  try {
    const tab = await activeTab();
    if (!tab) return announce("No active tab.");
    let r = await send({ type: "scan", tabId: tab.id });
    if (!r.ok && r.error === "no_page_access") {
      const origin = originOf(tab.url || "");
      if (origin && (await chrome.permissions.request({ origins: [`${origin}/*`] }))) {
        r = await send({ type: "scan", tabId: tab.id });
      }
    }
    if (!r.ok) return announce(r.message);
    current = { tabId: tab.id, pageOrigin: originOf(r.data.url), images: r.data.images };
    await refreshGrants(new Set(current.images.map((i) => originOf(i.src)).filter(Boolean)));
    render();
    const n = current.images.length;
    announce(n ? `Found ${n} image${n === 1 ? "" : "s"}.` : "No images found on this page.");
  } finally {
    btn.disabled = false;
  }
}

const ALT_LABEL = {
  existing: "Existing alt text",
  empty: "Alt is empty (marked decorative)",
  missing: "No alt attribute",
};

function fact(dl, term, value) {
  if (!value) return;
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  dd.textContent = value;
  dl.append(dt, dd);
}

function render() {
  list.replaceChildren();
  $("#empty").hidden = current.images.length > 0;
  if (!current.images.length) $("#empty").textContent = "No images found on this page.";
  const tpl = $("#image-item");
  for (const img of current.images) {
    const li = tpl.content.firstElementChild.cloneNode(true);
    li.dataset.index = img.index;
    li.querySelector(".image-title").textContent = `Image ${img.index} of ${img.total}`;
    const thumb = li.querySelector(".thumb");
    if (img.src.startsWith("http") || img.src.startsWith("data:")) thumb.src = img.src; else thumb.hidden = true;
    const dl = li.querySelector(".facts");
    fact(dl, "Alt", img.alt.status === "existing" ? `${ALT_LABEL.existing}: “${img.alt.text}”` : ALT_LABEL[img.alt.status]);
    if (img.functional) fact(dl, "Function", `Inside a ${img.functional.kind}` + (img.functional.label ? ` labelled “${img.functional.label}”` : "") + ". Its label describes the action and is kept.");
    fact(dl, "Heading", img.context.heading);
    fact(dl, "Caption", img.context.caption);
    if (!img.loaded) fact(dl, "Note", "Not loaded yet (may be lazy-loaded).");

    const describe = li.querySelector(".describe");
    describe.textContent = img.alt.status === "empty" ? `Describe image ${img.index} anyway` : img.alt.status === "existing" ? `Get AI description for image ${img.index}` : `Describe image ${img.index}`;
    describe.addEventListener("click", () => doDescribe(img, li));
    li.querySelector(".show").addEventListener("click", () => send({ type: "highlight", tabId: current.tabId, index: img.index }));
    li.querySelector(".show").setAttribute("aria-label", `Show image ${img.index} on the page`);
    li.querySelector(".copy").addEventListener("click", async () => {
      await navigator.clipboard.writeText(li.querySelector(".result-text").textContent);
      announce("Description copied.");
    });
    list.append(li);
  }
}

async function doDescribe(img, li) {
  const describeBtn = li.querySelector(".describe");
  const cancelBtn = li.querySelector(".cancel");
  const errorEl = li.querySelector(".error");
  const resultEl = li.querySelector(".result");
  errorEl.hidden = true;

  // Cross-origin images need host access to the image's site. Ask while the click's
  // user gesture is still active (before any other await).
  const imgOrigin = originOf(img.src);
  if (imgOrigin && imgOrigin !== current.pageOrigin && !grantedOrigins.has(imgOrigin)) {
    const granted = await chrome.permissions.request({ origins: [`${imgOrigin}/*`] });
    if (granted) grantedOrigins.add(imgOrigin);
  }

  const requestId = crypto.randomUUID();
  describeBtn.disabled = true;
  cancelBtn.hidden = false;
  cancelBtn.onclick = () => send({ type: "cancel", requestId });
  li.setAttribute("aria-busy", "true");
  announce(`Describing image ${img.index}…`);
  const m = mode();
  const r = await send({ type: "describe", tabId: current.tabId, index: img.index, mode: m, useContext: useContext(), requestId });
  li.removeAttribute("aria-busy");
  describeBtn.disabled = false;
  const hadFocus = document.activeElement === cancelBtn;
  cancelBtn.hidden = true;
  if (hadFocus) describeBtn.focus(); // keep focus on a visible control

  if (!r.ok) {
    errorEl.textContent = r.message;
    errorEl.hidden = false;
    announce(r.error === "cancelled" ? `Cancelled image ${img.index}.` : `Could not describe image ${img.index}.`);
    return;
  }
  const res = r.result;
  const secs = ((res.generation_latency_ms ?? res.latency_ms) / 1000).toFixed(1);
  li.querySelector(".result-label").textContent =
    `AI-generated description (${res.mode}${res.context_used ? ", with page context" : ", image only"}; ${res.model_version}; ${res.cached ? "cached" : `${secs} s`})`;
  li.querySelector(".result-text").textContent = res.description;
  const ctxEl = li.querySelector(".result-context");
  const c = r.context;
  ctxEl.hidden = !c;
  if (c) ctxEl.querySelector("pre").textContent = ["caption", "heading", "paragraph", "page_title"].filter((k) => c[k]).map((k) => `${k}: ${c[k]}`).join("\n");
  resultEl.hidden = false;
  announce(`Description ready for image ${img.index}: ${res.description}`);
}

$("#scan").addEventListener("click", doScan);
chrome.tabs.onActivated.addListener(() => {
  current = { tabId: null, pageOrigin: null, images: [] };
  list.replaceChildren();
  $("#empty").hidden = false;
  $("#empty").textContent = "Tab changed. Scan this page for images.";
});
loadPrefs();
refreshHealth();
