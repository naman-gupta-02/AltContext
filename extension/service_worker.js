// AltContext service worker: coordinates page access, image fetching and backend calls.
// It only fetches the image URL read from the selected page element at request time;
// it is not a general-purpose fetch proxy.

const DEFAULTS = { backendUrl: "http://127.0.0.1:8765", token: "", mode: "brief", useContext: true };
const MODEL_FORMATS = new Set(["image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"]);
const MAX_UPLOAD = 4 * 1024 * 1024;
const REQUEST_TIMEOUT_MS = 120000;
const inflight = new Map(); // requestId -> AbortController

chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});
});
chrome.runtime.onStartup.addListener(() => {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});
});

async function settings() {
  return { ...DEFAULTS, ...(await chrome.storage.local.get(Object.keys(DEFAULTS))) };
}

async function inPage(tabId, fn, args = []) {
  await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
  const [res] = await chrome.scripting.executeScript({ target: { tabId }, func: fn, args });
  return res && res.result;
}

function fail(error, message) {
  return { ok: false, error, message };
}

async function health() {
  const s = await settings();
  try {
    const res = await fetch(`${s.backendUrl}/health`, { signal: AbortSignal.timeout(4000) });
    return { ok: true, health: await res.json(), hasToken: !!s.token };
  } catch (e) {
    return fail("backend_unreachable", `Cannot reach the AltContext service at ${s.backendUrl}. Start it with: python -m backend.app`);
  }
}

async function scan(tabId) {
  try {
    const data = await inPage(tabId, () => window.__altcontext.scan());
    return { ok: true, data };
  } catch (e) {
    return fail("no_page_access", "AltContext cannot read this page. Browser pages (chrome://, the Web Store) are not supported; on other sites, grant access and scan again.");
  }
}

async function dataUrlToBlob(dataUrl) {
  return (await fetch(dataUrl)).blob();
}

async function normalizeBlob(blob) {
  // Re-encode formats the model service does not accept (e.g. AVIF) and shrink very large files.
  if (MODEL_FORMATS.has(blob.type) && blob.size <= MAX_UPLOAD) return blob;
  let bitmap;
  try {
    bitmap = await createImageBitmap(blob);
  } catch (_) {
    throw Object.assign(new Error("undecodable"), { code: "unsupported_image" });
  }
  const scale = Math.min(1, 1536 / Math.max(bitmap.width, bitmap.height));
  const canvas = new OffscreenCanvas(Math.max(1, Math.round(bitmap.width * scale)), Math.max(1, Math.round(bitmap.height * scale)));
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return canvas.convertToBlob({ type: "image/jpeg", quality: 0.92 });
}

async function imageBytes(tabId, index, src, signal) {
  const url = new URL(src);
  if (url.protocol === "data:") return dataUrlToBlob(src);
  if (url.protocol === "http:" || url.protocol === "https:") {
    try {
      const res = await fetch(src, { signal, credentials: "omit", cache: "force-cache" });
      if (res.ok) {
        const type = (res.headers.get("content-type") || "").split(";")[0];
        if (type && !type.startsWith("image/")) throw Object.assign(new Error("not an image"), { code: "not_image" });
        return await res.blob();
      }
    } catch (e) {
      if (signal.aborted || e.code === "not_image") throw e;
      // fall back to the page context below
    }
  }
  if (url.protocol !== "http:" && url.protocol !== "https:" && url.protocol !== "blob:") {
    throw Object.assign(new Error("scheme"), { code: "unsupported_url" });
  }
  const r = await inPage(tabId, (i) => window.__altcontext.bytes(i), [index]);
  if (r && r.dataUrl) return dataUrlToBlob(r.dataUrl);
  throw Object.assign(new Error("blocked"), { code: r && (r.error === "not_loaded" || r.error === "broken") ? r.error : "blocked_fetch", origin: url.origin });
}

async function describe({ tabId, index, mode, useContext, requestId }) {
  const s = await settings();
  if (!s.token) return fail("no_token", "Add the service token in AltContext settings (it is printed when the backend starts).");
  const controller = new AbortController();
  inflight.set(requestId, controller);
  const timer = setTimeout(() => controller.abort("timeout"), REQUEST_TIMEOUT_MS);
  try {
    let info;
    try {
      info = await inPage(tabId, (i) => window.__altcontext.collect(i), [index]);
    } catch (_) {
      return fail("no_page_access", "AltContext lost access to this page. Scan the page again.");
    }
    if (!info || info.error) return fail("stale", info ? info.message : "Scan the page again.");

    let blob;
    try {
      blob = await normalizeBlob(await imageBytes(tabId, index, info.src, controller.signal));
    } catch (e) {
      if (controller.signal.aborted) throw e;
      const messages = {
        blocked_fetch: `The image host (${e.origin || "unknown"}) blocked access. Grant AltContext access to that site and try again.`,
        broken: "This image failed to load on the page (broken image), so it cannot be described.",
        not_loaded: "The image has not loaded yet (it may be lazy-loaded). Scroll to it, then try again.",
        unsupported_image: "This image format cannot be decoded (SVG and some formats are not supported yet).",
        unsupported_url: "This image source type is not supported.",
        not_image: "The image URL did not return an image.",
      };
      return fail(e.code || "blocked_fetch", messages[e.code] || messages.blocked_fetch);
    }

    const form = new FormData();
    form.append("image", blob, "image");
    form.append("metadata", JSON.stringify({ ...info.context, mode, use_context: useContext }));
    let res;
    try {
      res = await fetch(`${s.backendUrl}/describe`, { method: "POST", body: form, headers: { "X-AltContext-Token": s.token }, signal: controller.signal });
    } catch (e) {
      if (controller.signal.aborted) throw e;
      return fail("backend_unreachable", `Cannot reach the AltContext service at ${s.backendUrl}. Is it running?`);
    }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const d = body.detail || {};
      return fail(d.error || `http_${res.status}`, d.message || `The service returned an error (${res.status}).`);
    }
    return { ok: true, result: body, context: useContext ? info.context : null, alt: info.alt, functional: info.functional };
  } catch (e) {
    if (controller.signal.aborted) {
      return controller.signal.reason === "timeout" ? fail("timeout", "The request timed out.") : fail("cancelled", "Request cancelled.");
    }
    return fail("unexpected", `Unexpected error: ${e.message}`);
  } finally {
    clearTimeout(timer);
    inflight.delete(requestId);
  }
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  const handlers = {
    health: () => health(),
    scan: () => scan(msg.tabId),
    describe: () => describe(msg),
    highlight: () => inPage(msg.tabId, (i) => window.__altcontext.highlight(i), [msg.index]).then((ok) => ({ ok })).catch(() => fail("no_page_access", "Cannot reach the page.")),
    cancel: async () => {
      const c = inflight.get(msg.requestId);
      if (c) c.abort("cancelled");
      return { ok: !!c };
    },
  };
  const h = handlers[msg && msg.type];
  if (!h) return false;
  h().then(sendResponse, (e) => sendResponse(fail("unexpected", String(e))));
  return true; // async response
});
