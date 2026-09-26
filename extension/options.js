const DEFAULT_URL = "http://127.0.0.1:8765";
const url = document.getElementById("backendUrl");
const token = document.getElementById("token");
const msg = document.getElementById("msg");

chrome.storage.local.get(["backendUrl", "token"]).then((s) => {
  url.value = s.backendUrl || DEFAULT_URL;
  token.value = s.token || "";
});

document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  let u;
  try {
    u = new URL(url.value.trim() || DEFAULT_URL);
  } catch {
    msg.textContent = "That address is not a valid URL.";
    return;
  }
  if (u.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(u.hostname)) {
    msg.textContent = "Use an http://127.0.0.1 or http://localhost address.";
    return;
  }
  await chrome.storage.local.set({ backendUrl: u.origin, token: token.value.trim() });
  msg.textContent = "Saved.";
});
