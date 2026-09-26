// Headless Chrome check of the extension against the fake backend.
// Setup: npm i puppeteer-core ; python demo/build_demo.py
//        ALTCONTEXT_TOKEN=testtok ALTCONTEXT_FAKE_DELAY_S=2 python -m backend.app --fake --port 8799
// Run:   node tests/extension_e2e.mjs
import puppeteer from "puppeteer-core";
import fs from "fs";
import os from "os";

const EXT = new URL("../extension", import.meta.url).pathname;
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true,
  pipe: true,
  enableExtensions: true,
  userDataDir: fs.mkdtempSync(os.tmpdir() + "/altcontext-e2e-"),
});
const id = await browser.installExtension(EXT);
console.log("extension id", id);

const demo = await browser.newPage();
await demo.goto("http://127.0.0.1:8799/demo/", { waitUntil: "networkidle0" });

const panel = await browser.newPage();
await panel.goto(`chrome-extension://${id}/options.html`);
await panel.evaluate(() => chrome.storage.local.set({ backendUrl: "http://127.0.0.1:8799", token: "testtok" }));
const tabId = await panel.evaluate(async () => (await chrome.tabs.query({ url: "http://127.0.0.1/*" }))[0].id);

const send = (msg) => panel.evaluate((m) => chrome.runtime.sendMessage(m), msg);
console.log("health", JSON.stringify(await send({ type: "health" })));
const scan = await send({ type: "scan", tabId });
if (!scan.ok) { console.log("scan failed", scan); process.exit(1); }
for (const im of scan.data.images) {
  console.log(`#${im.index} alt=${im.alt.status}${im.functional ? " func=" + im.functional.kind + ":" + im.functional.label : ""} loaded=${im.loaded}`);
  console.log(`   heading="${im.context.heading.slice(0, 60)}" caption="${im.context.caption.slice(0, 60)}"`);
  console.log(`   paragraph="${im.context.paragraph.slice(0, 70)}"`);
}
const d1 = await send({ type: "describe", tabId, index: 2, mode: "brief", useContext: true, requestId: "r1" });
console.log("describe #2", JSON.stringify(d1.ok ? d1.result : d1));
const d2 = await send({ type: "describe", tabId, index: 2, mode: "brief", useContext: true, requestId: "r2" });
console.log("repeat #2 cached =", d2.result && d2.result.cached);
const d3 = await send({ type: "describe", tabId, index: 2, mode: "brief", useContext: false, requestId: "r3" });
console.log("no-context #2 cached =", d3.result && d3.result.cached, "context_used =", d3.result && d3.result.context_used);
const broken = scan.data.images.find((i) => i.src.includes("does-not-exist"));
const d4 = await send({ type: "describe", tabId, index: broken.index, mode: "brief", useContext: true, requestId: "r4" });
console.log("broken image ->", JSON.stringify(d4));
// cancel test
const p = send({ type: "describe", tabId, index: 3, mode: "detailed", useContext: true, requestId: "r5" });
await new Promise((r) => setTimeout(r, 50));
await send({ type: "cancel", requestId: "r5" });
console.log("cancel ->", JSON.stringify(await p));

// Side panel page renders and scans when opened as a tab next to the demo page.
await demo.bringToFront();
const sp = await browser.newPage();
await sp.goto(`chrome-extension://${id}/sidepanel.html`);
await new Promise((r) => setTimeout(r, 800));
console.log("panel service line:", await sp.$eval("#service", (e) => e.textContent));
await browser.close();
