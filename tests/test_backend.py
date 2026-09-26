import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app

TOKEN = "test-token"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("ALTCONTEXT_FAKE_DELAY_S", "0")
    app = create_app(fake=True, token=TOKEN, adapter="none")
    with TestClient(app) as c:
        for _ in range(50):
            if c.get("/health").json()["status"] == "ready":
                break
            time.sleep(0.05)
        yield c


def png(color=(10, 120, 200), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def post(client, data, meta=None, token=TOKEN, headers=None):
    h = {"X-AltContext-Token": token} if token else {}
    h.update(headers or {})
    return client.post("/describe", files={"image": ("i.png", data, "image/png")}, data={"metadata": json.dumps(meta or {})}, headers=h)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ready" and body["model_version"] == "fake-model"


def test_requires_token(client):
    assert post(client, png(), token=None).status_code == 401
    assert post(client, png(), token="wrong").status_code == 401


def test_rejects_web_origin(client):
    r = post(client, png(), headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_describe_and_cache(client):
    meta = {"caption": "A bridge", "paragraph": "The article discusses pedestrian access.", "mode": "brief"}
    r1 = post(client, png(), meta)
    assert r1.status_code == 200, r1.text
    b1 = r1.json()
    assert b1["cached"] is False and b1["ai_generated"] is True and b1["context_used"] is True
    r2 = post(client, png(), meta).json()
    assert r2["cached"] is True and r2["description"] == b1["description"]
    # Changed context must not hit the cache.
    r3 = post(client, png(), {**meta, "paragraph": "Different text."}).json()
    assert r3["cached"] is False
    # Changed mode must not hit the cache.
    r4 = post(client, png(), {**meta, "mode": "detailed"}).json()
    assert r4["cached"] is False and r4["mode"] == "detailed"


def test_bad_inputs(client):
    assert post(client, b"not an image").status_code == 415
    assert post(client, b"<svg xmlns='http://www.w3.org/2000/svg'></svg>").status_code == 415
    assert post(client, png(), {"mode": "poem"}).status_code == 400
    assert post(client, png(size=(4, 4))).status_code == 400
    r = client.post("/describe", files={"image": ("i.png", png(), "image/png")}, data={"metadata": "[1,2"}, headers={"X-AltContext-Token": TOKEN})
    assert r.status_code == 400
