import hashlib
import hmac
import json
import os
import threading
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

ROOT_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT_DIR / "public"

load_dotenv(ROOT_DIR / ".env")

# Vercel's Python runtime looks for a top-level Flask instance named `app` in this
# file. Static assets (public/index.html) are served directly by Vercel's CDN in
# production — this app only ever needs to handle /api/* there. The "/" route below
# exists for local development (`python3 server.py`), which has no CDN layer.
app = Flask(__name__)
client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

MODEL = "claude-opus-4-8"

# ---------- Storage ----------
# Local dev: a plain JSON file next to this script.
# Deployed on Vercel: Upstash Redis (via the Vercel Marketplace integration), since
# serverless functions have no persistent local disk between invocations. Whichever
# is available is chosen automatically based on whether the Upstash env vars exist.
STATE_PATH = ROOT_DIR / "state.json"
STATE_KEY = "outfit:state"
_state_lock = threading.Lock()

USE_REDIS = bool(os.environ.get("UPSTASH_REDIS_REST_URL"))
if USE_REDIS:
    from upstash_redis import Redis
    redis = Redis.from_env()

DEFAULT_STATE = {
    "items": [],
    "week": {},
    "events": {},
    "location": None,
    "weatherCache": None,
    "tagCount": 0,
    "dismissedSuggestions": [],
}


def load_state():
    if USE_REDIS:
        raw = redis.get(STATE_KEY)
        data = json.loads(raw) if raw else {}
    else:
        if not STATE_PATH.exists():
            data = {}
        else:
            with open(STATE_PATH) as f:
                data = json.load(f)
    return {**DEFAULT_STATE, **data}


def save_state(state):
    if USE_REDIS:
        redis.set(STATE_KEY, json.dumps(state))
    else:
        with open(STATE_PATH, "w") as f:
            json.dump(state, f)


# ---------- Auth ----------
# A single shared password, set via the APP_PASSWORD env var. There's no per-user
# account system here — this only needs to keep random visitors off the app once
# it's reachable from the whole internet (not just home WiFi), since /api/tag-item
# spends real API budget on every call. If APP_PASSWORD is unset, the gate is a
# no-op — that's the local-dev default, but set it before deploying publicly.
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")


def expected_token():
    return hashlib.sha256(APP_PASSWORD.encode()).hexdigest()


@app.before_request
def require_auth():
    if not APP_PASSWORD:
        return None
    if request.path == "/api/login" or not request.path.startswith("/api/"):
        return None
    auth = request.headers.get("Authorization", "")
    token = auth[len("Bearer "):] if auth.startswith("Bearer ") else ""
    if not token or not hmac.compare_digest(token, expected_token()):
        return jsonify({"error": "unauthorized"}), 401
    return None


@app.route("/api/login", methods=["POST"])
def login():
    if not APP_PASSWORD:
        return jsonify({"error": "no password configured on the server"}), 500
    data = request.get_json(force=True) or {}
    if not hmac.compare_digest(str(data.get("password", "")), APP_PASSWORD):
        return jsonify({"error": "wrong password"}), 401
    return jsonify({"token": expected_token()})


TAG_TOOL = {
    "name": "record_clothing_item",
    "description": "Record structured tags for a single clothing item photo.",
    "input_schema": {
        "type": "object",
        "properties": {
            "brand": {
                "type": "string",
                "description": (
                    "Brand name if a visible logo, tag, or label identifies it, e.g. 'Everlane' or 'J.Crew'. "
                    "Use an empty string if no brand is visible or identifiable — never guess."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "Short human-readable name that leads with the brand (if known) and a descriptive style, "
                    "e.g. 'Everlane Cotton Poplin Shirt' or 'J.Crew Slim Chino'. "
                    "If no brand is identifiable, still include a descriptive style, e.g. 'Slim Navy Chinos'."
                ),
            },
            "category": {
                "type": "string",
                "enum": ["top", "bottom", "dress", "layer", "shoes", "accessory"],
                "description": "layer = jacket/blazer/cardigan worn over a top.",
            },
            "color": {
                "type": "string",
                "description": "Descriptive primary color, lowercase, e.g. 'burnt orange', 'charcoal', 'burgundy'.",
            },
            "colorFamily": {
                "type": "string",
                "enum": [
                    "black", "white", "gray", "beige", "brown", "navy",
                    "red", "orange", "yellow", "green", "blue", "purple", "pink",
                ],
                "description": (
                    "Bucket the item's dominant color into the closest family for outfit-matching purposes, "
                    "even if the exact shade differs — e.g. 'burgundy'/'maroon' -> 'red', 'olive'/'sage' -> 'green', "
                    "'tan'/'khaki' -> 'beige', 'cream'/'ivory' -> 'white', 'charcoal' -> 'gray', 'teal' -> 'blue'."
                ),
            },
            "patterned": {
                "type": "boolean",
                "description": "True if the item has a visible pattern or print (stripes, plaid, floral, etc.); false if solid.",
            },
            "formality": {
                "type": "string",
                "enum": ["casual", "business_casual", "business", "formal"],
            },
        },
        "required": ["brand", "name", "category", "color", "colorFamily", "patterned", "formality"],
        "additionalProperties": False,
    },
}


@app.after_request
def disable_caching(response):
    # This is a single actively-edited HTML file — never let the browser (especially
    # mobile Safari, which caches aggressively) serve a stale copy after an update.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
def index():
    # Local-dev only — Vercel serves public/index.html directly via its CDN in
    # production, without ever invoking this function.
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.route("/api/state", methods=["GET"])
def get_state():
    with _state_lock:
        return jsonify(load_state())


@app.route("/api/state", methods=["PUT"])
def put_state():
    incoming = request.get_json(force=True) or {}
    incoming.pop("tagCount", None)  # server-owned counter — never trust a client-supplied value
    with _state_lock:
        state = load_state()
        state.update(incoming)
        save_state(state)
        return jsonify(state)


@app.route("/api/tag-item", methods=["POST"])
def tag_item():
    data = request.get_json(force=True)
    image_b64 = data.get("imageBase64")
    media_type = data.get("mediaType", "image/jpeg")

    if not image_b64:
        return jsonify({"error": "imageBase64 is required"}), 400

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        tools=[TAG_TOOL],
        tool_choice={"type": "tool", "name": "record_clothing_item"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "This is a photo of one work-wardrobe clothing item. "
                            "Check for a visible brand logo, woven label, or tag, and note whether it's "
                            "solid or patterned. Identify the item and record its tags."
                        ),
                    },
                ],
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use":
            with _state_lock:
                state = load_state()
                state["tagCount"] = state.get("tagCount", 0) + 1
                save_state(state)
                tag_count = state["tagCount"]
            return jsonify({**block.input, "tagCount": tag_count})

    return jsonify({"error": "model did not return tags"}), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5177))
    app.run(host="0.0.0.0", port=port)
