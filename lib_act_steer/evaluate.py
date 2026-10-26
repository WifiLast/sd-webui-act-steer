"""Image evaluation through the SD MCP (~/mcp/sd_mcp, tool evaluate_image), and prompt
rephrasing through the rhyme MCP (~/mcp/rhyme_mcp, tool prompt_rephrase_tool, WordNet based).

The MCP scores an image file with learned quality models (Aesthetic V2.5, LAION aesthetic, MUSIQ,
NIMA), checks prompt adherence with the WD14 tagger (prompt tags found / missed, read from the
PNG's generation info), finds defect tags and combines them into quality_score (0-1). It runs on
the NVIDIA card in its own service, so it does not take VRAM from the webui. The call goes over
MCP's streamable HTTP transport (JSON-RPC, answers as server-sent events).

Off unless SD_ACT_STEER_MCP=1: then no request is sent and evaluation, rephrasing, ratings and the
history are skipped.
"""

import json
import os
import time

import requests
from PIL import PngImagePlugin

ENABLED = os.environ.get("SD_ACT_STEER_MCP", "").strip().lower() in ("1", "true", "yes", "on")
DISABLED = "MCP services are off (set SD_ACT_STEER_MCP=1 to use them)"
EVAL_URL = os.environ.get("SD_ACT_STEER_EVAL_URL", "http://127.0.0.1:2018/sd")
REPHRASE_URL = os.environ.get("SD_ACT_STEER_REPHRASE_URL", "http://127.0.0.1:5007/rhyme")
EVAL_DIR = os.environ.get("SD_ACT_STEER_EVAL_DIR", os.path.join("outputs", "act-steer-eval"))
TIMEOUT = float(os.environ.get("SD_ACT_STEER_EVAL_TIMEOUT", "300"))  # first call loads the models
# the SD MCP's learned quality models run in a worker of ~2 GB RAM: only asked for when this much RAM
# is free (else tags / adherence only), so background evaluations don't starve the webui
IQA_MIN_RAM = float(os.environ.get("SD_ACT_STEER_IQA_MIN_RAM_GB", "6")) * 2 ** 30
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _message(resp, want_id):
    """The JSON-RPC message with id want_id from a JSON or event-stream answer."""
    if "text/event-stream" not in resp.headers.get("content-type", ""):
        return resp.json()
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            msg = json.loads(line[5:])
            if msg.get("id") == want_id:
                return msg
    raise RuntimeError("no answer in the event stream")


def call_tool(name, arguments, timeout=TIMEOUT, url=None):
    """Call one tool of an MCP server (default the SD MCP) -> its result dict (raises on transport
    errors, and when the MCP services are off)."""
    if not ENABLED:
        raise RuntimeError(DISABLED)
    url = url or EVAL_URL
    s = requests.Session()
    r = s.post(url, headers=HEADERS, timeout=10, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                   "clientInfo": {"name": "sd-webui-act-steer", "version": "1"}}})
    r.raise_for_status()
    _message(r, 1)
    headers = dict(HEADERS)
    sid = r.headers.get("mcp-session-id")
    if sid:
        headers["mcp-session-id"] = sid
    s.post(url, headers=headers, timeout=10, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    r = s.post(url, headers=headers, timeout=timeout, json={
        "jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
    r.raise_for_status()
    msg = _message(r, 2)
    if "error" in msg:
        raise RuntimeError(msg["error"].get("message", str(msg["error"])))
    result = msg.get("result") or {}
    if result.get("structuredContent"):
        out = result["structuredContent"]
        return out.get("result", out) if isinstance(out, dict) and set(out) == {"result"} else out
    text = "".join(c.get("text", "") for c in result.get("content") or [])
    return json.loads(text) if text.strip().startswith("{") else {"ok": not result.get("isError"), "text": text}


def save_for_eval(image, infotext, tag):
    """Write image (PIL) with its generation info as PNG (the MCP reads the prompt from it)."""
    os.makedirs(EVAL_DIR, exist_ok=True)
    path = os.path.abspath(os.path.join(EVAL_DIR, f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}.png"))
    info = PngImagePlugin.PngInfo()
    if infotext:
        info.add_text("parameters", infotext)
    image.save(path, pnginfo=info)
    return path


def available_ram():
    """MemAvailable in bytes (0 when unknown)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def evaluate(image, infotext, tag="eval", iqa=None):
    """-> evaluate_image result dict (quality_score, adherence, missed_tags, defect_tags, scores,
    flags ...), or {"ok": False, "error": ...} when the MCP is not reachable or fails.
    iqa: learned quality models (None = only when IQA_MIN_RAM is free); without them the result has
    "iqa_skipped": True and its quality_score is adherence only (not comparable, see quality())."""
    if not ENABLED:
        return {"ok": False, "error": DISABLED}
    if iqa is None:
        iqa = available_ram() >= IQA_MIN_RAM
    try:
        path = save_for_eval(image, infotext, tag)
        out = call_tool("evaluate_image", {"image": path, "iqa": bool(iqa)})
        if isinstance(out, dict) and not iqa:
            out["iqa_skipped"] = True
        return out if isinstance(out, dict) else {"ok": False, "error": f"unexpected answer: {out!r:.200}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def quality(ev):
    """The comparable quality score of a result (None without the learned models or on error)."""
    if not ev or ev.get("error") or not ev.get("ok", True) or ev.get("iqa_skipped"):
        return None
    return ev.get("quality_score")


def summary_md(ev):
    """A few markdown lines from an evaluate_image result."""
    if not ev or not ev.get("ok", True) or ev.get("error"):
        return f"⚠️ Evaluation not available: {(ev or {}).get('error') or (ev or {}).get('reason', '?')}"
    lines = []
    q, adh = ev.get("quality_score"), ev.get("adherence")
    head = []
    if q is not None:
        head.append(f"quality **{q:.2f}**")
    if adh is not None:
        head.append(f"prompt adherence **{adh:.0%}**")
    sc = ev.get("scores") or {}
    head += [f"{k} {v:g}" for k, v in sc.items() if k in ("aes_v25", "musiq", "laion_aes", "nima")]
    if head:
        lines.append("**Evaluation**: " + ", ".join(head))
    if ev.get("iqa_skipped"):
        lines.append("(less than the RAM for the quality models was free: tags and adherence only, the quality "
                     "score is not comparable)")
    if ev.get("missed_tags"):
        lines.append("Not found in the image: " + ", ".join(ev["missed_tags"][:10]))
    if ev.get("defect_tags"):
        lines.append("Defects: " + ", ".join(ev["defect_tags"]))
    flags = [f for f in ev.get("flags") or [] if not f.startswith("no problems")]
    if flags:
        lines.append("Flags: " + "; ".join(flags[:4]))
    return "\n\n".join(lines) or "Evaluation: no scores returned."


def rephrase(prompt, word, other, overlap, exclude=(), vocabulary="auto", model=""):
    """Rhyme MCP suggestions for `word`, which shares `overlap` of its area with `other`, only with
    words the model understands (vocabulary auto | danbooru | natural, model = checkpoint hint):
    [{"kind": merge|place|synonym|specific|remove, "text", "prompt" (rewritten), "why"}], or
    {"error": ...} when the MCP is not reachable."""
    try:
        out = call_tool("prompt_rephrase_tool", {"prompt": prompt, "word": word, "other": other,
                                                 "overlap": round(float(overlap), 3), "exclude": list(exclude),
                                                 "vocabulary": vocabulary or "auto", "model": model or ""},
                        timeout=60, url=REPHRASE_URL)
        if not out.get("ok", True):
            return {"error": out.get("error", "?")}
        return out.get("suggestions") or []
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def adherence(ev):
    """Prompt adherence (0-1) of a result, None on error or when the tagger found no checkable tag.
    Needs only the tagger, so it is comparable with or without the learned quality models."""
    if not ev or ev.get("error") or not ev.get("ok", True):
        return None
    return ev.get("adherence")
