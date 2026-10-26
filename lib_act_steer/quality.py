"""How prompt words affect sharpness, blur, noise and detail.

In this image: local maps of the image (sharpness = mean |Laplacian|, detail = mean gradient
magnitude, noise = median absolute residual after a 3x3 median filter, on a GRID-cell grid) are
weighted with each word's attention area (its concentration map); the word's value is the ratio
to the image's median cell (noise: to the image's mean) (0.6 = its area is 40 % blurrier than a typical part of the image).

Across earlier images: the SD MCP's tag_effects fits all prompt tags together against the global
metrics of the recorded images (blur_value, noise_lv, detail_entropy), each image compared only
with images of the same checkpoint and size; effects are in standard deviations with a bootstrap
confidence. (Older SD MCPs without that tool: token_quality_report, one tag at a time, which
credits every tag of a frequent prompt with that prompt's look - used with stricter limits.)
Measured on 530 images (2026-10-24): soft lighting, shallow depth of field, gentle / golden
sunlight scenes, fluffy tail and detailed face soften the image; best quality, masterpiece,
vibrant colors and score_9 sharpen it; the wording explains ~30 % of the sharpness differences.

Optimize can use both ("Consider image quality"): a word whose area is blurry or flat is
strengthened, a word whose area is noisy is weakened, a tag that across earlier images goes with
blurrier / noisier images is weakened. When the image itself came out softer than most of the
checkpoint's images (SD MCP percentile), tags that soften are weakened even when they are smooth
by nature (soft lighting, depth of field ...) and tags that sharpen are strengthened. Words that
are smooth or grainy on purpose (sky, bokeh, film grain ...) are otherwise left alone.
"""

import threading
import time

import numpy as np
from PIL import Image

from lib_act_steer.wordmap import _resize, color_words, concentration

GRID = 32
METRICS = ("sharpness", "detail", "noise")
BLURRY = 0.7      # sharpness / detail ratio below this: the word's area is blurry / flat
NOISY = 1.4       # noise ratio above this: the word's area is noisy
BOOST, BOOST_MAX = 1.15, 1.6
WEAKEN, WEAKEN_FLOOR = 0.85, 0.5
HISTORY_R = 0.2   # |correlation| from which a tag's effect across earlier images counts ...
HISTORY_P = 0.01  # ... when this significant
MAX_CHANGES = 4
STAPLE_SHARE = 0.2  # tags in more of the recorded images are staples (1girl, masterpiece): their correlation
                    # says more about what kind of images they are in than about the tag
JOINT_MIN = 0.15   # tag_effects: |effect| in standard deviations from which a tag counts ...
JOINT_CONF = 0.95  # ... with this bootstrap sign agreement ...
JOINT_R2 = 0.05    # ... when the wording explains at least this much on held-out prompts
SOFT_PCT = 40      # the image is "soft" below this sharpness percentile among the checkpoint's images
SHARPEN, SHARPEN_MAX = 1.1, 1.3
QUALITY_TAGS = {"masterpiece", "best quality", "high quality", "amazing quality", "very aesthetic", "absurdres",
                "highres", "high resolution", "4k", "8k", "score 9", "score_9"}
SUBJECTS = {"1girl", "1boy", "2girls", "2boys", "solo", "girl", "boy", "woman", "man", "multiple girls",
            "multiple boys", "no humans", "male focus", "female focus", "masterpiece", "best quality",
            "high quality", "amazing quality", "very aesthetic", "absurdres", "highres"}
# smooth or soft by nature or on purpose: a blurry / flat area is expected
SMOOTH = {"sky", "background", "skin", "fog", "mist", "haze", "water", "cloud", "clouds", "bokeh", "blur",
          "blurry", "depth", "field", "dof", "soft", "gradient", "simple", "wall", "light", "lighting", "glow",
          "smoke", "snow", "night", "dark", "shadow", "ocean", "sea", "lake", "steam", "motion", "focus",
          "out", "abstract", "silhouette", "white", "black", "plain"}
# grainy on purpose: noise is expected
GRAINY = {"grain", "film", "noise", "noisy", "glitter", "sparkle", "sparkles", "sand", "rain", "dust", "static",
          "texture", "fur", "hair", "lace", "embroidery", "detailed", "intricate"}


def _gray(image, long_side=512):
    im = image.convert("L")
    w, h = im.size
    s = long_side / max(w, h)
    if s < 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _shift_stack(a):
    p = np.pad(a, 1, mode="edge")
    h, w = a.shape
    return np.stack([p[y:y + h, x:x + w] for y in range(3) for x in range(3)])


def _cells(a, gw, gh, stat=np.mean):
    """stat (mean / median) of `a` over a gh x gw grid of cells."""
    h, w = a.shape
    ys = np.linspace(0, h, gh + 1).astype(int)
    xs = np.linspace(0, w, gw + 1).astype(int)
    return np.array([[stat(a[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)])
                      for j in range(gw)] for i in range(gh)], dtype=np.float32)


def local_maps(image):
    """{metric: [gh, gw] grid} of the image (PIL)."""
    g = _gray(image)
    h, w = g.shape
    gw, gh = (GRID, max(1, round(GRID * h / w))) if w >= h else (max(1, round(GRID * w / h)), GRID)
    s = _shift_stack(g)  # 3x3 neighbourhoods: 0..8 row-major, 4 = centre
    lap = np.abs(s[1] + s[3] + s[5] + s[7] - 4 * s[4])
    gx = (s[2] + 2 * s[5] + s[8]) - (s[0] + 2 * s[3] + s[6])
    gy = (s[6] + 2 * s[7] + s[8]) - (s[0] + 2 * s[1] + s[2])
    resid = np.abs(g - np.median(s, axis=0))
    return {"sharpness": _cells(lap, gw, gh), "detail": _cells(np.hypot(gx, gy), gw, gh),
            "noise": _cells(resid, gw, gh, np.median)}  # median: edges are few pixels of a cell


def word_quality(image, words):
    """{word: {"sharpness": ratio, "detail": ratio, "noise": ratio, "flags": [...]}} for the
    coloured words of a word map: the image's local quality in the word's area, relative to the
    image's median cell (1 = typical)."""
    maps = local_maps(image)
    gh, gw = maps["sharpness"].shape
    # reference: the median cell; for noise the mean (half of an image can be noise-free)
    logs = {m: np.log(v + 1e-3) - np.log((np.mean(v) if m == "noise" else np.median(v)) + 1e-3)
            for m, v in maps.items()}
    out = {}
    for key, (_, _, share, coloured, raw) in color_words(words).items():
        if not coloured or raw is None:
            continue
        c = _resize(concentration(raw), (gw, gh))
        if c.sum() < 1e-3:
            continue
        wgt = c / c.sum()
        vals = {m: float(np.exp((wgt * logs[m]).sum())) for m in METRICS}
        flags = []
        if key not in SMOOTH:
            if vals["sharpness"] < BLURRY:
                flags.append("blurry")
            if vals["detail"] < BLURRY:
                flags.append("flat")
        if key not in GRAINY and vals["noise"] > NOISY:
            flags.append("noisy")
        out[key] = {**{m: round(v, 2) for m, v in vals.items()}, "flags": flags}
    return out


# ---- across earlier images (SD MCP tag_effects / token_quality_report) -------------------

_history_cache = {}
_history_busy = set()
# (target, name of the bad direction, name of the good direction, sign of "good")
_TARGETS = (("blur_value", "blurrier", "sharper", 1), ("noise_lv", "noisier", "cleaner", -1),
            ("detail_entropy", "flatter", "more detailed", 1))


def _joint_history(model, call_tool):
    """tag_effects of the SD MCP -> {tag: {bad or good name: |effect| in std, "n", "unit": "σ",
    "group": [tags always used together with it]}}, None when the MCP has no such tool."""
    out, have = {}, False
    for target, bad, good, sign in _TARGETS:
        rep = None
        for m in dict.fromkeys([model or "", ""]):  # too few images of this checkpoint: all of them
            try:
                rep = call_tool("tag_effects", {"target": target, "model": m, "min_conf": JOINT_CONF, "top": 200},
                                timeout=60)
            except Exception as e:
                if "tag_effects" in str(e) or "Unknown tool" in str(e) or "not found" in str(e).lower():
                    return None
                rep = None
            if rep and rep.get("ok", True) and rep.get("effects") is not None:
                break
        if not rep or not rep.get("ok", True) or rep.get("effects") is None:
            continue
        have = True
        if (rep.get("cv_r2") or 0) < JOINT_R2:
            continue
        for e in rep["effects"]:
            eff, tags = e.get("effect", 0.0) * sign, e.get("tags") or []
            if abs(eff) < JOINT_MIN or e.get("conf", 0) < JOINT_CONF:
                continue
            for tag in tags:
                d = out.setdefault(tag.lower(), {"n": e.get("n"), "unit": "σ"})
                d[good if eff > 0 else bad] = round(abs(eff), 2)
                if len(tags) > 1:
                    d["group"] = [t for t in tags if t.lower() != tag.lower()]
    return out if have else None


def history(model, call_tool, ttl=600):
    """{tag: {"blurrier" / "noisier" / "flatter": strength, "sharper" / "cleaner" / "more detailed":
    strength, "n": images, "unit": "σ" (tag_effects, effect in std) or "r" (correlation)}} for tags
    that across the SD MCP's recorded images of `model` go with blurrier / sharper ... images, or {}
    when there is not enough data. call_tool = evaluate.call_tool."""
    key = model or ""
    hit = _history_cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    out = _joint_history(model, call_tool)
    if out is not None:
        _history_cache[key] = (time.time(), out)
        return out
    out = {}
    for target, name, bad_sign in (("blur_value", "blurrier", -1), ("noise_lv", "noisier", 1),
                                   ("detail_entropy", "flatter", -1)):
        rep = None
        for m in dict.fromkeys([model or "", ""]):  # too few images of this checkpoint: all of them
            try:
                rep = call_tool("token_quality_report", {"target": target, "token_type": "tags", "min_count": 3,
                                                         "top": 100, "model": m}, timeout=60)
            except Exception:
                rep = None
            if rep and rep.get("ok", True):
                break
        if not rep or not rep.get("ok", True):
            continue
        for row in (rep.get("negative") or []) + (rep.get("positive") or []):
            r, p = row.get("r", 0.0), row.get("p", 1.0)
            staple = (row.get("n_with") or 0) > STAPLE_SHARE * (rep.get("images") or 1e9)
            if r * bad_sign >= HISTORY_R and p <= HISTORY_P and not staple and row["token"].lower() not in SUBJECTS:
                d = out.setdefault(row["token"].lower(), {"n": row.get("n_with"), "unit": "r"})
                d[name] = round(abs(r), 2)
    _history_cache[key] = (time.time(), out)
    return out


def history_cached(model, call_tool):
    """history() without waiting (for the view): the cached result, or None while it is fetched in
    the background (the MCP can be down: never block the UI on it)."""
    key = model or ""
    hit = _history_cache.get(key)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    if key not in _history_busy:
        _history_busy.add(key)

        def run():
            try:
                history(model, call_tool)
            finally:
                _history_busy.discard(key)
        threading.Thread(target=run, daemon=True, name="act-steer-quality-history").start()
    return hit[1] if hit else None


def _strength(h, names):
    return max((h.get(k) or 0.0) for k in names)


def _harm(h):
    """How much a tag hurt earlier images: blurrier / flatter count; noisier only when the tag did not
    also make them sharper or more detailed (the noise estimate rises with fine detail: vibrant
    colors, forest and best quality measure as both sharper and noisier)."""
    bad = max(h.get("blurrier") or 0.0, h.get("flatter") or 0.0)
    if not (h.get("sharper") or h.get("more detailed")):
        bad = max(bad, h.get("noisier") or 0.0)
    return bad


def _what(h):
    unit = h.get("unit", "r")
    return ", ".join(f"{k} ({v:.2f}{'σ' if unit == 'σ' else ' r'})" for k, v in h.items()
                     if k not in ("n", "unit", "group"))


def plan(local, hist, prompt_phrases, current, locked=(), skip=(), sharpness_pct=None):
    """Optimize changes from image quality -> [(word or tag, factor, reason)], at most MAX_CHANGES.
    local: word_quality(); hist: history(); prompt_phrases: lower-case comma phrases of the prompt;
    current: {word: factor the image was made with}; skip: words other changes already touch;
    sharpness_pct: the image's sharpness percentile among the checkpoint's images (SD MCP
    evaluate_image, 100 = sharpest; None = unknown) - below SOFT_PCT softening tags are weakened
    even when smooth by nature, and sharpening tags are strengthened."""
    soft = sharpness_pct is not None and sharpness_pct < SOFT_PCT
    locked = {w.strip().lower() for w in locked if w.strip()}
    skip = {w.lower() for w in skip} | locked
    out = []
    for word, q in sorted(local.items(), key=lambda kv: min(kv[1]["sharpness"], kv[1]["detail"])):
        if word in skip or not q["flags"]:
            continue
        cur = current.get(word, 1.0)
        if "noisy" in q["flags"] and cur > WEAKEN_FLOOR + 1e-6:
            out.append((word, round(max(WEAKEN_FLOOR, cur * WEAKEN), 2),
                        f"its area is noisier than the image ({q['noise']:.2f}x noise)"))
        elif ("blurry" in q["flags"] or "flat" in q["flags"]) and cur < BOOST_MAX - 1e-6:
            out.append((word, round(min(BOOST_MAX, max(cur, 1.0) * BOOST), 2),
                        f"its area is blurrier / flatter than the image ({q['sharpness']:.2f}x sharpness, "
                        f"{q['detail']:.2f}x detail): more attention for more definition"))
    hist_changes = []
    for tag, h in hist.items():
        words = set(tag.split())
        if tag not in prompt_phrases or tag in skip or words & skip or tag in [o[0] for o in out]:
            continue
        cur = current.get(tag, 1.0)
        bad, joint = _harm(h), h.get("unit") == "σ"
        by_nature = bool(words & (SMOOTH | GRAINY))
        where = f"across {h.get('n') or 'earlier'} images with it, they come out {_what(h)}"
        if bad > 0 and tag not in SUBJECTS and cur > WEAKEN_FLOOR + 1e-6:
            # smooth-by-nature tags (soft lighting, depth of field) soften on purpose: only when this
            # image did come out soft, and only from the joint analysis
            blur_only = h.get("blurrier") and not (h.get("noisier") or h.get("flatter"))
            if not by_nature or (joint and soft and blur_only):
                why = where + (f"; this image is softer than {100 - sharpness_pct}% of the checkpoint's images"
                               if soft and h.get("blurrier") else "")
                hist_changes.append((bad, (tag, round(max(WEAKEN_FLOOR, cur * WEAKEN), 2), why)))
        elif joint and soft and h.get("sharper") and not h.get("flatter") and (tag not in SUBJECTS or tag in QUALITY_TAGS) \
                and cur < SHARPEN_MAX - 1e-6:
            hist_changes.append((h["sharper"], (tag, round(min(SHARPEN_MAX, max(cur, 1.0) * SHARPEN), 2),
                                 f"{where}; this image is softer than {100 - sharpness_pct}% of the checkpoint's "
                                 "images: more of it")))
    out += [c for _, c in sorted(hist_changes, key=lambda x: -x[0])]
    return out[:MAX_CHANGES]


def sharpening_suggestions(hist, prompt_phrases, n=4):
    """Tags that sharpened earlier images (joint analysis) and are not in this prompt -> [(tag,
    effect)], strongest first: shown as a hint only (adding words changes the picture)."""
    rows = [(t, h["sharper"]) for t, h in hist.items()
            if h.get("unit") == "σ" and h.get("sharper") and not h.get("group") and t not in prompt_phrases
            and t not in SUBJECTS - QUALITY_TAGS and not _harm(h)]
    return sorted(rows, key=lambda r: -r[1])[:n]


def history_html(hist, prompt_phrases, sharpness_pct=None):
    """Collapsed list: what the prompt's tags did to the sharpness / noise / detail of earlier images."""
    if not hist:
        return ""
    rows = []
    for tag in prompt_phrases:
        h = hist.get(tag)
        if not h:
            continue
        good = _strength(h, [t[2] for t in _TARGETS])
        bad = _harm(h)
        color = "rgba(220,60,60,.35)" if bad > good else "rgba(60,180,90,.3)"
        grp = f" <small>(always with {', '.join(h['group'][:3])})</small>" if h.get("group") else ""
        rows.append((max(good, bad), f'<tr><th>{tag}{grp}</th><td style="background:{color}">{_what(h)}</td>'
                                     f'<td>{h.get("n") or ""}</td></tr>'))
    sugg = sharpening_suggestions(hist, set(prompt_phrases))
    if not rows and not sugg:
        return ""
    joint = any(h.get("unit") == "σ" for h in hist.values())
    state = ""
    if sharpness_pct is not None:
        state = (f" - this image: sharper than {sharpness_pct}% of the checkpoint's images"
                 + (" (soft)" if sharpness_pct < SOFT_PCT else ""))
    summary = (f"{len(rows)} prompt tag{'s' if len(rows) != 1 else ''} changed sharpness / noise / detail in earlier "
               f"images{state}")
    body = "".join(r for _, r in sorted(rows, key=lambda r: -r[0]))
    hint = ("<p class=\"wm-p\" style=\"font-size:12px\">Sharpened earlier images, not in this prompt: "
            + ", ".join(f"<b>{t}</b> (+{v:.2f}σ)" for t, v in sugg) + "</p>") if sugg else ""
    how = ("all tags fitted together, each image compared with images of the same checkpoint and size; "
           "σ = standard deviations" if joint else
           "one tag at a time (r = correlation; update the SD MCP for the joint analysis)")
    return (f'<details class="wm-tab wm-check"><summary>Prompt tags vs sharpness: {summary}</summary>'
            f'<table><tr><th></th><th>in earlier images</th><th>images</th></tr>{body}</table>{hint}'
            f'<p class="wm-p" style="font-size:12px">From the SD MCP\'s recorded images ({how}). Red: '
            'softer / noisier / flatter. With "Consider image quality", Optimize weakens red tags and, when '
            f'this image is softer than {100 - SOFT_PCT}% of the checkpoint\'s images, strengthens green ones.</p>'
            '</details>')


def table_html(local):
    """Collapsed table: the image's quality in each word's area."""
    if not local:
        return ""
    def cell(v, bad_low=True):
        bad = v < BLURRY if bad_low else v > NOISY
        return f'<td style="background:{"rgba(220,60,60,.35)" if bad else "transparent"}">{v:.2f}</td>'
    rows = "".join(f"<tr><th>{w}</th>{cell(q['sharpness'])}{cell(q['detail'])}{cell(q['noise'], False)}"
                   f"<td>{', '.join(q['flags']) or ''}</td></tr>"
                   for w, q in sorted(local.items(), key=lambda kv: kv[1]["sharpness"]))
    flagged = [w for w, q in local.items() if q["flags"]]
    summary = (f"{len(flagged)} word{'s' if len(flagged) != 1 else ''} with a blurry / flat / noisy area: "
               + ", ".join(flagged[:6])) if flagged else "no word's area stands out"
    return (f'<details class="wm-tab wm-check"><summary>Image quality by word: {summary}</summary>'
            '<table><tr><th></th><th>sharpness</th><th>detail</th><th>noise</th><th></th></tr>'
            f'{rows}</table><p class="wm-p" style="font-size:12px">The image\'s sharpness, detail and noise in '
            'each word\'s area relative to a typical part of the image (1 = typical; red: blurrier / flatter '
            'than 0.7x or noisier than 1.4x). Smooth-by-nature words (sky, skin, bokeh ...) are not flagged '
            'for blur, grainy ones (film grain, fur ...) not for noise.</p></details>')
