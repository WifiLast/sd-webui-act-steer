"""Word map: per prompt word, where in the image it was attended to, as a clickable HTML view.

The maps are accumulated during generation by lib_act_steer.core (state.word_maps). The view is
plain HTML/CSS: every area tinted in the colour of its word, hidden radio buttons select one word,
hover previews one; javascript/act_steer.js reports clicks (data-word) to the UI for editing.
"""

import base64
import html
import io
import json
import re
import uuid

import numpy as np
import torch.nn.functional as F
from PIL import Image

from lib_act_steer import core
from lib_act_steer.core import state

# !important on layout: the webui's .prose rules for img/input are more specific than plain classes
STYLE = """<style>
.wm-box{position:relative !important;display:inline-block !important;max-width:100%;margin:0 !important}
.wm-box img{margin:0 !important;max-width:100% !important;border-radius:6px}
.wm-box>img.wm-base{display:block !important;width:100% !important;height:auto !important;position:static !important}
.wm-box img.wm-all,.wm-box img.wm-sel,.wm-box img.wm-ov{position:absolute !important;left:0 !important;
  top:0 !important;width:100% !important;height:auto !important;pointer-events:none}
.wm-box img.wm-sel,.wm-box img.wm-ov,input.wm-r{display:none !important}
.wm-rw:checked~.wm-box img.wm-all{display:none !important}
.wm-box:has(.wm-w:hover) img.wm-all,.wm-box:has(.wm-w:hover) img.wm-sel{display:none !important}
.wm-words{line-height:2.3;margin-top:8px;font-size:15px}
.wm-w,.wm-c{cursor:pointer;padding:2px 6px;border-radius:5px;background:var(--wm-c);color:var(--wm-t);margin:0 1px}
.wm-c{background:transparent;color:var(--body-text-color);border:1px solid var(--border-color-primary)}
.wm-w:hover{outline:2px solid var(--body-text-color)}
.wm-box .wm-w:hover img.wm-ov{display:block !important}
.wm-g{background:transparent;color:var(--body-text-color-subdued);border:1px dashed var(--border-color-primary)}
.wm-p{color:var(--body-text-color-subdued)}
.wm-e{font-weight:bold;margin-left:2px}
.wm-rel{font-size:11px;margin-left:3px;padding:0 3px;border-radius:3px;background:rgba(0,0,0,.45);color:#fff}
.wm-check{margin-top:10px;font-size:14px}
.wm-check summary{cursor:pointer}
.wm-check ul{margin:4px 0 0 18px !important;padding:0 !important}
.wm-check li{margin:3px 0 !important}
.wm-tab table{border-collapse:collapse;font-size:12px;margin-top:6px}
.wm-tab td,.wm-tab th{padding:2px 5px !important;text-align:center;border:1px solid var(--border-color-primary)}
.wm-box>img.wm-base{cursor:crosshair}
.wm-x{text-decoration:line-through;opacity:.45}
.wm-w.wm-hit{outline:3px solid var(--body-text-color);outline-offset:1px;box-shadow:0 0 0 calc(var(--wm-h) * 6px) var(--wm-c)}
.wm-box.wm-spotted .wm-w:not(.wm-hit){opacity:.35}
.wm-dot{position:absolute;width:18px;height:18px;margin:-9px 0 0 -9px;border:2px solid #fff;border-radius:50%;
  box-shadow:0 0 0 2px #000;pointer-events:none;display:none}
.wm-spot{min-height:1.4em;font-size:13px;margin-top:4px}
</style>"""

# filler words: still hoverable, but not coloured into the all-words view
STOPWORDS = {"a", "an", "the", "on", "in", "of", "at", "with", "and", "or", "no", "to", "is", "by", "for",
             "from", "very", "some"}
GREY = (150, 150, 150)


# clearly distinguishable on photos and illustrations (from Sasha Trubetskoy's 20 distinct colours)
PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
           "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3"]


def palette(n):
    return [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in (PALETTE * (n // len(PALETTE) + 1))[:n]]


def _css(rgb):
    return "#%02x%02x%02x" % rgb


def _text_color(rgb):
    return "#000" if 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 140 else "#fff"


def _png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _jpeg(img, size=768):
    img = img.convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def overlay_png(m, rgb=(255, 40, 40)):
    """Spotlight: the word's area tinted in its colour, the rest darkened."""
    m = np.asarray(m, dtype=np.float32)
    rgba = np.zeros(m.shape + (4,), dtype=np.uint8)
    for c in range(3):
        rgba[..., c] = (rgb[c] * m).astype(np.uint8)
    rgba[..., 3] = ((0.65 * (1 - m) + 0.55 * m) * 255).astype(np.uint8)
    return _png(Image.fromarray(rgba, "RGBA"))


def _resize(m, size):
    return np.asarray(Image.fromarray(np.asarray(m, dtype=np.float32), mode="F").resize(size, Image.BILINEAR))


def composite(raws, colors, size, lift_low=1.4, lift_full=2.6, margin=0.4):
    """Every area coloured like the word that is most concentrated there: each word's attention is
    divided by its own average over the image (lift), so broad words like "girl" or "scenery"
    don't take everything; a pixel is tinted if the best lift is high and ahead of the runner-up.

    Returns (all words RGBA, [one RGBA per word: only the areas that word won]).
    """
    stack = np.stack([_resize(r, size) for r in raws])          # [G, H, W]
    lift = stack / (stack.mean(axis=(1, 2), keepdims=True) + 1e-8)
    order = np.sort(lift, axis=0)
    val = order[-1]
    second = order[-2] if len(raws) > 1 else np.ones_like(val)
    idx = lift.argmax(0)
    alpha = (np.clip((val - lift_low) / (lift_full - lift_low), 0, 1)
             * np.clip((val - second) / margin, 0, 1) * 0.6)
    cols = np.asarray(colors, dtype=np.float32)[idx]             # [H, W, 3]
    every = np.dstack([cols, alpha[..., None] * 255]).astype(np.uint8)
    single = [np.dstack([np.broadcast_to(np.asarray(c, np.float32), cols.shape),
                         (alpha * (idx == g))[..., None] * 255]).astype(np.uint8)
              for g, c in enumerate(colors)]
    return every, single


def tint(m, rgb, size):
    """A word's own map in its colour, nothing darkened (for words that win no area in the composite)."""
    m = np.clip(_resize(m, size), 0, 1)
    return np.dstack([np.broadcast_to(np.asarray(rgb, np.float32), m.shape + (3,)),
                      (m * 0.6)[..., None] * 255]).astype(np.uint8)


def color_words(words):
    """-> {word key: (rgb, merged map, share, coloured)}; same text = same colour, maps merged by max.

    Filler words and attention sinks (a word with > 30 % of all attention) stay grey.
    """
    groups = {}
    for n, (text, m, share, raw) in enumerate(words):
        if m is None:
            continue
        key = text.lower()
        g = groups.setdefault(key, {"map": m, "share": 0.0, "first": n, "raw": raw})
        g["map"] = np.maximum(g["map"], m)
        g["raw"] = g["raw"] + raw if g["raw"] is not raw else raw
        g["share"] += share
    # attention sinks: the first token, and any word soaking up a large part of all attention
    # (seen: "full" at the end of a prompt with 65 %)
    coloured = [k for k, g in groups.items()
                if k not in STOPWORDS and not (g["share"] > 0.3 and len(groups) > 3)
                and not (g["first"] == 0 and g["share"] > 0.1 and len(groups) > 1)]
    colors = dict(zip(coloured, palette(len(coloured))))
    return {k: (colors.get(k, GREY), g["map"], g["share"], k in colors, g["raw"]) for k, g in groups.items()}


def strength_filter(groups, show):
    """Keys of the words to show for a strength filter: show = None (all), ("top", n) = the n words with
    the largest share of the attention, ("below", fraction) = the words with less than that share.
    Filler words (a, the, of ...) are only listed without a filter."""
    if not show:
        return list(groups)
    cand = [k for k in groups if k not in STOPWORDS]
    kind, val = show
    if kind == "top":
        return sorted(cand, key=lambda k: -groups[k][2])[:max(1, int(val))]
    return [k for k in cand if groups[k][2] < float(val)]


def no_effect_words(words, keep=()):
    """Words with little visible effect, as the prompt check means it: attention share below WEAK_SHARE and
    not concentrated anywhere (peak lift below WEAK_LIFT); filler words and `keep` left out."""
    groups = color_words(words)
    keep = {k.lower() for k in keep}
    out = []
    for k, (_, _, share, coloured, raw) in groups.items():
        if coloured and k not in keep and share < WEAK_SHARE and float(np.max(raw) / (np.mean(raw) + 1e-8)) < WEAK_LIFT:
            out.append(k)
    return out


def weak_words(words, threshold, keep=()):
    """Prompt words whose share of the attention is below threshold (fraction, 0.005 = 0.5 %), weakest
    first; filler words and `keep` (locked / kept by rules) left out."""
    groups = color_words(words)
    keep = {k.lower() for k in keep}
    return sorted((k for k in strength_filter(groups, ("below", threshold)) if k not in keep),
                  key=lambda k: groups[k][2])


GRID = 32  # long side of the per-word lift grid embedded for clicks on the image (javascript/act_steer.js)


def _grid_json(groups, keys, w, h):
    """{"w", "h", "words": {key: base64 uint8 grid}}: each word's lift (attention / its image mean)
    per cell, x 255/4 (lift 4 and more = 255)."""
    gw, gh = (GRID, max(1, round(GRID * h / w))) if w >= h else (max(1, round(GRID * w / h)), GRID)
    out = {}
    for k in keys:
        raw = groups[k][4]
        if raw is None:
            continue
        lift = _resize(raw, (gw, gh)) / (float(np.mean(raw)) + 1e-8)
        out[k] = base64.b64encode(np.clip(lift * 255 / 4, 0, 255).astype(np.uint8).tobytes()).decode()
    return json.dumps({"w": gw, "h": gh, "words": out})


def render_html(image, words, edits=None, selected=None, lora=None, removed=(), rules=None, show=None):
    """image: PIL; words: [(text, map [H, W] 0..1 or None, share, raw map)] -> HTML.

    edits: {word: influence factor} shown as a badge on the word; selected: word whose radio
    starts checked (keeps the selection when the view is re-rendered after an edit); lora: block
    maps from harvest_blocks (adds the LoRA block section); removed: words taken out of the prompt
    (struck through); show: strength filter (strength_filter: ("top", n) / ("below", fraction)), only
    those words are listed and coloured, each with its share. A click on the image highlights the words acting there, a right click opens a
    menu (javascript/act_steer.js, using the lift grid in data-grid).
    Shows all areas coloured like their words; clicking a word shows only that word's colour,
    "all words" / "none" switch back; hovering previews a word as a spotlight. Pure HTML/CSS:
    hidden radio buttons, one per word, select which overlay is visible.
    """
    w, h = image.size
    uid = uuid.uuid4().hex[:8]
    groups = color_words(words)
    keys = list(groups)
    num = {k: i for i, k in enumerate(keys)}
    size = (384, max(1, round(384 * h / w))) if w >= h else (max(1, round(384 * w / h)), 384)
    visible = set(strength_filter(groups, show))
    shown = [k for k in keys if groups[k][3] and k in visible]
    every, single = composite([groups[k][4] for k in shown], [groups[k][0] for k in shown], size) \
        if shown else (None, [])
    sel = {}
    for k in keys:
        if k not in visible:  # filtered out: no overlay images (they are most of the page's size)
            continue
        rgba = single[shown.index(k)] if k in shown else None
        if rgba is None or rgba[..., 3].max() < 25:  # wins no area: show its own map instead
            rgba = tint(groups[k][1], groups[k][0], size)
        sel[k] = _png(Image.fromarray(rgba, "RGBA"))
    overlays = {k: overlay_png(g[1], g[0]) for k, g in groups.items() if k in visible}

    rid = lambda n: f"wm-{uid}-{n}"  # noqa: E731
    edits = {k.lower(): v for k, v in (edits or {}).items()}
    sel_key = (selected or "").lower() if (selected or "").lower() in num else None
    checked = lambda on: " checked" if on else ""  # noqa: E731
    radios = [f'<input type="radio" class="wm-r" name="wm-{uid}" id="{rid("all")}"{checked(sel_key is None)}>',
              f'<input type="radio" class="wm-r wm-rw" name="wm-{uid}" id="{rid("none")}">']
    radios += [f'<input type="radio" class="wm-r wm-rw" name="wm-{uid}" id="{rid(num[k])}"{checked(k == sel_key)}>'
               for k in keys]
    css = [f'#{rid("all")}:checked~.wm-box .wm-l-all,#{rid("none")}:checked~.wm-box .wm-l-none'
           '{outline:3px solid var(--body-text-color);outline-offset:1px}']
    for k in keys:
        n = num[k]
        css.append(f'#{rid(n)}:checked~.wm-box img.wm-s-{n}{{display:block !important}}'
                   f'#{rid(n)}:checked~.wm-box .wm-l-{n}{{outline:3px solid var(--body-text-color);'
                   'outline-offset:1px}')
    style = STYLE.replace("</style>", "\n".join(css) + "\n</style>")

    rel = dict(related_to(words, sel_key, n=99)) if sel_key else {}
    # per word: the words it shares most area with (right-click menu: allow overlap / prefer)
    rk, rov, _ = relations(words)
    near = {}
    for i, k in enumerate(rk):
        pairs = sorted(((float(rov[i, j]), o) for j, o in enumerate(rk) if j != i), reverse=True)
        near[k] = ",".join(f"{o}:{v:.2f}" for v, o in pairs[:4] if v >= 0.15 and not (rules and rules.allowed(k, o)))
    parts = [f'<label class="wm-c wm-l-all" for="{rid("all")}">all words</label>',
             f'<label class="wm-c wm-l-none" for="{rid("none")}">none</label> &nbsp;']
    if show:
        kind, val = show
        what = (f"the {len(visible)} strongest of {len(keys)} words" if kind == "top" else
                f"{len(visible)} of {len(keys)} words with less than {float(val) * 100:g}% of the attention")
        parts.append(f'<span class="wm-p" style="font-size:12px">Showing {what}:</span>')
    listed, seq = set(), words
    if show:  # strongest first
        first = {}
        for wd in words:
            if wd[1] is not None and wd[0].lower() in visible:
                first.setdefault(wd[0].lower(), wd)
        seq = sorted(first.values(), key=lambda wd: -groups[wd[0].lower()][2])
    for text, m, _, _ in seq:
        if m is None:
            if not show:
                parts.append(f'<span class="wm-p">{html.escape(text)}</span>')
            continue
        key = text.lower()
        if show and (key not in visible or key in listed):
            continue
        listed.add(key)
        rgb, _, share, coloured, _ = groups[key]
        n = num[key]
        cls = ("wm-w" if coloured else "wm-w wm-g") + (" wm-x" if key in removed else "")
        badge = f'<sup class="wm-e">×{edits[key]:g}</sup>' if key in edits else ""
        if show:
            badge += f'<sub class="wm-rel">{share * 100:.{1 if share >= 0.01 else 2}f}%</sub>'
        if key in rel and rel[key] >= 0.1:
            badge += f'<sub class="wm-rel" title="shares {rel[key]:.0%} of its area with {html.escape(sel_key)}">' \
                     f'{rel[key]:.2f}</sub>'
        parts.append(
            f'<label class="{cls} wm-l-{n}" for="{rid(n)}" data-word="{html.escape(text, quote=True)}" '
            f'data-key="{html.escape(key, quote=True)}" data-rel="{html.escape(near.get(key, ""), quote=True)}" '
            f'style="--wm-c:{_css(rgb)};--wm-t:{_text_color(rgb)}" '
            f'title="{html.escape(text)}: {share * 100:.1f}% of the attention to prompt words">'
            f'{html.escape(text)}{badge}<img class="wm-ov" style="aspect-ratio:{w}/{h}" src="{overlays[key]}">'
            '</label>')
    layers = ""
    if every is not None:
        layers += f'<img class="wm-all" style="aspect-ratio:{w}/{h}" src="{_png(Image.fromarray(every, "RGBA"))}">'
    layers += "".join(f'<img class="wm-sel wm-s-{num[k]}" style="aspect-ratio:{w}/{h}" src="{sel[k]}">'
                      for k in keys if k in sel)
    return (style + "".join(radios)
            + f'<div class="wm-box" style="width:min(100%,{min(768, w) if w >= h else round(768 * w / h)}px)" '
            f'data-grid="{html.escape(_grid_json(groups, keys, w, h), quote=True)}">'
            f'<img class="wm-base" src="{_jpeg(image)}">{layers}<div class="wm-dot"></div>'
            '<div class="wm-spot wm-p">Click the image: the words acting there light up. Right click a word or '
            'the image for a menu.</div>'
            f'<div class="wm-words">{" ".join(parts)}</div>{_check_html(words, rules)}{_lora_html(words, lora, rules)}{_table_html(words)}</div>'
            '<p class="wm-p" style="font-size:12px">Each area has the colour of the word that influenced it '
            'most. Click a word to show only its colour and to edit its influence below; "all words" / '
            '"none" switch back, hovering previews a word. Grey words (filler words, the first token) are '
            'not in the all-words colouring but can be clicked. Right click a word or an area: set its '
            'influence, lock it, or remove it from the prompt.</p>')


def harvest(img_local, negative=False, early=False):
    """Accumulated maps of one image of the current batch (negative: of its negative prompt; early: at
    core.EARLY_FRAC of the steps, [] when not taken) -> [(text, map 0..1 or None, share, raw attention
    map or None)]."""
    all_texts = state.neg_word_texts if negative else state.word_texts
    texts = all_texts[img_local % len(all_texts)] if all_texts else []
    per = (state.neg_word_maps if negative else state.early_word_maps if early else state.word_maps).get(img_local, {})
    if early and not per:
        return []
    mass = {key: float(acc.sum()) for key, acc in per.items()}
    total = sum(mass.values()) or 1.0
    out = []
    for key, text in texts:
        acc = per.get(key) if key is not None else None
        if acc is None:
            out.append((text, None, 0.0, None))
        else:
            raw = F.avg_pool2d(acc[None, None].float(), 3, stride=1, padding=1, count_include_pad=False)[0, 0]
            out.append((text, core.soft_map(acc).cpu().numpy(), mass[key] / total, raw.cpu().numpy()))
    return out


def _block_order(blk):
    m = re.match(r"(IN|M|OUT)(\d+)", blk)
    return ("IN", "M", "OUT").index(m.group(1)), int(m.group(2))


def harvest_blocks(img_local):
    """LoRA block maps of one image of the current batch -> None or
    {"loras": [names], "blocks": [block ids], "maps": {(LoRA name or "", block): {word: raw map}}}."""
    per = state.block_maps.get(img_local)
    if not per:
        return None
    texts = state.word_texts[img_local % len(state.word_texts)] if state.word_texts else []
    word_of = {key: text.lower() for key, text in texts if key is not None}
    maps = {}
    for nb, accs in per.items():
        d = maps.setdefault(nb, {})
        for key, acc in accs.items():
            w = word_of.get(key)
            if w is None:
                continue
            raw = F.avg_pool2d(acc[None, None].float(), 3, stride=1, padding=1, count_include_pad=False)[0, 0]
            raw = raw.cpu().numpy()
            d[w] = d[w] + raw if w in d else raw
    loras = sorted({n for n, _ in maps if n})
    if not loras:
        return None
    return {"loras": loras, "blocks": sorted({b for _, b in maps}, key=_block_order), "maps": maps,
            "weights": dict(state.lora_weights)}


# ---- relations between prompt words -----------------------------------------------------

# thresholds for the prompt check (overlap of concentration areas, 0..1). No adjective/noun
# binding check: colour words concentrate on the same places as each other (0.7-0.8 overlap)
# even when every colour lands on the right object, so their maps say nothing about binding.
SAME_AREA_MIN = 0.33  # two nouns of different phrases cover the same area (ocean/sea 0.35)
WEAK_SHARE = 0.012    # below this share of the attention ...
WEAK_LIFT = 2.0       # ... and this peak concentration: little visible effect


def concentration(raw):
    lift = np.asarray(raw, dtype=np.float32) / (float(np.mean(raw)) + 1e-8)
    return np.clip((lift - 1.2) / 1.3, 0, 1)


def relations(words):
    """-> (keys, overlap [G, G], groups) for the coloured (content) words.

    overlap = soft Jaccard (sum(min) / sum(max)) of the areas where each word is concentrated
    (attention 1.2x .. 2.5x its own image average -> 0 .. 1), 1 = same area, 0 = disjoint.
    Plain maps made spread-out words (all colour words) look related to everything.
    """
    groups = color_words(words)
    keys = [k for k, g in groups.items() if g[3]]
    if not keys:
        return keys, np.zeros((0, 0)), groups
    M = np.stack([concentration(groups[k][4]) for k in keys]).reshape(len(keys), -1)
    inter = np.minimum(M[:, None], M[None]).sum(-1)
    union = np.maximum(M[:, None], M[None]).sum(-1) + 1e-8
    return keys, inter / union, groups


def _overlap(raw_a, raw_b):
    a, b = concentration(raw_a).ravel(), concentration(raw_b).ravel()
    return float(np.minimum(a, b).sum() / (np.maximum(a, b).sum() + 1e-8))


def same_area_pairs(words, min_overlap=None, rules=None):
    """Nouns of different phrases acting on the same area (overlap >= min_overlap, default
    SAME_AREA_MIN; pairs a rule allows left out): [(a, b, overlap)], biggest first."""
    min_overlap = SAME_AREA_MIN if min_overlap is None else min_overlap
    keys, ov, groups = relations(words)
    if len(keys) < 2:
        return []
    word_runs = runs(words, groups)
    modifiers = {k for r in word_runs for k in r[:-1]}
    run_of = {k: i for i, r in enumerate(word_runs) for k in r}
    nouns = [k for k in keys if k not in modifiers]
    out = []
    for i, a in enumerate(nouns):
        for b in nouns[i + 1:]:
            if a in run_of and run_of.get(a) == run_of.get(b):
                continue
            o = float(ov[keys.index(a), keys.index(b)])
            if o >= min_overlap and not (rules and rules.allowed(a, b)):
                out.append((a, b, o))
    return sorted(out, key=lambda t: -t[2])


def _changeable(locked=(), allowed=None):
    locked = {w.strip().lower() for w in locked if w.strip()}
    return lambda k: k not in locked and (allowed is None or k in allowed)


def max_conflict(words, locked=(), allowed=None, rules=None):
    """The worst conflict: (overlap, a, b) of the noun pair sharing the most area; pairs where
    neither word can change (locked, or not in `allowed`, e.g. added by a prompt extension) are
    left out; (0, None, None) when there is none."""
    can = _changeable(list(locked) + (rules.kept() if rules else []), allowed)
    for a, b, o in same_area_pairs(words, 0.0, rules):
        if can(a) or can(b):
            return o, a, b
    return 0.0, None, None


# LoRA Block Weight block lists (BASE = text encoder, not measured here)
LBW_BLOCKS = {
    12: ["BASE", "IN04", "IN05", "IN07", "IN08", "M00", "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05"],
    17: ["BASE", "IN01", "IN02", "IN04", "IN05", "IN07", "IN08", "M00", "OUT03", "OUT04", "OUT05", "OUT06",
         "OUT07", "OUT08", "OUT09", "OUT10", "OUT11"],
}
LORA_BLOCK_MIN = 0.05  # a LoRA adding less overlap than this in a block is not named


def lora_blocks(words, lb, rules=None, min_overlap=None):
    """Which LoRA blocks cause the same-area overlaps.

    Per same-area pair and block: the overlap with all LoRAs, and per LoRA how much of it goes away
    when that LoRA's to_q/to_k change is taken out of the block (first cross-attention of the
    block; LoRA changes to other layers of the block are not separated). ->
    [{"pair": (a, b), "overlap": total, "blocks": {block: overlap}, "added": {LoRA: {block: +overlap}}}]
    """
    if not lb:
        return []
    out = []
    for a, b, o in same_area_pairs(words, min_overlap, rules=rules):
        row = {"pair": (a, b), "overlap": o, "blocks": {}, "added": {n: {} for n in lb["loras"]}}
        for blk in lb["blocks"]:
            m = lb["maps"].get(("", blk), {})
            if a not in m or b not in m:
                continue
            ov = _overlap(m[a], m[b])
            row["blocks"][blk] = ov
            for n in lb["loras"]:
                w = lb["maps"].get((n, blk))
                if w and a in w and b in w:
                    row["added"][n][blk] = ov - _overlap(w[a], w[b])
        out.append(row)
    return out


def lbw_suggestion(rows, lora, blocks):
    """Block weights for <lora:name:w:lbw=...> that lower the blocks where `lora` adds overlap:
    weight 1 - (overlap it adds / block overlap), the worst pair per block; text or None."""
    ids = LBW_BLOCKS[17] if any(b in ("IN01", "IN02") for b in blocks) else LBW_BLOCKS[12]
    weights = []
    for blk in ids:
        share = max((r["added"][lora].get(blk, 0.0) / max(r["blocks"].get(blk, 0.0), 1e-6) for r in rows),
                    default=0.0)
        weights.append(round(1.0 - min(max(share, 0.0), 1.0), 1))
    return ",".join(f"{w:g}" for w in weights) if any(w < 1 for w in weights) else None


def _weight(lb, lora):
    w = lb.get("weights", {}).get(lora, 1.0)
    return f"{w:g}" if isinstance(w, (int, float)) else "1"


def _lora_html(words, lb, rules=None):
    """Collapsed per-block view of the same-area pairs: overlap per block, what each LoRA adds."""
    if not lb:
        return ""
    rows = lora_blocks(words, lb, rules)
    if not rows:
        return ('<details class="wm-check"><summary>LoRA blocks: no same-area pair to explain</summary>'
                '</details>')
    blocks = lb["blocks"]
    head = "".join(f"<th>{b}</th>" for b in blocks)
    worst, tables = {}, []
    for r in rows:
        a, b = r["pair"]
        cells = "".join(f'<td style="background:rgba(127,127,127,{min(1.0, r["blocks"].get(k, 0) * 1.4):.2f})">'
                        f'{r["blocks"][k]:.2f}</td>' if k in r["blocks"] else "<td></td>" for k in blocks)
        trs = [f"<tr><th>overlap</th>{cells}</tr>"]
        for n in lb["loras"]:
            add = r["added"][n]
            cells = []
            for k in blocks:
                v = add.get(k)
                if v is None:
                    cells.append("<td></td>")
                    continue
                if v >= LORA_BLOCK_MIN:
                    worst.setdefault(n, {})[k] = max(worst.get(n, {}).get(k, 0.0), v)
                color = f"rgba(220,60,60,{min(1.0, v * 3):.2f})" if v > 0 else f"rgba(60,160,90,{min(1.0, -v * 3):.2f})"
                cells.append(f'<td style="background:{color}">{v:+.2f}</td>')
            trs.append(f"<tr><th>{html.escape(n)}</th>{''.join(cells)}</tr>")
        tables.append(f"<p class=\"wm-p\"><b>{html.escape(a)}</b> &amp; <b>{html.escape(b)}</b> ({r['overlap']:.2f})"
                      f"</p><table><tr><th></th>{head}</tr>{''.join(trs)}</table>")
    lines = []
    for n in lb["loras"]:
        named = sorted(worst.get(n, {}).items(), key=lambda kv: -kv[1])
        if not named:
            lines.append(f"<li><b>{html.escape(n)}</b>: adds no overlap worth naming</li>")
            continue
        sug = lbw_suggestion(rows, n, blocks)
        lines.append(f"<li><b>{html.escape(n)}</b> adds the overlap in "
                     + ", ".join(f"<b>{k}</b> (+{v:.2f})" for k, v in named)
                     + (f" - try <code>&lt;lora:{html.escape(n)}:{_weight(lb, n)}:lbw={sug}&gt;</code>" if sug else "")
                     + "</li>")
    summary = "; ".join(f"{n}: " + ", ".join(k for k, _ in sorted(worst[n].items(), key=lambda kv: -kv[1])[:3])
                        for n in lb["loras"] if worst.get(n)) or "no LoRA block adds overlap"
    return (f'<details class="wm-check wm-tab"><summary>LoRA blocks behind the overlap: {html.escape(summary)}'
            f'</summary><ul>{"".join(lines)}</ul>{"".join(tables)}'
            '<p class="wm-p" style="font-size:12px">Overlap per block (first cross-attention of each block) with '
            'all LoRAs; per LoRA: how much overlap that block loses when the LoRA\'s attention change is taken '
            'out (red = the LoRA makes the words share the area, green = it separates them). The lbw '
            'weights are relative to the block weights used for this image.</p></details>')


def runs(words, groups):
    """Adjacent content words within a phrase, e.g. 'red dress', 'yellow flowers': [[key, ...]].

    The last word of a run is taken as its noun, the ones before it as describing words."""
    out, cur = [], []
    for text, m, _, _ in words:
        key = text.lower()
        if m is not None and key in groups and groups[key][3]:
            cur.append(key)
            continue
        if len(cur) > 1:
            out.append(cur)
        cur = []
    if len(cur) > 1:
        out.append(cur)
    return out


def check_prompt(words, rules=None):
    """Prompt check from the word maps -> [(kind, message markdown)], one line per kind:
    "weak" (words with little visible effect), "same_area" (nouns acting on the same area)."""
    keys, ov, groups = relations(words)
    if len(keys) < 2:
        return []
    word_runs = runs(words, groups)
    modifiers = {k for r in word_runs for k in r[:-1]}
    run_of = {k: i for i, r in enumerate(word_runs) for k in r}
    weak = []
    for k in keys:
        _, _, share, _, raw = groups[k]
        lift = float(np.max(raw) / (np.mean(raw) + 1e-8))
        if share < WEAK_SHARE and lift < WEAK_LIFT:
            weak.append(f"**{k}** ({share * 100:.1f}%)")
    same = []
    nouns = [k for k in keys if k not in modifiers]
    for i, a in enumerate(nouns):
        for b in nouns[i + 1:]:
            if a in run_of and run_of.get(a) == run_of.get(b):
                continue
            o = float(ov[keys.index(a), keys.index(b)])
            if o >= SAME_AREA_MIN and not (rules and rules.allowed(a, b)):
                same.append(f"**{a}** & **{b}** ({o:.2f})")
    found = []
    if weak:
        found.append(("weak", "Little visible effect: " + ", ".join(weak) + " - little attention, spread out. "
                              "Raise their influence (1.5) or write `(word:1.3)`; or drop them."))
    if same:
        found.append(("same_area", "Same area: " + ", ".join(same) + ". Same thing: one is enough. Part of "
                                   "the other (hair on a girl): fine. Different things: they compete for that "
                                   "place - give the important one more influence."))
    return found


def _md(text):
    """The bit of markdown the check messages use -> HTML."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


def _check_html(words, rules=None):
    """Collapsed prompt check: a one-line summary, details on click."""
    found = check_prompt(words, rules)
    if not found:
        return '<details class="wm-check"><summary>Prompt check: nothing stands out</summary></details>'
    counts = {"weak": "weak word", "same_area": "same-area pair"}
    parts = []
    for kind, msg in found:
        n = msg.count("**") // 2 // (2 if kind == "same_area" else 1)
        parts.append(f"{n} {counts.get(kind, kind)}{'s' if n != 1 else ''}")
    items = "".join(f"<li>{_md(msg)}</li>" for _, msg in found)
    return (f'<details class="wm-check"><summary>Prompt check: {", ".join(parts)}</summary>'
            f'<ul>{items}</ul></details>')


def _table_html(words):
    keys, ov, groups = relations(words)
    if len(keys) < 2:
        return ""
    head = "".join(f'<th style="background:{_css(groups[k][0])};color:{_text_color(groups[k][0])}">'
                   f'{html.escape(k)}</th>' for k in keys)
    rows = []
    for i, a in enumerate(keys):
        cells = "".join("<td></td>" if i == j else
                        f'<td style="background:rgba(127,127,127,{min(1.0, ov[i, j] * 1.4):.2f})">{ov[i, j]:.2f}</td>'
                        for j in range(len(keys)))
        rows.append(f'<tr><th style="background:{_css(groups[a][0])};color:{_text_color(groups[a][0])}">'
                    f'{html.escape(a)}</th>{cells}</tr>')
    return (f'<details class="wm-tab"><summary>Relation table (how much two words act on the same area)'
            f'</summary><table><tr><th></th>{head}</tr>{"".join(rows)}</table></details>')


def related_to(words, word, n=6):
    """Words sharing the most area with `word`: [(key, overlap)]."""
    keys, ov, groups = relations(words)
    key = (word or "").lower()
    if key not in keys:
        return []
    i = keys.index(key)
    pairs = sorted(((float(ov[i, j]), k) for j, k in enumerate(keys) if j != i), reverse=True)
    return [(k, o) for o, k in pairs[:n]]


# ---- one-click optimisation ----------------------------------------------------------------

MAX_CHANGES = 6
DROP_SHARE = 0.005   # weak words below this share are switched off instead of boosted
BOOST = 1.5
WEAKEN = 0.6
WEAKEN_FLOOR = 0.2  # overlapping words are weakened step by step down to this
BOOST_MAX = 3.0


def impact(groups, key):
    """How much a word shapes the image: its attention share x how concentrated it is."""
    _, _, share, _, raw = groups[key]
    lift = float(np.max(raw) / (np.mean(raw) + 1e-8))
    return share * float(np.log1p(lift))


def important_words(words, n, allowed=None):
    """The n words that shape the image most (impact: attention share x concentration), coloured words only
    (no fillers / attention sinks), limited to `allowed` (the words written in the prompt) -> [key]."""
    if n <= 0:
        return []
    groups = color_words(words)
    keys = [k for k, g in groups.items() if g[3] and (allowed is None or k in allowed)]
    return sorted(keys, key=lambda k: -impact(groups, k))[:n]


REMOVED = "too high: removed"  # reason text of an overlap removal (auto mode rephrases these)


def optimize(words, edits=None, locked=(), same_area_min=None, weak=True, allowed=None, remove_above=None,
             rules=None):
    """Prompt check -> word changes, most impactful first.

    edits: the factors the shown image was made with ({word: factor}, from word edits and prompt
    weights); a new factor builds on them, so running it again after a regenerate keeps
    separating words that still overlap. Locked words never change. Same-area noun pairs: the
    locked word is kept, else the one with more impact; the other is weakened (biggest overlap
    first). Weak words (not already weakened): boosted, or switched off when they get almost no
    attention. same_area_min: overlap from which a pair counts (default SAME_AREA_MIN); weak=False
    only separates words (auto mode); allowed: the only words that may change (the words written in
    the prompt; None = all); remove_above: overlap from which the weaker word is removed (factor 0)
    instead of weakened; rules (lib_act_steer.rules): allowed pairs are skipped, prefer decides which
    word changes, keep / never remove / limit are respected. Returns ([(word, factor, reason)],
    merged edits dict).
    """
    same_area_min = SAME_AREA_MIN if same_area_min is None else same_area_min
    keys, ov, groups = relations(words)
    edits = {k.lower(): v for k, v in (edits or {}).items()}
    locked = {w.strip().lower() for w in list(locked) + (rules.kept() if rules else []) if w.strip()}
    can = _changeable(locked, allowed)
    if rules:
        base_can = can
        can = lambda k: base_can(k) and not rules.is_kept(k)  # noqa: E731
    if len(keys) < 2:
        return [], edits
    word_runs = runs(words, groups)
    modifiers = {k for r in word_runs for k in r[:-1]}
    run_of = {k: i for i, r in enumerate(word_runs) for k in r}
    imp = {k: impact(groups, k) for k in keys}

    candidates = []  # (priority, word, factor, reason)
    nouns = [k for k in keys if k not in modifiers]
    for i, a in enumerate(nouns):
        for b in nouns[i + 1:]:
            if a in run_of and run_of.get(a) == run_of.get(b):
                continue
            o = float(ov[keys.index(a), keys.index(b)])
            if o < same_area_min or not (can(a) or can(b)) or (rules and rules.allowed(a, b)):
                continue
            pref = rules.preferred(a, b) if rules else None
            if pref and can(b if pref == a else a):
                keep, weaker = (a, b) if pref == a else (b, a)
                why = f"same area as **{keep}** ({o:.2f}), preferred by a rule"
            elif not (can(a) and can(b)):
                keep, weaker = (b, a) if can(a) else (a, b)
                why = (f"same area as locked **{keep}** ({o:.2f})" if keep in locked else
                       f"same area as **{keep}** ({o:.2f}), which is not in your prompt")
            else:
                keep, weaker = (a, b) if imp[a] >= imp[b] else (b, a)
                why = f"same area as **{keep}** ({o:.2f}), which has more impact"
            cur = edits.get(weaker, 1.0)
            if remove_above is not None and o >= remove_above and not (rules and rules.never_remove(weaker)):
                candidates.append((2.0 + o, weaker, 0.0, f"{why}; overlap {o:.2f} >= {remove_above:.2f} is {REMOVED}"))
                continue
            factor = round(max(WEAKEN_FLOOR, min(cur, 1.0) * WEAKEN), 2)
            if factor < cur - 1e-6:
                candidates.append((1.0 + o, weaker, factor, why))
    for k in keys:
        _, _, share, _, raw = groups[k]
        lift = float(np.max(raw) / (np.mean(raw) + 1e-8))
        cur = edits.get(k, 1.0)
        if weak and can(k) and share < WEAK_SHARE and lift < WEAK_LIFT and cur >= 1.0:
            if share < DROP_SHARE:
                candidates.append((0.5 + share, k, 0.0, f"almost no attention ({share * 100:.1f}%): switched off"))
            elif cur < BOOST_MAX:
                candidates.append((0.5 + share, k, round(min(BOOST_MAX, cur * BOOST), 2),
                                   f"little visible effect ({share * 100:.1f}%): boosted"))

    plan, used = [], set(locked)
    for _, word, factor, reason in sorted(candidates, key=lambda c: -c[0]):
        if word in used:
            continue
        used.add(word)
        plan.append((word, factor, reason))
        if len(plan) >= MAX_CHANGES:
            break
    if rules:  # limit rules clamp the factors
        plan = [(w, f if f == 0 else round(rules.limit(w, f), 2), why) for w, f, why in plan]
        plan = [(w, f, why) for w, f, why in plan if abs(f - edits.get(w, 1.0)) > 1e-6]
    merged = dict(edits)
    merged.update({w: f for w, f, _ in plan})
    return plan, merged
