"""Tune <lora:name:te:unet:lbw=...> tags from the LoRA block analysis (wordmap.harvest_blocks).

Two signals, both from the attention maps of one image (with all LoRAs vs. with one LoRA's to_q/to_k
change taken out, per block):

- conflicts: the overlap the LoRA adds between two prompt nouns in a block (wordmap.lora_blocks).
  Per block factor 1 - share of the block's overlap the LoRA adds (at most STEP_MIN per round).
  When the LoRA adds overlap in most of the blocks it acts on, it is too strong as a whole: the
  median factor goes into the strength and only the blocks worse than that stay lowered in lbw=.
- suppression: the LoRA takes the prompt words' areas away (their attention gets less concentrated
  with the LoRA than without). Many suppressed words, or a word the evaluation missed in the image
  that the LoRA suppresses -> strength x STRENGTH_STEP.

Block weights are written for every block of the tag's lbw list (12 / 17 / 20 / 26 values, kept
when the tag has one; new: 12 SDXL, 17 SD1.5); BASE (text encoder) and blocks without
cross-attention are left as they are. The tag's te weight is kept; only unet changes.
"""

import os
import re

import numpy as np

from lib_act_steer import wordmap

LBW_IDS = {
    12: wordmap.LBW_BLOCKS[12],
    17: wordmap.LBW_BLOCKS[17],
    20: ["BASE", "IN00", "IN01", "IN02", "IN03", "IN04", "IN05", "IN06", "IN07", "IN08", "M00", "OUT00", "OUT01",
         "OUT02", "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08"],
    26: ["BASE"] + [f"IN{i:02d}" for i in range(12)] + ["M00"] + [f"OUT{i:02d}" for i in range(12)],
}
STEP_MIN = 0.5        # a block weight drops to at most half per round
BLOCK_FLOOR = 0.0
STRENGTH_STEP = 0.85  # strength factor for a suppressing LoRA
STRENGTH_MIN = 0.2
SPREAD_SHARE = 0.6    # LoRA adds overlap in this share of its blocks -> lower the strength instead
SUPPRESS_DROP = 0.25  # a word's concentration falls this much with the LoRA ...
SUPPRESS_SHARE = 0.5  # ... for this share of the prompt words -> the LoRA dominates

_TAG = re.compile(r"<(lora|lyco):([^>]*)>", re.I)


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _presets():
    """LoRA Block Weight presets {NAME: "1,0,..."} (the extension's file, the webui's scripts/ one wins)."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    out = {}
    for path in (os.path.join(root, "extensions", "sd-webui-lora-block-weight", "scripts", "lbwpresets.txt"),
                 os.path.join(root, "scripts", "lbwpresets.txt")):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        out[k.strip()] = v.strip()
        except OSError:
            pass
    return out


def _weights(text):
    """lbw value (list or preset name) -> [float] of a known length, or None (random R/U, unknown preset)."""
    if text is None:
        return None
    text = _presets().get(text.strip(), text)
    vals = [_num(v.strip()) for v in text.split(",")]
    if len(vals) not in LBW_IDS or any(v is None for v in vals):
        return None
    return vals


def parse(prompt):
    """<lora:...> / <lyco:...> tags of a prompt -> [{"span", "kind", "name", "te", "unet", "lbw",
    "lbw_text", "named"}]; te / unet None when not a plain number (e.g. loractl schedules)."""
    out = []
    for m in _TAG.finditer(prompt or ""):
        items = m.group(2).split(":")
        pos = [x for x in items[1:] if "=" not in x]
        named = [x for x in items[1:] if "=" in x]
        kv = {x.split("=", 1)[0].strip().lower(): x.split("=", 1)[1] for x in named}
        te_t = kv.get("te", pos[0] if pos else "1")
        unet_t = kv.get("unet", pos[1] if len(pos) > 1 and _num(pos[1]) is not None else te_t)
        lbw_t = kv.get("lbw")
        if lbw_t is None and len(pos) > 1 and _num(pos[1]) is None:  # old syntax <lora:name:1:ALL>
            lbw_t = pos[1]
        extra = pos[2:]  # dyn and the like
        out.append({"span": m.span(), "kind": m.group(1), "name": items[0].strip(), "te": _num(te_t),
                    "te_text": te_t, "unet": _num(unet_t), "unet_text": unet_t, "lbw": _weights(lbw_t),
                    "lbw_text": lbw_t, "extra": extra,
                    "named": [x for x in named if x.split("=", 1)[0].strip().lower() not in ("te", "unet", "lbw")]})
    return out


def _fmt(v):
    return f"{round(v, 2):g}"


def format_tag(tag, unet=None, lbw=None):
    """The tag text with a new unet strength and / or block weights (other params kept)."""
    unet_t = _fmt(unet) if unet is not None else tag["unet_text"]
    parts = [tag["name"], tag["te_text"], unet_t] + tag["extra"]
    lbw_t = ",".join(_fmt(v) for v in lbw) if lbw is not None else tag["lbw_text"]
    if lbw_t is not None:
        parts.append(f"lbw={lbw_t}")
    parts += tag["named"]
    return f"<{tag['kind']}:{':'.join(parts)}>"


def _focus(raw):
    return float(wordmap.concentration(raw).mean())


def suppression(lb, lora, words):
    """{word: drop} how much less concentrated each word's attention is with the LoRA than without
    it (0.3 = 30 % less area where the word stands out), over the blocks the LoRA acts on."""
    with_, without = {}, {}
    for (n, blk), m in lb["maps"].items():
        if n != lora:
            continue
        full = lb["maps"].get(("", blk), {})
        for w in words:
            if w in m and w in full:
                with_[w] = with_.get(w, 0.0) + _focus(full[w])
                without[w] = without.get(w, 0.0) + _focus(m[w])
    return {w: 1.0 - with_[w] / without[w] for w in with_ if without[w] > 1e-3}


def plan(prompt, words, lb, rules=None, locked=(), min_overlap=None, missed=(), allowed=None):
    """-> [{"tag", "name", "unet", "lbw", "old", "text", "why": [text], "changes": [(kind, word,
    detail) for learn.change]}] for the LoRA tags of `prompt` the block analysis `lb` measured;
    nothing for locked / kept LoRA names.
    missed: words the evaluation did not find in the image; allowed: prompt words to judge
    suppression on (the ones written in the prompt)."""
    if not lb:
        return []
    locked = {w.strip().lower() for w in locked}
    rows = wordmap.lora_blocks(words, lb, rules, min_overlap)
    groups = wordmap.color_words(words)
    content = [k for k in groups if k not in wordmap.STOPWORDS and (allowed is None or k in allowed)]
    measured = set(lb["blocks"])
    missed = {w for t in missed for w in re.findall(r"\w+", str(t).lower())}
    out = []
    for tag in parse(prompt):
        name = next((n for n in lb["loras"] if n.lower() == tag["name"].lower()), None)
        if name is None or name.lower() in locked or (rules and rules.is_kept(name.lower())):
            continue
        acts = [b for b in lb["blocks"] if (name, b) in lb["maps"]]
        if not acts:
            continue
        factor, why = {}, []
        for blk in acts:
            share = max((r["added"][name].get(blk, 0.0) / max(r["blocks"].get(blk, 0.0), 1e-6) for r in rows
                         if r["added"][name].get(blk, 0.0) >= wordmap.LORA_BLOCK_MIN), default=0.0)
            factor[blk] = max(1.0 - min(share, 1.0), STEP_MIN)
        hit = [b for b in acts if factor[b] < 0.95]
        strength = 1.0
        if hit and len(hit) >= SPREAD_SHARE * len(acts):
            strength = float(np.median([factor[b] for b in acts]))
            factor = {b: min(1.0, f / strength) for b, f in factor.items()}
            why.append(f"adds word overlap in {len(hit)} of its {len(acts)} blocks: strength x{strength:.2f}")
        elif hit:
            pairs = sorted({f"{r['pair'][0]} & {r['pair'][1]}" for r in rows
                            if any(r["added"][name].get(b, 0) >= wordmap.LORA_BLOCK_MIN for b in hit)})
            why.append("makes " + ", ".join(pairs) + " share an area in " + ", ".join(hit))
        drop = suppression(lb, name, content)
        sup = sorted((w for w, d in drop.items() if d >= SUPPRESS_DROP), key=lambda w: -drop[w])
        miss = [w for w in sup if w in missed]
        if miss or (len(drop) >= 3 and len(sup) >= SUPPRESS_SHARE * len(drop)):
            strength *= STRENGTH_STEP
            why.append("takes the area of " + ", ".join(miss or sup[:5])
                       + (" (missing in the image)" if miss else f" ({len(sup)} of {len(drop)} words)")
                       + f": strength x{STRENGTH_STEP:g}")
        unet = new_lbw = None
        changes = []
        if strength < 0.99 and tag["unet"] is not None:
            unet = max(STRENGTH_MIN, tag["unet"] * strength)
            if rules:
                unet = rules.limit(name.lower(), unet)
            if round(unet, 2) < round(tag["unet"], 2):
                changes.append(("lora_weaken", name.lower(), round(unet, 2)))
            else:
                unet = None
        if any(f < 0.95 for f in factor.values()):
            cur = tag["lbw"]
            if cur is None or not set(factor) <= set(LBW_IDS[len(cur)]):  # none, or a list of the other model type
                if cur is not None:
                    why.append(f"its lbw list ({len(cur)} values) does not fit this model: replaced")
                cur = [1.0] * (17 if measured & {"IN01", "IN02"} else 12)
            ids = LBW_IDS[len(cur)]
            new_lbw = [max(BLOCK_FLOOR, v * factor.get(b, 1.0)) if b in factor else v for b, v in zip(ids, cur)]
            if [round(v, 2) for v in new_lbw] == [round(v, 2) for v in cur]:
                new_lbw = None
            else:
                low = [b for b, v, c in zip(ids, new_lbw, cur) if round(v, 2) < round(c, 2)]
                changes.append(("lora_blocks", name.lower(), ",".join(_fmt(v) for v in new_lbw)))
                if not why:
                    why.append("lowered " + ", ".join(low))
        if unet is None and new_lbw is None:
            continue
        out.append({"tag": tag, "name": name, "unet": unet, "lbw": new_lbw, "why": why, "changes": changes,
                    "old": prompt[tag["span"][0]:tag["span"][1]], "text": format_tag(tag, unet, new_lbw)})
    return out


def apply(prompt, changes):
    """Write the planned tags into the prompt (by their old text: other changes may have moved them)."""
    for ch in changes:
        prompt = prompt.replace(ch["old"], ch["text"], 1)
    return prompt


def pair_words(changes, words, lb, rules=None, min_overlap=None):
    """Words of the overlap pairs the LoRA changes take care of (the LoRA adds at least half of the
    pair's overlap in a block it lowers): Optimize leaves weakening them for this round."""
    if not changes:
        return set()
    out = set()
    for r in wordmap.lora_blocks(words, lb, rules, min_overlap):
        for ch in changes:
            n = ch["name"]
            if any(v >= 0.5 * max(r["blocks"].get(b, 0.0), 1e-6) for b, v in r["added"][n].items()):
                out.update(r["pair"])
    return out
