"""Negative prompt check and optimisation, from the negative prompt's word maps (uncond rows).

Works on the comma phrases of the negative prompt ("bad hands", "(worst quality:1.4)"):
- idle: the model hardly looks at the phrase (attention share below IDLE_SHARE): removed.
- conflict: the phrase names something the positive prompt asks for ("hat" against "straw hat"):
  removed when the phrase is only that, weakened when its area overlaps the positive word's.
- duplicate: the same phrase twice: the later one removed.
- defect: a defect tag the evaluation (SD MCP) found in the image ("blurry", "bad hands"):
  strengthened when the phrase is there, added when not.
Textual inversion embeddings, <...> tags and locked words are never changed.
"""

import re

import numpy as np

from lib_act_steer import promptedit
from lib_act_steer.wordmap import _overlap

IDLE_SHARE = 0.002       # a phrase with less of the negative prompt's attention does nothing visible
CONFLICT_OVERLAP = 0.3   # area overlap from which a conflicting phrase that says more is weakened
WEAKEN = 0.6
WEAKEN_FLOOR = 0.3
DEFECT_BOOST = 1.15
DEFECT_MAX = 1.5
MAX_CHANGES = 8
# words both prompts use without meaning the same thing ("best quality" / "worst quality")
GENERIC = {"quality", "detail", "details", "detailed", "resolution", "res", "image", "art", "style", "score",
           "version", "body", "skin", "face", "eyes", "hands", "hand", "fingers", "lighting", "light", "color",
           "colors", "colour", "background", "photo", "picture", "drawing", "anatomy", "the", "and", "with",
           "very", "high", "low", "bad", "good", "best", "worst", "normal"}

_WORD = re.compile(r"[A-Za-z][\w']*")


def _phrases(neg_prompt):
    """[(raw piece, inner text, weight float or None, lower-case words)] of the negative prompt."""
    out = []
    for piece in promptedit._split(neg_prompt or ""):
        raw = piece.strip()
        if not raw:
            continue
        m = re.fullmatch(r"\((.*):\s*(-?[\d.]+)\s*\)", raw, re.S)
        inner, weight = (m.group(1).strip(), float(m.group(2))) if m else (raw.strip("()[] "), None)
        words = [w.lower() for w in _WORD.findall(promptedit._WEIGHT_NUM.sub(" ", inner))]
        out.append((raw, inner, weight, words))
    return out


def _stats(words):
    """{word: (share, raw map)} from harvested negative word maps (same word: summed)."""
    out = {}
    for text, m, share, raw in words or []:
        if m is None:
            continue
        k = text.lower()
        s, r = out.get(k, (0.0, None))
        out[k] = (s + share, raw if r is None else r + raw)
    return out


def _pos_maps(pos_words):
    out = {}
    for text, m, _, raw in pos_words or []:
        if m is not None:
            k = text.lower()
            out[k] = raw if k not in out else out[k] + raw
    return out


def check(neg_prompt, neg_words, pos_prompt, pos_words=None, defects=(), locked=(), protected=()):
    """-> [{"kind", "phrase" (raw text), "action": remove|weaken|boost|add|none, "weight", "why"}],
    most important first. protected: embedding names and other words never to touch."""
    stats = _stats(neg_words)
    pos_maps = _pos_maps(pos_words)
    pos_words_set = {w for w in promptedit.prompt_words(pos_prompt or "") if w not in GENERIC and len(w) > 2}
    keep = {w.strip().lower() for w in list(locked) + list(protected) if w.strip()}
    found, seen = [], {}
    phrases = _phrases(neg_prompt)
    for raw, inner, weight, words in phrases:
        if raw.startswith("<") or not words or set(words) & keep or any(w.lower() in keep for w in [inner]):
            continue
        norm = " ".join(words)
        if norm in seen:
            found.append({"kind": "duplicate", "phrase": raw, "action": "remove", "weight": None,
                          "why": f"same as '{seen[norm]}' earlier in the negative prompt"})
            continue
        seen[norm] = raw
        hits = [w for w in words if w in pos_words_set]
        if hits:
            w = hits[0]
            content = [x for x in words if x not in GENERIC and x not in ("a", "an", "of", "and", "no")]
            ov = _overlap(stats[w][1], pos_maps[w]) if w in stats and w in pos_maps \
                and stats[w][1] is not None else None
            if set(content) <= set(hits):
                found.append({"kind": "conflict", "phrase": raw, "action": "remove", "weight": None,
                              "why": f"the positive prompt asks for '{w}'"})
            elif ov is not None and ov >= CONFLICT_OVERLAP:
                cur = weight if weight is not None else 1.0
                if cur > WEAKEN_FLOOR + 1e-6:
                    found.append({"kind": "conflict", "phrase": raw, "action": "weaken",
                                  "weight": round(max(WEAKEN_FLOOR, cur * WEAKEN), 2),
                                  "why": f"contains '{w}', which the positive prompt asks for; same area "
                                         f"({ov:.2f})"})
            continue
        share = sum(stats.get(w, (0.0, None))[0] for w in words)
        if stats and share < IDLE_SHARE and all(w in stats for w in words):
            found.append({"kind": "idle", "phrase": raw, "action": "remove", "weight": None,
                          "why": f"the model hardly looks at it ({share * 100:.2f}% of the negative attention)"})
    for tag in defects or ():
        text = tag.replace("_", " ").strip()
        tw = [w.lower() for w in _WORD.findall(text)]
        if not tw or set(tw) & keep:
            continue
        match = next(((raw, weight) for raw, _, weight, words in phrases if words == tw), None)
        if match is None:
            found.append({"kind": "defect", "phrase": text, "action": "add", "weight": DEFECT_BOOST,
                          "why": f"the evaluation found '{tag}' in the image"})
        elif (match[1] or 1.0) < DEFECT_MAX - 1e-6:
            found.append({"kind": "defect", "phrase": match[0], "action": "boost",
                          "weight": round(min(DEFECT_MAX, (match[1] or 1.0) * DEFECT_BOOST), 2),
                          "why": f"the evaluation still finds '{tag}' in the image"})
    order = {"conflict": 0, "defect": 1, "duplicate": 2, "idle": 3}
    return sorted(found, key=lambda f: order.get(f["kind"], 9))


def apply(neg_prompt, issues, limit=MAX_CHANGES):
    """-> (new negative prompt, [applied issues]); at most `limit` changes."""
    todo = [f for f in issues if f["action"] != "none"][:limit]
    pieces = [p.strip() for p in promptedit._split(neg_prompt or "")]
    done = []
    for f in todo:
        if f["action"] == "add":
            pieces.append(f"({f['phrase']}:{f['weight']:g})")
            done.append(f)
            continue
        idx = [i for i, p in enumerate(pieces) if p == f["phrase"]]
        if f["kind"] == "duplicate":
            idx = idx[1:]  # keep the first one
        if not idx:
            continue
        i = idx[-1] if f["kind"] == "duplicate" else idx[0]
        if f["action"] == "remove":
            pieces[i] = ""
        else:
            inner = re.fullmatch(r"\((.*):\s*(-?[\d.]+)\s*\)", pieces[i], re.S)
            text = inner.group(1).strip() if inner else pieces[i].strip("()[] ")
            pieces[i] = f"({text}:{f['weight']:g})"
        done.append(f)
    return ", ".join(p for p in pieces if p), done


def summary(issues):
    counts = {}
    for f in issues:
        counts[f["kind"]] = counts.get(f["kind"], 0) + 1
    names = {"conflict": "conflict", "defect": "defect", "duplicate": "duplicate", "idle": "idle phrase"}
    return ", ".join(f"{n} {names.get(k, k)}{'s' if n != 1 else ''}" for k, n in counts.items()) or "nothing found"


def lines(issues):
    verbs = {"remove": "removed", "weaken": "weakened to", "boost": "strengthened to", "add": "added as"}
    out = []
    for f in issues:
        w = f" {f['weight']:g}" if f["action"] in ("weaken", "boost", "add") else ""
        out.append(f"- negative **{f['phrase']}** {verbs.get(f['action'], '')}{w}: {f['why']}")
    return "\n".join(out)


def add_required(neg_prompt, items):
    """Rule "negative add": put each (phrase, weight or None) into the negative prompt when no phrase
    of it says the same. -> (new negative prompt, [issue dicts for lines()])."""
    have = {" ".join(words) for _, _, _, words in _phrases(neg_prompt)}
    pieces = [p.strip() for p in promptedit._split(neg_prompt or "") if p.strip()]
    done = []
    for phrase, weight in items:
        words = " ".join(w.lower() for w in _WORD.findall(phrase.replace("_", " ")))
        if not words or words in have:
            continue
        pieces.append(f"({phrase}:{weight:g})" if weight else phrase)
        have.add(words)
        done.append({"kind": "rule", "phrase": phrase, "action": "add", "weight": weight or 1.0,
                     "why": "a rule asks for it"})
    return ", ".join(pieces), done


def phrase_key(raw):
    """A negative phrase as learned: lower case, no weight / brackets, spaces."""
    inner = re.sub(r"[()\[\]]", " ", promptedit._WEIGHT_NUM.sub(" ", raw or "")).replace("_", " ")
    return " ".join(inner.lower().split())


def set_phrase(neg_prompt, key, weight):
    """Give the negative phrase `key` (phrase_key form) the weight (None = remove it, 1 = plain).
    -> (new negative prompt, changed?)."""
    pieces, changed = [], False
    for piece in promptedit._split(neg_prompt or ""):
        raw = piece.strip()
        if raw and not raw.startswith("<") and phrase_key(raw) == key:
            changed = True
            if weight is None:
                continue
            inner = re.fullmatch(r"\((.*):\s*(-?[\d.]+)\s*\)", raw, re.S)
            text = inner.group(1).strip() if inner else raw.strip("()[] ")
            raw = text if abs(weight - 1.0) < 1e-6 else f"({text}:{weight:g})"
        if raw:
            pieces.append(raw)
    return ", ".join(pieces), changed
