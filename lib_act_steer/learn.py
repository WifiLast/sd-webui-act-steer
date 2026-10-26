"""Learning from Optimize / Auto optimize: which changes made the next image better.

Every optimisation is an experiment: the image before (its worst word conflict, the number of
blurry / flat / noisy word areas, the SD MCP quality score when known), the changes (weaken /
strengthen / remove a word, a rephrasing, a replace rule, a negative prompt change) and the image
after (the next generation with the changed prompt, same seed). Experiments are stored in
<steer dir>/learned.json and summed up per change ("change key"):

    score = (conflict before - after) + QUALITY_WEIGHT * (quality after - before) + 0.05 * (flags
    before - after) + ADHERENCE_WEIGHT * (prompt adherence after - before) + RATING_WEIGHT * (your
    rating after - before), averaged over the experiments the change was part of (each change of an
    experiment gets its full result: several changes at once share the credit).

Your ratings (👍 +1 / 👎 -1 on the shown image, unrated 0) outweigh the measurements: a change
whose images you rated worse is bad right away, one you rated better good (unless measured bad).

Reuse: Optimize skips changes with a bad record and prefers ones with a good record; new prompts
get the changes with a good record applied before they are generated (before_process); word pairs
whose changes never helped become "allow" rule suggestions.
"""

import json
import os
import threading
import time
import uuid

from lib_act_steer import core

PATH = os.path.join(core.STEER_DIR, "learned.json")
QUALITY_WEIGHT = 4.0  # quality_score moves in hundredths, conflicts in tenths
ADHERENCE_WEIGHT = 1.0  # one missed tag of eight = 0.125
RATING_WEIGHT = 0.3
MIN_N = 2             # experiments before a change counts as good / bad
GOOD, BAD = 0.02, -0.02
MAX_RECORDS = 5000

_lock = threading.RLock()
_data = None
_pending = {}  # tab -> experiment waiting for the next image
_ratings = {}  # tab -> {word map version: +1 / -1}, your ratings of the shown images


def _load():
    global _data
    if _data is None:
        try:
            with open(PATH, encoding="utf-8") as f:
                _data = json.load(f)
        except (OSError, ValueError):
            _data = {"records": []}
    return _data


def _save():
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_data, f, ensure_ascii=False)
    os.replace(tmp, PATH)


def change(kind, word, detail="", other=""):
    """A change as stored: kind = weaken | strengthen | remove | rephrase | replace | neg_remove |
    neg_weaken | neg_add | neg_boost; detail = new factor or text; other = the word it overlapped."""
    word = (word or "").strip().lower()
    if kind in ("weaken", "strengthen"):
        key = f"{kind}:{word}"
    elif kind == "rephrase":
        key = f"rephrase:{word}->{(detail or '').strip().lower()}"
    else:
        key = f"{kind}:{word}"
    return {"key": key, "kind": kind, "word": word, "detail": detail, "other": (other or "").strip().lower()}


def start(tab, model, prompt_before, neg_before, prompt_after, neg_after, before, changes, version=None):
    """An optimisation was applied; its result is the next image of this tab (finish()).
    before = {"conflict", "flags", "quality" or None, "adherence" or None}; version = word map version
    of the image before (its rating). -> the experiment id."""
    if not changes:
        return None
    before = dict(before)
    if version is not None and version in _ratings.get(tab, {}):
        before["rating"] = _ratings[tab][version]
    exp = {"id": uuid.uuid4().hex[:12], "time": time.time(), "model": model or "", "tab": tab,
           "prompt_before": prompt_before or "", "neg_before": neg_before or "",
           "prompt_after": prompt_after or "", "neg_after": neg_after or "",
           "before": before, "after": None, "changes": changes, "before_version": version}
    with _lock:
        _pending[tab] = exp
    return exp["id"]


def finish(tab, prompt, after, version=None):
    """The next image of the tab is there: when it was made with the experiment's prompt, store the
    experiment with after = {"conflict", "flags", "quality", "adherence"}; version = its word map
    version (for a rating given later). -> the experiment id or None."""
    with _lock:
        exp = _pending.pop(tab, None)
        if exp is None:
            return None
        if _norm(prompt) != _norm(exp["prompt_after"]):
            return None  # the prompt was changed by hand in between: not this experiment's result
        exp["after"] = dict(after)
        exp["after_version"] = version
        data = _load()
        data["records"].append(exp)
        data["records"] = data["records"][-MAX_RECORDS:]
        _save()
        return exp["id"]


def set_quality(exp_id, side, quality, adherence=None):
    """The SD MCP quality score / prompt adherence of the image before / after arrived (background
    evaluation; None = not measured, kept as it is)."""
    vals = {k: v for k, v in (("quality", quality), ("adherence", adherence)) if v is not None}
    if exp_id is None or not vals:
        return
    with _lock:
        for tab, exp in _pending.items():
            if exp["id"] == exp_id:
                exp[side].update(vals)
                return
        for rec in reversed(_load()["records"]):
            if rec["id"] == exp_id:
                if rec[side] is not None:
                    rec[side].update(vals)
                    _save()
                return


def rate(tab, version, value):
    """Your rating (+1 / -1, 0 = clear) of the image with word map `version`: stored for the
    experiment that made it (after) and the one started from it (before). -> number of experiments."""
    n = 0
    with _lock:
        if value:
            _ratings.setdefault(tab, {})[version] = int(value)
        else:
            _ratings.get(tab, {}).pop(version, None)
        exp = _pending.get(tab)
        if exp and exp.get("before_version") == version:
            exp["before"]["rating"] = int(value) or None
            n += 1
        changed = False
        for rec in reversed(_load()["records"][-50:]):
            if rec.get("tab", tab) != tab:
                continue
            for side in ("before", "after"):
                if rec.get(f"{side}_version") == version and rec.get(side) is not None:
                    rec[side]["rating"] = int(value) or None
                    changed = True
                    n += 1
        if changed:
            _save()
    return n


def rating_of(tab, version):
    return _ratings.get(tab, {}).get(version)


def pending_id(tab):
    with _lock:
        exp = _pending.get(tab)
        return exp["id"] if exp else None


def _norm(text):
    return " ".join((text or "").lower().split())


def _result(rec):
    """-> (score, quality change or None, your rating change: -2..2, 0 when you rated neither)."""
    b, a = rec["before"], rec["after"] or {}
    score = (b.get("conflict") or 0.0) - (a.get("conflict") or 0.0)
    score += 0.05 * ((b.get("flags") or 0) - (a.get("flags") or 0))
    dq = None
    if b.get("quality") is not None and a.get("quality") is not None:
        dq = a["quality"] - b["quality"]
        score += QUALITY_WEIGHT * dq
    if b.get("adherence") is not None and a.get("adherence") is not None:
        score += ADHERENCE_WEIGHT * (a["adherence"] - b["adherence"])
    user = (a.get("rating") or 0) - (b.get("rating") or 0)
    score += RATING_WEIGHT * user
    return score, dq, user


def stats(model=None):
    """{change key: {"n", "score", "d_quality" (mean, None if never measured), "kind", "word",
    "detail", "others"}} over the stored experiments (of `model` first when it has any)."""
    with _lock:
        recs = [r for r in _load()["records"] if r.get("after")]
    if model:
        own = [r for r in recs if r.get("model") == model]
        recs = own or recs
    out = {}
    for rec in recs:
        score, dq, user = _result(rec)
        for ch in rec["changes"]:
            s = out.setdefault(ch["key"], {"n": 0, "score": 0.0, "dq": [], "kind": ch["kind"], "word": ch["word"],
                                           "detail": ch.get("detail"), "others": set(), "user": 0})
            s["n"] += 1
            s["score"] += score
            s["user"] += user
            s["detail"] = ch.get("detail")  # the latest value / text used
            if dq is not None:
                s["dq"].append(dq)
            if ch.get("other"):
                s["others"].add(ch["other"])
    for s in out.values():
        s["score"] /= s["n"]
        s["d_quality"] = sum(s["dq"]) / len(s["dq"]) if s["dq"] else None
        del s["dq"]
    return out


def verdict(st, key):
    """good | bad | None (not enough experiments) for a change key."""
    s = st.get(key)
    if s and s.get("user", 0) < 0:  # you rated its images worse
        return "bad"
    if s and s.get("user", 0) > 0 and s["score"] > BAD:
        return "good"
    if not s or s["n"] < MIN_N:
        return None
    bad_quality = s["d_quality"] is not None and s["d_quality"] < -0.01
    if s["score"] >= GOOD and not bad_quality:
        return "good"
    if s["score"] <= BAD or bad_quality:
        return "bad"
    return None


def proven(st, prompt_words, neg_phrases=()):
    """Changes with a good record that fit a new prompt -> [stats entry + key], best first: prompt
    changes whose word is in the prompt (and, for overlap fixes, one of the words it overlapped);
    negative ones whose phrase is in the negative prompt (neg_add: is not)."""
    neg_phrases = set(neg_phrases)
    out = []
    for key, s in st.items():
        if verdict(st, key) != "good":
            continue
        if s["kind"] == "neg_add":
            fits = s["word"] not in neg_phrases
        elif s["kind"].startswith("neg_"):
            fits = s["word"] in neg_phrases
        else:
            words = set(s["word"].split())
            fits = bool(words) and words <= prompt_words and (not s["others"] or bool(s["others"] & prompt_words))
        if fits:
            out.append({**s, "key": key})
    return sorted(out, key=lambda s: -s["score"])


def allow_suggestions(st, min_n=3):
    """Word pairs where weakening / removing / rephrasing never helped: 'allow a ~ b' rule lines."""
    lines = []
    for key, s in st.items():
        if s["kind"] in ("weaken", "remove", "rephrase") and s["n"] >= min_n and s["score"] <= 0 and \
                (s["d_quality"] is None or s["d_quality"] <= 0.005):
            for o in sorted(s["others"]):
                line = f"allow {s['word']} ~ {o}"
                if line not in lines:
                    lines.append(line)
    return lines


def summary(model=None):
    with _lock:
        n = sum(1 for r in _load()["records"] if r.get("after"))
    st = stats(model)
    good = [k for k in st if verdict(st, k) == "good"]
    bad = [k for k in st if verdict(st, k) == "bad"]
    return n, st, good, bad


def reset():
    global _data
    with _lock:
        _data = {"records": []}
        _pending.clear()
        _ratings.clear()
        _save()
