"""Prompt adherence: fix the prompt phrases the image does not show.

The SD MCP evaluation (lib_act_steer/evaluate.py) tags the image with the WD14 tagger and returns
missed_tags (prompt phrases the tagger knows but did not find) and image_shows (what it found). For
each missed phrase, in this order:

* bind: an attribute landed on another object ("red hat" missed, the image shows "red dress" and the
  hat): the attribute word gets a layer "red -> on hat" (its attention only where the hat is),
* boost: the phrase's last word (the noun) is strengthened, x1.2 per round up to BOOST_MAX,
* move up: already strong enough (or boosting did not help before): the phrase moves up in the prompt
  (lib_act_steer/order.py does the moving).

Missed words are never weakened by the conflict check in the same round (see protected()).
"""

import re

from lib_act_steer import promptedit

BOOST = 1.2
BOOST_MAX = 1.4
MAX_FIXES = 4  # missed phrases fixed per round (the rest next round), so learning can tell what helped


def _norm(tag):
    return re.sub(r"\s+", " ", (tag or "").replace("_", " ")).strip().lower()


def missed(ev):
    """Missed tags of an evaluation result (normalised), [] when there is none."""
    if not isinstance(ev, dict) or ev.get("error") or not ev.get("ok", True):
        return []
    return [_norm(t) for t in ev.get("missed_tags") or [] if _norm(t)]


def protected(ev, prompt):
    """Words of the missed phrases that are in the prompt: the conflict check must not weaken them."""
    words = promptedit.prompt_words(prompt)
    return {w for t in missed(ev) for w in t.split() if w in words}


def shown(ev, prompt):
    """Words of the prompt phrases the tagger found in the image: they are there, so a shared area with
    another word is no reason to weaken or remove them (a hat under an umbrella overlaps it)."""
    if not isinstance(ev, dict) or ev.get("error") or not ev.get("ok", True):
        return set()
    words = promptedit.prompt_words(prompt)
    return {w for t in ev.get("matched_tags") or [] for w in _norm(t).split() if w in words}


def plan(ev, prompt, current, locked=(), rules=None, verdict=None):
    """-> [{"kind": bind | boost | move_up, "word", "tag", "factor" + "scope" (boost: "phrase" = the whole
    missed phrase is weighted, "word" = only this word inside it), "layer" (bind: "red -> on hat"), "why"}]
    for the missed phrases of ev that are in the prompt. current: {word: factor} of the word edits (the
    prompt weights are read from the missed phrase itself); verdict(key) -> "good" | "bad" | None."""
    verdict = verdict or (lambda key: None)
    locked = {w.lower() for w in locked}
    shows = [_norm(t) for t in (ev or {}).get("image_shows") or []]
    seen = shown(ev, prompt) | {w for t in shows for w in t.split()}
    ps = promptedit.pieces(prompt)
    out = []
    for tag in missed(ev):
        if len(out) >= MAX_FIXES:
            break
        if promptedit.find_piece(ps, tag) < 0:
            continue
        toks = tag.split()
        noun = toks[-1]
        attrs = toks[:-1]
        if attrs and (noun in shows or any(s.split()[-1:] == [noun] for s in shows)):
            # the object is there: is the attribute on another prompt object?
            # (any object, asked for or not: "red dress" when the prompt has no dress is the same leak;
            # a pair the prompt itself asks for, "red eyes", is not)
            phrases = promptedit.prompt_phrases(prompt)
            other = next((s.split()[-1] for s in shows if len(s.split()) >= 2 and s.split()[:-1] == attrs
                          and s.split()[-1] != noun and s not in phrases), None)
            if other and verdict(f"bind:{tag}") != "bad":
                out.append({"kind": "bind", "word": " ".join(attrs), "tag": tag, "layer": f"{' '.join(attrs)} -> on {noun}",
                            "why": f"**{tag}** missed, the image shows **{' '.join(attrs)} {other}**: "
                                   f"'{' '.join(attrs)}' only where the {noun} is"})
                continue
        # the object is in the image ("blue eyes" missed, "red eyes" shown): the attribute is what is missing;
        # else the whole phrase gets the weight (a word of it would also change other phrases with that word)
        piece = ps[promptedit.find_piece(ps, tag)]
        scope = "word" if attrs and noun in seen else "phrase"
        target = attrs[-1] if scope == "word" else noun
        if any(a["word"] == target for a in out):
            continue
        cur = float(current.get(target, 1.0)) * promptedit.piece_weight(piece, target if scope == "word" else None)
        kept = rules is not None and rules.is_kept(target)
        if cur < BOOST_MAX - 1e-6 and target not in locked and not kept and verdict(f"boost_missed:{target}") != "bad":
            f = round(min(BOOST_MAX, max(cur * BOOST, 1.1)), 2)
            if rules is not None:
                f = round(rules.limit(target, f), 2)
            if f > cur + 1e-6:
                out.append({"kind": "boost", "word": target, "factor": f, "scope": scope, "tag": tag,
                            "why": f"**{tag}** is not in the image (SD MCP tagger): strengthened"
                                   + (f" ('{noun}' is in the image, '{target}' is missing)" if target != noun else "")})
                continue
        if verdict(f"move_up:{tag}") != "bad":
            out.append({"kind": "move_up", "word": tag, "tag": tag,
                        "why": f"**{tag}** still missing at {cur:g}: moved up in the prompt"})
    return out
