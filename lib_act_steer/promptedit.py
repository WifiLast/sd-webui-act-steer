"""Write word factors into the prompt text as webui emphasis: (word:0.6), word removed for 0.

Only single words are touched (the word map works on CLIP words). <lora:...> style tags and the
inside of other words are left alone; a word that already carries (word:x) gets x replaced.
"""

import re

_TAG = re.compile(r"<[^>]*>")


def _weighted(word):
    return re.compile(r"\(\s*(" + re.escape(word) + r")\s*:\s*(-?[\d.]+)\s*\)", re.I)


def _bare(word):
    # not inside other words, and not a number of a weight like (sea:0.6)
    return re.compile(r"(?<![\w:.])" + re.escape(word) + r"(?![\w]|\.\d)", re.I)


_WEIGHT_NUM = re.compile(r":\s*-?[\d.]+\s*(?=[)\]])")


def prompt_words(prompt):
    """The words written in the prompt (lower case; not inside <...> tags, no weight numbers, no
    pure numbers): only these can be changed in the prompt."""
    text = _WEIGHT_NUM.sub(" ", _TAG.sub(" ", prompt or ""))
    return {w.lower() for w in re.findall(r"\w+", text) if not w.isdigit()}


def _outside_tags(prompt, fn):
    """Apply fn to the parts of prompt outside <...> tags."""
    out, last = [], 0
    for m in _TAG.finditer(prompt):
        out.append(fn(prompt[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(fn(prompt[last:]))
    return "".join(out)


def prompt_weights(prompt, words):
    """{word: weight} for the words written as (word:x) in prompt (others: not in the dict)."""
    out = {}
    text = _TAG.sub(" ", prompt or "")
    for w in words:
        m = _weighted(w).search(text)
        if m:
            try:
                out[w.lower()] = float(m.group(2))
            except ValueError:
                pass
    return out


def _tidy(prompt):
    prompt = re.sub(r"\(\s*\)", "", prompt)
    prompt = re.sub(r"[ \t]+", " ", prompt)
    prompt = re.sub(r"\s+,", ",", prompt)
    prompt = re.sub(r",(\s*,)+", ",", prompt)
    prompt = re.sub(r"\(\s+", "(", prompt)
    prompt = re.sub(r"\s+\)", ")", prompt)
    return prompt.strip().strip(",").strip()


def prompt_phrases(prompt):
    """The comma phrases of the prompt, lower case, without weights, brackets and <...> tags."""
    out = set()
    for piece in _split(_TAG.sub(" ", prompt or "")):
        t = re.sub(r"[()\[\]]", " ", _WEIGHT_NUM.sub(" ", piece)).replace("_", " ")
        t = re.sub(r"\s+", " ", t).strip().lower()
        if t:
            out.add(t)
    return out


def _split(prompt):
    """Comma phrases outside (), [] and <>."""
    pieces, cur, depth = [], [], 0
    for ch in prompt:
        depth += ch in "([<"
        depth = max(0, depth - (ch in ")]>"))
        if ch == "," and depth == 0:
            pieces.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    return pieces + ["".join(cur)]


def _drop_phrases(prompt, word):
    """Remove the comma phrases whose last word is `word` ("blue sky" for sky; a lone
    "(sky:0.6)" too), so no dangling describing words are left."""
    rx = re.compile(r"[A-Za-z][\w']*")
    out = []
    for piece in _split(prompt):
        text = _WEIGHT_NUM.sub(" ", piece)
        toks = rx.findall(text) if not piece.strip().startswith("<") else []
        if toks and toks[-1].lower() == word.lower() and "," not in piece:
            continue
        out.append(piece.strip())
    return ", ".join(p for p in out if p)



def drop_weak_phrases(prompt, weak, ignore=()):
    """Remove the comma phrases all of whose words are in `weak` (filler words in `ignore` don't count;
    <lora:...> tags and phrases with a word not in `weak` stay, so "best quality" never becomes
    "quality"). -> (new prompt, [removed phrases], [kept phrases that have weak words])."""
    weak, ignore = {w.lower() for w in weak}, {w.lower() for w in ignore}
    out, removed, partial = [], [], []
    for piece in _split(prompt):
        words = prompt_words(piece) - ignore if not piece.strip().startswith("<") else set()
        if words and words <= weak:
            removed.append(piece.strip())
            continue
        if words & weak:
            partial.append(piece.strip())
        out.append(piece.strip())
    return _tidy(", ".join(p for p in out if p)), removed, partial

def apply_to_prompt(prompt, factors):
    """prompt with each {word: factor} written as (word:factor); 1 = plain word, 0 = removed
    (with its phrase when it is the phrase's last word / noun)."""
    for word, f in factors.items():
        if not word.strip() or word.strip().isdigit():
            continue
        f = float(f)
        if abs(f) < 1e-6:
            prompt = _drop_phrases(prompt, word)

        def rep(m):
            w = m.group(1) if m.re.groups else m.group(0)
            return "" if abs(f) < 1e-6 else w if abs(f - 1.0) < 1e-6 else f"({w}:{f:g})"

        weighted, bare = _weighted(word), _bare(word)
        prompt = _outside_tags(prompt, lambda part: weighted.sub(rep, part) if weighted.search(part)
                               else bare.sub(rep, part))
    return _tidy(prompt)


def replace_text(prompt, old, new):
    """prompt with the word / phrase `old` replaced by `new` outside <...> tags; "(old:0.6)" keeps
    its weight. -> (prompt, number of replacements)."""
    weighted, bare = _weighted(old), _bare(old)
    count = 0

    def rep(m):
        nonlocal count
        count += 1
        return f"({new}:{m.group(2)})" if m.re is weighted else new

    prompt = _outside_tags(prompt, lambda part: bare.sub(rep, weighted.sub(rep, part)))
    return prompt, count


def dedupe_phrases(prompt):
    """Drop later comma phrases that say the same as an earlier one ("ocean, (ocean:0.6)" after a
    replacement keeps the first)."""
    seen, out = set(), []
    for piece in _split(prompt or ""):
        key = re.sub(r"\s+", " ", re.sub(r"[()\[\]]", " ", _WEIGHT_NUM.sub(" ", piece)).replace("_", " ")).strip().lower()
        if key and not piece.strip().startswith("<"):
            if key in seen:
                continue
            seen.add(key)
        if piece.strip():
            out.append(piece.strip())
    return ", ".join(out)


# ---- comma pieces: find, move, delay ([piece:0.3] = from 30 % of the steps on) ------------------

_SCHED = re.compile(r"\[([^\[\]]+?):\s*(\d*\.?\d+)\s*\]")


def phrase_key(piece):
    """A comma piece as the SD MCP tags it: lower case, no weights, brackets or underscores."""
    t = re.sub(r"[()\[\]]", " ", _WEIGHT_NUM.sub(" ", piece or "")).replace("_", " ")
    return re.sub(r"\s+", " ", t).strip().lower()


def pieces(prompt):
    return [p.strip() for p in _split(prompt or "") if p.strip()]


def find_piece(ps, words):
    """Index of the comma piece holding `words` (a word or phrase): the piece that is just this phrase,
    else one ending in its last word, else one containing all its words; <...> tags never. -1 if none."""
    want = [w.lower() for w in re.findall(r"\w+", (words or "").replace("_", " "))]
    if not want:
        return -1
    keys = [phrase_key(p) if not p.startswith("<") else "" for p in ps]
    for test in (lambda k: k == " ".join(want),
                 lambda k: k.split()[-1:] == want[-1:] and set(want) <= set(k.split()),
                 lambda k: set(want) <= set(k.split())):
        for i, k in enumerate(keys):
            if k and test(k):
                return i
    return -1


def move_piece(prompt, words, to):
    """prompt with the piece holding `words` moved to index `to` (-1 = the end) -> (prompt, moved piece
    or None when it is not there / already in place)."""
    ps = pieces(prompt)
    i = find_piece(ps, words)
    to = len(ps) - 1 if to < 0 else min(to, len(ps) - 1)
    if i < 0 or i == to:
        return prompt, None
    piece = ps.pop(i)
    ps.insert(to, piece)
    return ", ".join(ps), piece


def _outer_colon(text):
    depth = 0
    for ch in text:
        depth += ch in "(["
        depth = max(0, depth - (ch in ")]"))
        if ch == ":" and depth == 0:
            return True
    return False


def is_delayed(piece):
    """'[red hat:0.3]' (added from 30 % of the steps on) -> True; [a:b:0.5] swaps are not delays."""
    m = _SCHED.fullmatch((piece or "").strip())
    return bool(m) and not _outer_colon(m.group(1))


def delayed_words(prompt):
    """Words of the delayed pieces of the prompt."""
    return {w for p in pieces(prompt) if is_delayed(p) for w in prompt_words(p)}


def delay_piece(prompt, words, when=0.3):
    """The piece holding `words` leaves its place and is added at the end as [piece:when]: it joins
    from that step fraction on, after the layout has formed. At the end, its missing tokens in the
    first steps shift no other word's position. -> (prompt, piece or None)."""
    ps = pieces(prompt)
    i = find_piece(ps, words)
    if i < 0 or is_delayed(ps[i]) or "[" in ps[i] or ps[i].startswith("<"):
        return prompt, None
    piece = ps.pop(i)
    return ", ".join(ps + [f"[{piece}:{when:g}]"]), piece


def undelay_piece(prompt, words):
    """[piece:when] holding `words` -> piece, in place. -> (prompt, piece or None)."""
    ps = pieces(prompt)
    for i, p in enumerate(ps):
        if is_delayed(p) and set(re.findall(r"\w+", words.lower())) <= prompt_words(p):
            ps[i] = _SCHED.fullmatch(p).group(1).strip()
            return ", ".join(ps), ps[i]
    return prompt, None


_OUTER = re.compile(r"^\((.*):\s*(-?\d*\.?\d+)\s*\)$", re.S)


def piece_weight(piece, word=None):
    """Weight of a comma piece "(red hat:1.2)" (word=None), or of one word inside it "red (hat:1.2)";
    1.0 when unweighted."""
    piece = (piece or "").strip()
    if word is None:
        m = _OUTER.match(piece)
        return float(m.group(2)) if m and not _outer_colon(m.group(1)) else 1.0
    m = _weighted(word).search(piece)
    return float(m.group(2)) if m else 1.0


def weight_piece(prompt, words, factor, word=None):
    """Weight only inside the piece holding `words`: the whole piece -> "(red hat:1.2)", or one word of it
    -> "(red:1.2) hat"; other pieces with the same word stay as they are. -> (prompt, new piece or None)."""
    ps = pieces(prompt)
    i = find_piece(ps, words)
    if i < 0 or ps[i].startswith("<") or is_delayed(ps[i]):
        return prompt, None
    p, f = ps[i], float(factor)
    if word is None:
        m = _OUTER.match(p)
        core = m.group(1).strip() if m and not _outer_colon(m.group(1)) else p
        ps[i] = core if abs(f - 1.0) < 1e-6 else f"({core}:{f:g})"
    else:
        ps[i] = _outside_tags(p, lambda part: _weighted(word).sub(lambda m: f"({m.group(1)}:{f:g})", part)
                              if _weighted(word).search(part) else _bare(word).sub(lambda m: f"({m.group(0)}:{f:g})", part, count=1))
    return ", ".join(ps), ps[i]
