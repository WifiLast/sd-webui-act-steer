"""Phrase order and 75-token chunks of the prompt.

CLIP reads the prompt in chunks of 75 tokens and gives the first tokens of a chunk the most weight;
a phrase cut by a chunk boundary ("long | hair") loses its meaning. Optimize moves only phrases it
has a reason for:

* the subject (person word: 1girl, woman ...) comes first after the leading quality / meta tags
  (masterpiece, best quality, score_9 ... stay where they are: models are trained with them first),
* phrases the image misses (SD MCP tagger) move up, right after the subject,
* phrases that get almost no attention move to the end,
* a phrase split across a chunk boundary gets a BREAK before it (the webui only moves phrases up to
  "Prompt word wrap length limit" tokens long to the next chunk itself).
"""

import re

from lib_act_steer import promptedit

META = re.compile(
    r"^(masterpiece|best quality|high quality|amazing quality|great quality|good quality|normal quality|"
    r"very aesthetic|aesthetic|absurdres|highres|incredibly absurdres|ultra detailed|highly detailed|"
    r"extremely detailed|official art|newest|recent|mid|early|late|old|8k|4k|uhd|hdr|safe|sensitive|nsfw|"
    r"explicit|questionable|general|score \d+( up)?|source \w+|rating \w+|year \d{4}|very awa|"
    r"best aesthetic|detailed|high resolution|ultra high res)$")


def is_meta(piece):
    return piece.startswith("<") or bool(META.match(promptedit.phrase_key(piece)))


def lead_len(ps):
    """Number of leading quality / meta pieces (and <lora> tags) - they keep their place."""
    n = 0
    while n < len(ps) and is_meta(ps[n]):
        n += 1
    return n


def plan(prompt, subject=(), up=(), down=(), locked=()):
    """-> (new prompt, [(kind, piece, why)]) with kind subject_first | move_up | move_down.
    subject: person words; up: words / phrases the image misses; down: weak words (a phrase only goes
    down when all its words are weak, like the weak-words button); locked words' pieces don't move down."""
    ps = promptedit.pieces(prompt)
    if len(ps) < 3 or re.search(r"\b(BREAK|AND)\b", prompt or ""):  # hand-made chunks / regions keep their order
        return prompt, []
    moves = []
    n0 = lead_len(ps)
    subject = {w.lower() for w in subject}
    si = next((i for i in range(n0, len(ps)) if promptedit.prompt_words(ps[i]) & subject and not ps[i].startswith("<")),
              -1)
    if si > n0:
        piece = ps.pop(si)
        ps.insert(n0, piece)
        moves.append(("subject_first", piece, "the subject first, right after the quality tags"))
    at = n0 + (1 if si >= n0 else 0)
    for words in up:
        i = promptedit.find_piece(ps, words)
        if i < 0 or i <= at or ps[i].startswith("<") or promptedit.is_delayed(ps[i]):
            at = max(at, i + 1) if 0 <= i <= at else at
            continue
        piece = ps.pop(i)
        ps.insert(at, piece)
        at += 1
        moves.append(("move_up", piece, "missing in the image: moved up (early tokens weigh more)"))
    weak = {w.lower() for w in down}
    locked = {w.lower() for w in locked}
    tail = []
    for i in range(len(ps) - 1, n0 - 1, -1):
        p = ps[i]
        words = promptedit.prompt_words(p)
        if p.startswith("<") or promptedit.is_delayed(p) or not words or not words <= weak or words & locked:
            continue
        if i < len(ps) - 1 - len(tail):
            tail.insert(0, ps.pop(i))
            moves.append(("move_down", p, "almost no attention: moved to the end"))
    ps += tail
    return ", ".join(ps), moves


def chunk_splits(lengths, chunk=75, backtrack=20):
    """lengths: tokens per comma piece (a comma token between pieces). Simulates the webui's chunking
    (modules/sd_hijack_clip.py tokenize_line, comma word wrap) -> [index of each piece cut by a chunk
    boundary]."""
    owner = []  # piece of each token in the current chunk (-1 = comma)
    chunk_of, cur, last_comma = {}, 0, -1

    def close():
        nonlocal cur, owner, last_comma
        for p in owner:
            if p >= 0:
                chunk_of.setdefault(p, set()).add(cur)
        cur, owner, last_comma = cur + 1, [], -1

    for i, n in enumerate(lengths):
        stream = ([-1] if i else []) + [i] * n
        for p in stream:
            if p == -1:
                last_comma = len(owner)
            elif backtrack and len(owner) == chunk and last_comma != -1 and len(owner) - last_comma <= backtrack:
                moved = owner[last_comma + 1:]
                owner = owner[:last_comma + 1]
                close()
                owner = moved
            if len(owner) == chunk:
                close()
            owner.append(p)
    close()
    return [p for p, cs in sorted(chunk_of.items()) if len(cs) > 1]


def fix_splits(prompt, count_tokens, backtrack=20, allow_break=True):
    """Pieces cut by a chunk boundary -> (prompt, [(piece, how)]). A cut piece moves to the first later
    place where no piece is cut (the content stays the same); when there is none and allow_break, a
    BREAK goes before it (the chunk is padded). count_tokens(text) -> number of CLIP tokens. Prompts that
    already have BREAK / AND are left alone (hand-made chunks or Regional Prompter regions)."""
    if re.search(r"\b(BREAK|AND)\b", prompt or ""):
        return prompt, []
    ps = promptedit.pieces(prompt)
    lengths = [0 if p.startswith("<") else count_tokens(re.sub(r"[()\[\]]", " ", promptedit._WEIGHT_NUM.sub(" ", p)))
               for p in ps]
    cut = chunk_splits(lengths, backtrack=backtrack)
    done = []
    while cut:
        i = cut[0]
        best = None
        for j in range(i + 1, len(ps)):  # move piece i behind piece j
            order = list(range(len(ps)))
            order.insert(j, order.pop(i))
            left = chunk_splits([lengths[k] for k in order], backtrack=backtrack)
            if not left or (best is None and len(left) < len(cut)):
                best = (order, left)
                if not left:
                    break
        if best is None:
            break
        order, cut = best
        done.append((ps[i], "moved so it stays inside one 75-token chunk"))
        ps, lengths = [ps[k] for k in order], [lengths[k] for k in order]
    if cut and allow_break:
        done += [(ps[i], "BREAK before it: it was cut by the 75-token chunk boundary") for i in cut]
        ps = [("BREAK " if i in cut else "") + p for i, p in enumerate(ps)]
    return ", ".join(ps), done
