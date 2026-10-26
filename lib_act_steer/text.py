"""Prompt tokens -> cross-attention context positions (chunk * 77 + token).

A1111 (cond_stage_model) and Forge (text_processing_engine_l / _engine) CLIP models; Forge's Anima, whose
cross-attention context has one row per T5 token of the prompt (no start token, one chunk, padded to 512)."""

import torch

from modules import prompt_parser, shared


def tokenizer():
    """The CLIP text engine (tokenize_line, tokenize, id_start / id_end / id_pad), None for other models."""
    m = shared.sd_model
    c = getattr(m, "cond_stage_model", None)
    for e in (getattr(c, "embedders", None) or [c]) + [getattr(m, "text_processing_engine_l", None),
                                                       getattr(m, "text_processing_engine", None)]:
        if hasattr(e, "tokenize_line") and hasattr(e, "tokenize") and hasattr(e, "id_start"):
            return e
    return None


def _anima():
    e = getattr(shared.sd_model, "text_processing_engine_anima", None)
    return e if hasattr(e, "t5_tokenizer") else None


def _final(prompt, steps):
    try:  # prompt editing [a:b:0.5]: use the text of the final schedule
        return prompt_parser.get_learned_conditioning_prompt_schedules([prompt], steps)[0][-1][1]
    except Exception:
        return prompt


def _chunks(emb, prompt, steps):
    out = emb.tokenize_line(_final(prompt, steps))
    return out[0] if isinstance(out, tuple) else out


def _t5_tokens(eng, prompt, steps):
    """Anima: the T5 ids of the prompt = the context rows, in order."""
    chunks = eng.tokenize_line(_final(prompt, steps))
    return list(chunks[0].t5_tokens) if chunks else []


def _find(seq, ids):
    return [j for j in range(len(seq) - len(ids) + 1) if seq[j:j + len(ids)] == ids]


def focus_positions(prompt, focus, steps=20):
    """Positions of the comma-separated focus words in prompt, as a LongTensor."""
    eng = _anima()
    if eng is not None and focus:
        seq, pos = _t5_tokens(eng, prompt, steps), set()
        for word in focus.split(","):
            ids = eng.t5_tokenizer([word.strip()], add_special_tokens=False)["input_ids"][0] if word.strip() else []
            for j in _find(seq, ids) if ids else []:
                pos.update(range(j, j + len(ids)))
        return torch.tensor(sorted(pos), dtype=torch.long)
    emb = tokenizer()
    if emb is None or not focus:
        return torch.zeros(0, dtype=torch.long)
    chunks = _chunks(emb, prompt, steps)
    pos = set()
    for word in focus.split(","):
        ids = emb.tokenize([word.strip()])[0] if word.strip() else []
        if not ids:
            continue
        for ci, chunk in enumerate(chunks):
            t = chunk.tokens
            for j in range(len(t) - len(ids) + 1):
                if t[j:j + len(ids)] == ids:
                    pos.update(range(ci * len(t) + j, ci * len(t) + j + len(ids)))
    return torch.tensor(sorted(pos), dtype=torch.long)


def word_positions(prompt, steps=20):
    """(LongTensor of positions holding real prompt tokens, {position: token text})."""
    emb = tokenizer()
    if emb is None:
        return torch.zeros(0, dtype=torch.long), {}
    special = {emb.id_start, emb.id_end, emb.id_pad}
    tok = getattr(emb, "tokenizer", None) or getattr(getattr(emb, "wrapped", None), "tokenizer", None)
    pos, text = [], {}
    for ci, chunk in enumerate(_chunks(emb, prompt, steps)):
        for j, t in enumerate(chunk.tokens):
            if t in special:
                continue
            p = ci * len(chunk.tokens) + j
            pos.append(p)
            try:
                text[p] = tok.decode([t]).strip() if tok is not None else str(t)
            except Exception:
                text[p] = str(t)
    return torch.tensor(pos, dtype=torch.long), text


def word_groups(prompt, steps=20):
    """[(word text, LongTensor of its token positions or None for punctuation)] in prompt order.

    CLIP marks the last token of a word with '</w>', so tokens are joined until one ends a word; T5 (Anima)
    marks the first token of a word with '▁'.
    """
    eng = _anima()
    if eng is not None:
        return _t5_word_groups(eng, prompt, steps)
    emb = tokenizer()
    if emb is None:
        return []
    special = {emb.id_start, emb.id_end, emb.id_pad}
    tok = getattr(emb, "tokenizer", None) or getattr(getattr(emb, "wrapped", None), "tokenizer", None)
    words, ids, pos = [], [], []

    def flush():
        if not ids:
            return
        try:
            text = tok.decode(ids).strip()
        except Exception:
            text = " ".join(map(str, ids))
        has_word = any(ch.isalnum() for ch in text)
        words.append((text, torch.tensor(pos, dtype=torch.long) if has_word else None))
        ids.clear()
        pos.clear()

    for ci, chunk in enumerate(_chunks(emb, prompt, steps)):
        for j, t in enumerate(chunk.tokens):
            if t in special:
                continue
            ids.append(t)
            pos.append(ci * len(chunk.tokens) + j)
            try:
                piece = tok.convert_ids_to_tokens([t])[0]
            except Exception:
                piece = "</w>"
            if piece.endswith("</w>"):
                flush()
        flush()
    return words


def _t5_word_groups(eng, prompt, steps):
    tok = eng.t5_tokenizer
    seq = _t5_tokens(eng, prompt, steps)
    special = {eng.id_end, eng.id_pad, getattr(tok, "pad_token_id", None)}
    words, ids, pos = [], [], []

    def flush():
        if not ids:
            return
        text = tok.decode(ids).strip()
        has_word = any(ch.isalnum() for ch in text)
        words.append((text, torch.tensor(pos, dtype=torch.long) if has_word else None))
        ids.clear()
        pos.clear()

    for j, t in enumerate(seq):
        if t in special:
            continue
        piece = tok.convert_ids_to_tokens([t])[0]
        alnum = any(ch.isalnum() for ch in piece)
        prev_alnum = bool(ids) and any(ch.isalnum() for ch in tok.convert_ids_to_tokens([ids[-1]])[0])
        if piece.startswith("\u2581") or not alnum or not prev_alnum:  # a new word, or punctuation
            flush()
        ids.append(t)
        pos.append(j)
        if not alnum:
            flush()
    flush()
    return words
