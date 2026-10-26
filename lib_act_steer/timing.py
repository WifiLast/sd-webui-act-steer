"""Timing: composition words early, detail words late (webui prompt editing, [word:0.3]).

The layout of an image forms in the first ~30 % of the steps. core keeps a snapshot of the word maps
at EARLY_FRAC of the steps (the early maps) next to the final ones. Two words fighting for the same
area (the conflict check) where the weaker one already pulls at the layout early (its early share is
at least EARLY_RATIO of its final share) are separated in time instead of weakening the weaker
word: it joins at DELAY of the steps, when the stronger word has placed itself, and still appears as
a detail. Words that act mostly late anyway are weakened as before (a delay would not change much).
A delayed word the image then misses is moved back (un-delayed). Quality / meta tags (masterpiece, best
quality ...) are never delayed.
"""

from lib_act_steer import order, promptedit

DELAY = 0.3
EARLY_RATIO = 0.5


def shares(words):
    """{word (lower case): attention share} of a harvested word list."""
    out = {}
    for t, m, share, _ in words or []:
        if m is not None:
            out[t.lower()] = out.get(t.lower(), 0.0) + float(share)
    return out


def plan(conflict_plan, words, early_words, prompt, missed=(), current=None, verdict=None):
    """conflict_plan: [(word, factor, why)] of wordmap.optimize. -> (delays [(word, why)], undelays
    [(word, why)], the conflict plan without the delayed words)."""
    verdict = verdict or (lambda key: None)
    current = current or {}
    delayed = promptedit.delayed_words(prompt)
    late, early = shares(words), shares(early_words)
    # a delayed word is not in the first steps' prompt: its positions held padding then, not the word
    early = {k: v for k, v in early.items() if k not in delayed}
    total = sum(early.values()) or 1.0
    early = {k: v / total for k, v in early.items()}
    missed_words = {w for t in missed for w in t.split()}
    undelays = [(w, f"**{w}** was delayed and the image misses it: back in place")
                for w in sorted(delayed & missed_words) if verdict(f"undelay:{w}") != "bad"]
    ps = promptedit.pieces(prompt)
    delays, rest = [], []
    for w, f, why in conflict_plan:
        weakening = 0 < f < current.get(w, 1.0) and "same area" in why
        e, l = early.get(w, 0.0), late.get(w, 0.0)
        i = promptedit.find_piece(ps, w)
        if (weakening and early_words and w not in delayed and w not in missed_words and l > 0
                and e >= EARLY_RATIO * l and i >= 0 and "[" not in ps[i] and not order.is_meta(ps[i])
                and verdict(f"delay:{w}") != "bad"):
            delays.append((w, f"{why}; it pulls at the layout early ({e * 100:.1f}% of the attention at "
                              f"{DELAY:.0%} vs {l * 100:.1f}% at the end): joins at {DELAY:.0%} of the steps "
                              "instead of being weakened"))
        else:
            rest.append((w, f, why))
    return delays, undelays, rest
