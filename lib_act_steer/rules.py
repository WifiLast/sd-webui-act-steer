"""User rules for Optimize / Auto optimize, one per line. Two sets, both always applied:
default rules (<steer dir>/default_rules.txt, created from BUILTIN_DEFAULTS on the first start,
loaded on every start) and the current rules (<steer dir>/rules.txt).

    allow hair ~ girl            the two may share an area: no conflict, nothing changed for it
    allow * ~ background         * and ? are wildcards, a|b|c alternatives
    prefer ocean over sea        when they share an area, sea is the one that changes
    keep straw hat               never changed (like a locked word)
    never remove sky             weakened instead of removed or rephrased
    limit dress 0.7..1.3         the factor Optimize gives it stays in this range
    replace sea -> ocean         Optimize writes ocean where the prompt says sea
    negative keep lowres         this negative phrase is never changed
    negative add blurry:1.2      Optimize puts it into the negative prompt when missing
    place sea|ocean -> background   a fixed layer for every generation whose prompt has the word
                                 (targets as in the Layers field: background, foreground, left, right,
                                 top, bottom, center, on <word>, away from <word>, skin, clothes, hair, face, body; ":0.6" strength)
    ... when beach               any rule: only while the prompt contains "beach"
    # comment

Words are matched case-insensitively against the word map's words or the prompt's phrases.
"""

import fnmatch
import os
import re

from lib_act_steer import core, promptedit

PATH = os.path.join(core.STEER_DIR, "rules.txt")
DEFAULT_PATH = os.path.join(core.STEER_DIR, "default_rules.txt")

PERSONS = "girl|boy|woman|man|1girl|1boy|2girls|2boys|person|character|lady|child|kemonomimi|furry|anthro"
PARTS = ("hair|eyes|eye|face|mouth|lips|smile|expression|ears|ear|tail|hand|hands|arm|arms|leg|legs|skin|body|"
         "breasts|neck|fingers|nails|horns|wings|fur|blush|bangs|ponytail|twintails|braid|head|pose|sitting|"
         "standing|looking")
CLOTHES = ("dress|shirt|skirt|hat|cap|jacket|coat|hoodie|sweater|pants|shorts|shoes|boots|gloves|scarf|ribbon|bow|"
           "uniform|kimono|bikini|swimsuit|clothes|outfit|armor|cape|necklace|earrings|jewelry|glasses|hairband|"
           "hair ornament|thighhighs|socks|apron|belt|collar|veil|crown")
BUILTIN_DEFAULTS = f"""# Default rules: loaded on every start and applied together with the rules below.
# Parts and clothes of a person naturally share the person's area:
allow {PERSONS} ~ {PARTS}
allow {PERSONS} ~ {CLOTHES}
allow {PARTS} ~ {PARTS}
allow {PARTS} ~ {CLOTHES}
# Background words cover everything behind the subject:
allow * ~ background|scenery|outdoors|indoors|landscape
"""

_PATTERNS = [
    ("allow", re.compile(r"allow\s+(.+?)\s*~\s*(.+)$")),
    ("prefer", re.compile(r"prefer\s+(.+?)\s+over\s+(.+)$")),
    ("never_remove", re.compile(r"never\s+remove\s+(.+)$")),
    ("keep", re.compile(r"keep\s+(.+)$")),
    ("limit", re.compile(r"limit\s+(.+?)\s+(-?[\d.]+)\s*\.\.\s*(-?[\d.]+)$")),
    ("replace", re.compile(r"replace\s+(.+?)\s*->\s*(.+)$")),
    ("place", re.compile(r"place\s+(.+?)\s*->\s*((?:background|foreground|left|right|top|bottom|center|"
                         r"skin|clothes|cloth|clothing|outfit|hair|face|body|person|on \S.*?|away from \S.*?)(?::\s*[\d.]+)?)$")),
    ("neg_keep", re.compile(r"negative\s+keep\s+(.+)$")),
    ("neg_add", re.compile(r"negative\s+add\s+(.+?)(?::\s*(-?[\d.]+))?$")),
]


def _norm(text):
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip().lower()


class Rules:
    def __init__(self, text="", defaults=""):
        """text: the current rules; defaults: the default rules (applied too, errors marked)."""
        self.text = text or ""
        self.defaults = defaults or ""
        self.rules, self.errors = [], []
        self._parse(self.defaults, "default ")
        self._parse(self.text, "")

    def _parse(self, text, where):
        for n, line in enumerate(text.splitlines(), 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            when = None
            m = re.search(r"\s+when\s+(.+)$", line)
            if m:
                when, line = _norm(m.group(1)), line[:m.start()].strip()
            for kind, rx in _PATTERNS:
                mm = rx.match(line)
                if mm:
                    args = [a if a is None else (a.strip() if kind in ("limit", "neg_add") and i > 0 else _norm(a))
                            for i, a in enumerate(mm.groups())]
                    self.rules.append({"kind": kind, "args": args, "when": when, "line": n, "default": bool(where)})
                    break
            else:
                self.errors.append(f"{where}line {n}: '{line}' is not a rule")

    def for_prompt(self, prompt):
        """The rules that apply to this prompt (their `when` word is in it)."""
        text = _norm(re.sub(r"<[^>]*>|[()\[\]]|:\s*-?[\d.]+", " ", prompt or ""))
        words = set(re.findall(r"[\w']+", text))
        out = Rules()
        out.text, out.defaults, out.errors = self.text, self.defaults, self.errors
        out.rules = [r for r in self.rules if not r["when"] or r["when"] in words or r["when"] in text]
        return out

    # ---- queries ---------------------------------------------------------------------

    def _of(self, kind):
        return [r["args"] for r in self.rules if r["kind"] == kind]

    @staticmethod
    def _match(pattern, word):
        """The pattern matches the word / phrase, or one word of the phrase ("background" in
        "simple background")."""
        w = _norm(word)
        return any(fnmatch.fnmatchcase(w, p) or any(fnmatch.fnmatchcase(x, p) for x in w.split())
                   for p in pattern.split("|"))

    def allowed(self, a, b):
        """True when a rule allows a and b to share an area."""
        return any((self._match(x, a) and self._match(y, b)) or (self._match(x, b) and self._match(y, a))
                   for x, y in self._of("allow"))

    def preferred(self, a, b):
        """The one of a, b a prefer rule keeps, or None."""
        for x, y in self._of("prefer"):
            if self._match(x, a) and self._match(y, b):
                return a
            if self._match(x, b) and self._match(y, a):
                return b
        return None

    def kept(self):
        return [x for (x,) in self._of("keep")]

    def is_kept(self, word):
        return any(self._match(x, word) for x in self.kept())

    def never_remove(self, word):
        return any(self._match(x, word) for (x,) in self._of("never_remove"))

    def limit(self, word, factor):
        for x, lo, hi in self._of("limit"):
            if self._match(x, word):
                try:
                    return min(max(factor, float(lo)), float(hi))
                except ValueError:
                    pass
        return factor

    def replacements(self):
        return [(a, b) for a, b in self._of("replace")]

    def neg_kept(self):
        return [x for (x,) in self._of("neg_keep")]

    def neg_added(self):
        out = []
        for phrase, w in self._of("neg_add"):
            try:
                out.append((phrase, float(w) if w else None))
            except ValueError:
                out.append((phrase, None))
        return out

    def placements(self, prompt):
        """Place rules for this prompt -> ["words -> target[:strength]"] (the Layers field syntax), one
        per prompt phrase or word the rule's pattern matches (wildcards and alternatives expanded)."""
        phrases = promptedit.prompt_phrases(prompt)
        words = promptedit.prompt_words(prompt)
        out = []
        for pattern, target in self._of("place"):
            hits = []
            for alt in pattern.split("|"):
                alt = alt.strip()
                if not any(c in alt for c in "*?["):
                    found = alt in phrases or alt in words or any(f" {alt} " in f" {x} " for x in phrases)
                    hits += [alt] if found else []
                else:
                    ph = sorted(x for x in phrases if fnmatch.fnmatchcase(x, alt))
                    hits += ph + sorted(w for w in words if fnmatch.fnmatchcase(w, alt)
                                        and not any(w in x.split() for x in ph))
            out += [f"{h} -> {target}" for h in dict.fromkeys(hits)]
        return out

    def summary(self):
        d = sum(r["default"] for r in self.rules)
        n = len(self.rules) - d
        text = f"{n} rule{'s' if n != 1 else ''} + {d} default rule{'s' if d != 1 else ''}"
        return text + (" — " + "; ".join(self.errors) if self.errors else "")


_current = None


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write((text or "").rstrip() + "\n")


def default_text():
    """The default rules; the first start writes BUILTIN_DEFAULTS to default_rules.txt."""
    text = _read(DEFAULT_PATH)
    if text is None:
        _write(DEFAULT_PATH, BUILTIN_DEFAULTS)
        text = BUILTIN_DEFAULTS
    return text


def current():
    """Default + current rules (read on the first call = webui start, updated by the save functions)."""
    global _current
    if _current is None:
        _current = Rules(_read(PATH) or "", default_text())
    return _current


def save(text):
    """Save the current rules."""
    global _current
    _write(PATH, text)
    _current = Rules(text, current().defaults)
    return _current


def save_defaults(text):
    """Save the default rules (loaded on every start)."""
    global _current
    _write(DEFAULT_PATH, text)
    _current = Rules(current().text, text)
    return _current


def add_line(line):
    """Append a rule to the current rules (unless the same line is there) and save -> their text."""
    text = current().text.rstrip()
    if line.strip().lower() not in {x.strip().lower() for x in text.splitlines()}:
        text = (text + "\n" if text else "") + line
    save(text)
    return current().text
