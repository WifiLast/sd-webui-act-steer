"""Parts of the figure for the Layers field: skin, clothes, hair, face, body.

`lace -> clothes, glossy -> skin`: the word may only act where that part is. Made like Regional Prompter's
Prompt mode, no segmentation model: a part's area is where its own words of the prompt (dress / shirt ... for
clothes, hair / ponytail ... for hair, the person word for body) get cross-attention in this generation.
skin = body without clothes and hair. A part without a word in the prompt gets its name appended
(", clothes"), as RP appends "background"; body falls back to the subject (the strongest word).
With "As Regional Prompter regions" on, the moved word gets a BREAK region whose target is that word
(scripts/act_steer.py _rp_regions); otherwise core masks the word's attention with the same areas.
"""

import re

# target name -> canonical layer
ALIASES = {"skin": "skin", "clothes": "clothes", "cloth": "clothes", "clothing": "clothes", "outfit": "clothes",
           "hair": "hair", "face": "face", "body": "body", "person": "body"}
NAMES = tuple(ALIASES)
# the words of the prompt that mark a part (rules.py has the same person / clothes lists for its defaults)
WORDS = {
    "body": {"girl", "boy", "woman", "man", "1girl", "1boy", "2girls", "2boys", "person", "character", "lady",
             "child", "kemonomimi", "furry", "anthro", "body"},
    "clothes": {"dress", "shirt", "skirt", "jacket", "coat", "hoodie", "sweater", "pants", "shorts", "shoes", "boots",
                "gloves", "scarf", "uniform", "kimono", "bikini", "swimsuit", "clothes", "clothing", "outfit", "armor",
                "cape", "apron", "thighhighs", "socks", "blouse", "suit", "leotard", "bodysuit", "robe", "lingerie",
                "vest", "corset"},
    "hair": {"hair", "bangs", "ponytail", "twintails", "braid", "braids", "hairstyle"},
    "face": {"face", "eyes", "eye", "mouth", "lips", "smile", "expression", "nose", "cheeks", "blush", "freckles"},
    "skin": {"skin"},
}
APPEND = {"clothes": "clothes", "hair": "hair", "face": "face", "skin": "skin"}  # body: the subject instead


def canonical(target):
    return ALIASES.get((target or "").strip().lower())


def _is(layer, word):
    word = word.lower()
    return word in WORDS[layer] or (layer == "hair" and word.endswith("hair"))


def words_in(layer, words, exclude=()):
    """The words of `words` (prompt words, any case) that mark `layer`, without `exclude`."""
    exclude = {w.lower() for w in exclude}
    return [w for w in words if _is(layer, w) and w.lower() not in exclude]


def minus(layer):
    """Parts taken out of the layer's area."""
    return ("clothes", "hair") if layer == "skin" else ()


def ensure_words(prompt, layers, prompt_words, exclude=()):
    """Append the part names that have no word in the prompt -> (prompt, [appended])."""
    added = []
    for layer in sorted(set(layers)):
        name = APPEND.get(layer)
        if name and not words_in(layer, prompt_words, exclude) and name not in added:
            added.append(name)
    if added:
        prompt = re.sub(r"[\s,]*$", "", prompt or "") + ", " + ", ".join(added)
    return prompt, added
