"""Examples of attention-guided (focus) profiles.

For each example a profile focused on one word is built from same-seed pairs
that differ only in that detail (positive / negative), then applied to a new
prompt that names the detail differently. Profiles are applied from the
middle of the steps (Start 0.5) so the change stays inside the focus area.

  python focus_examples.py [--url http://127.0.0.1:7860] [--maps DIR]

--maps: the folder the webui writes attention maps to (started with
SD_ACT_STEER_DUMP_MAPS=DIR); adds a column showing where the focus word landed.
"""

import argparse
import base64
import io
import os

import requests
from PIL import Image, ImageDraw, ImageFont

PORTRAIT = "1girl, solo, portrait, upper body, looking at viewer, simple background"
FULL = "1girl, solo, standing, full body, park, trees"
SCENERY = "scenery, mountains, lake, no humans, wide shot"

EXAMPLES = [
    # profile,        focus,  base,     positive,               negative,     target
    ("focus_eyes",  "eyes",  PORTRAIT, "glowing red eyes",     "blue eyes",  "brown eyes"),
    ("focus_hair",  "hair",  PORTRAIT, "silver hair",          "black hair", "brown hair"),
    ("focus_dress", "dress", FULL,     "red dress",            "blue dress", "white dress"),
    ("focus_sky",   "sky",   SCENERY,  "sunset sky, orange sky", "blue sky", "cloudy sky"),
]
SEEDS = (101, 102)
TARGET_SEED = 1234
STRENGTHS = (6.0, 10.0)
START = 0.5


def steer_args(record=False, profile=None, auto="ask", apply=(), strength=1.0, rec_focus="", start=0.0):
    # [keep, profile, buckets, auto, apply, strength, mode, cond_only, start, end, blocks, rec_focus, override]
    return [record, profile, 1, auto, list(apply), strength, "add", False, start, 1.0, "all", rec_focus, ""]


def generate(url, model, prompt, seed, args):
    payload = {
        "prompt": prompt, "negative_prompt": "lowres, worst quality", "seed": seed, "steps": 20, "cfg_scale": 6,
        "width": 832, "height": 832, "sampler_name": "Euler a",
        "override_settings": {"sd_model_checkpoint": model},
        "alwayson_scripts": {"activation steering": {"args": args}},
    }
    r = requests.post(f"{url}/sdapi/v1/txt2img", json=payload, timeout=1200)
    r.raise_for_status()
    return Image.open(io.BytesIO(base64.b64decode(r.json()["images"][0]))).convert("RGB")


def overlay(image, mask):
    """Base image darkened, focus area tinted red."""
    mask = mask.convert("L").resize(image.size)
    red = Image.new("RGB", image.size, (255, 40, 40))
    dark = Image.blend(image, Image.new("RGB", image.size, "black"), 0.55)
    return Image.composite(red, dark, mask.point(lambda v: int(v * 0.8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--model", default="prefectIllustriousXL_v8")
    ap.add_argument("--maps", default=None)
    ap.add_argument("--out", default="focus_examples_grid.png")
    a = ap.parse_args()

    rows = []
    for profile, focus, base, pos, neg, target in EXAMPLES:
        requests.post(f"{a.url}/actsteer/delete", json={"profile": profile})
        shown = []
        for seed in SEEDS:
            for impact, detail in (("positive", pos), ("negative", neg)):
                im = generate(a.url, a.model, f"{base}, {detail}", seed,
                              steer_args(record=True, profile=profile, auto=impact, rec_focus=focus))
                if seed == SEEDS[0]:
                    shown.append((f"{impact}: {detail}", im))
        prompt = f"{base}, {target}"
        # strength 0 changes nothing but still computes the focus map
        base_im = generate(a.url, a.model, prompt, TARGET_SEED, steer_args(apply=[profile], strength=0.0))
        cells = shown[:]
        map_path = os.path.join(a.maps, f"{TARGET_SEED}_{focus}.png") if a.maps else None
        if map_path and os.path.exists(map_path):
            cells.append((f"where '{focus}' is", overlay(base_im, Image.open(map_path))))
        cells.append((f"no profile: {target}", base_im))
        for s in STRENGTHS:
            im = generate(a.url, a.model, prompt, TARGET_SEED,
                          steer_args(apply=[profile], strength=s, start=START))
            cells.append((f"{profile} +{s:g}", im))
        rows.append((f"focus '{focus}'", cells))
        print(f"done: {profile}")

    cell, label, title = 256, 22, 26
    cols = max(len(c) for _, c in rows)
    grid = Image.new("RGB", (cols * cell, len(rows) * (title + cell + label)), "white")
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = bold = ImageFont.load_default()
    for r, (name, cells) in enumerate(rows):
        y = r * (title + cell + label)
        draw.text((6, y + 5), name, fill="black", font=bold)
        for i, (text, im) in enumerate(cells):
            grid.paste(im.resize((cell, cell)), (i * cell, y + title))
            draw.text((i * cell + 4, y + title + cell + 3), text, fill="black", font=font)
    grid.save(a.out)
    print(f"grid saved to {a.out}")


if __name__ == "__main__":
    main()
