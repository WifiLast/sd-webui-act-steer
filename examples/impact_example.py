"""Simple example of positive and negative impact.

Builds the profile "example_mood":
  positive: 2 portraits with "smiling, happy"
  negative: 2 portraits with "crying, sad", same seeds as the positives
then renders the same plain portrait (same seed) at strengths -3 .. +3 and
writes a labelled grid. Needs a webui started with --api.

What to expect: + makes the portrait warm and bright like the happy images,
- makes it cold and dark like the sad ones. The expression itself barely
changes: a profile stores one value per channel for the whole picture, so it
carries the overall look (colour, light, mood, style), not where things are
(a mouth shape). Positives and negatives use the same seeds so nothing else
(hair colour of whichever image happened to be liked) leaks in.

  python impact_example.py [--url http://127.0.0.1:7860] [--model prefectIllustriousXL_v8]
"""

import argparse
import base64
import io

import requests
from PIL import Image, ImageDraw, ImageFont

PROFILE = "example_mood"
BASE = "1girl, solo, portrait, upper body, looking at viewer, simple background"
NEG_PROMPT = "lowres, worst quality"
STRENGTHS = [-3.0, -1.5, 0.0, 1.5, 3.0]


def steer_args(record=False, profile=None, auto="ask", apply=(), strength=1.0):
    # [keep, profile, buckets, auto, apply profiles, strength, mode, cond_only, start, end, blocks]
    # 1 time bucket: the profile can also be baked into a checkpoint exactly
    return [record, profile, 1, auto, list(apply), strength, "add", False, 0.0, 1.0, "all"]


def generate(url, model, prompt, seed, args):
    payload = {
        "prompt": prompt, "negative_prompt": NEG_PROMPT, "seed": seed, "steps": 20, "cfg_scale": 6,
        "width": 832, "height": 832, "sampler_name": "Euler a",
        "override_settings": {"sd_model_checkpoint": model},
        "alwayson_scripts": {"activation steering": {"args": args}},
    }
    r = requests.post(f"{url}/sdapi/v1/txt2img", json=payload, timeout=1200)
    r.raise_for_status()
    return Image.open(io.BytesIO(base64.b64decode(r.json()["images"][0]))).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--model", default="prefectIllustriousXL_v8")
    ap.add_argument("--out", default="impact_example_grid.png")
    a = ap.parse_args()

    requests.post(f"{a.url}/actsteer/delete", json={"profile": PROFILE})  # start clean

    examples = [("positive", f"{BASE}, smiling, happy, open mouth", 101),
                ("negative", f"{BASE}, crying, sad, tears", 101),
                ("positive", f"{BASE}, smiling, happy, open mouth", 102),
                ("negative", f"{BASE}, crying, sad, tears", 102)]
    sources = []
    for impact, prompt, seed in examples:
        im = generate(a.url, a.model, prompt, seed, steer_args(record=True, profile=PROFILE, auto=impact))
        sources.append((f"{impact}: {prompt[len(BASE) + 2:]}", im))
        print(f"merged as {impact}: {prompt}")

    results = []
    for s in STRENGTHS:
        args = steer_args(apply=[PROFILE], strength=s) if s else steer_args()
        results.append((f"strength {s:+g}" if s else "no profile", generate(a.url, a.model, BASE, 1234, args)))
        print(f"rendered strength {s:+g}")

    cell, pad = 320, 34
    cols = max(len(sources), len(results))
    grid = Image.new("RGB", (cols * cell, 2 * (cell + pad) + pad), "white")
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), f"1) images merged into profile '{PROFILE}'", fill="black", font=font)
    for i, (label, im) in enumerate(sources):
        grid.paste(im.resize((cell, cell)), (i * cell, pad))
        draw.text((i * cell + 6, pad + cell + 6), label, fill="black", font=font)
    y = 2 * pad + cell + pad
    draw.text((8, y - pad + 6), f"2) '{BASE[:40]}...', same seed, profile applied", fill="black", font=font)
    for i, (label, im) in enumerate(results):
        grid.paste(im.resize((cell, cell)), (i * cell, y))
        draw.text((i * cell + 6, y + cell - 26), label, fill="white", font=font, stroke_width=2,
                  stroke_fill="black")
    grid.save(a.out)
    print(f"grid saved to {a.out}")


if __name__ == "__main__":
    main()
