"""Fix Wrong Areas: paint over parts of a generated image that came out wrong;
the activations that put them there are reduced in that area and the image
is rendered again with the same seed.

Pass 1 renders the image again (same seed) and measures, in every
cross-attention layer, how much the painted area attends to each prompt
word compared to the rest of the image. Pass 2..n render it once per
strength with a correction inside the mask only (lib_act_steer.core):
  remove responsible words: the words the area attends to most (or the ones
      typed in) get their attention scaled by (1 - strength) there
  weaken prompt in area: classifier-free guidance scaled by (1 - strength)
      there, so the area follows the prompt less
"""

import base64
import copy
import importlib
import io
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import processing, scripts
from modules.processing import Processed


def _reload_if_changed(packages, root):
    """Settings -> Reload UI runs the script files again but keeps imported packages: import them
    again when one of their files changed (once per reload: the next script finds the new stamp)."""
    files = [f for pkg in packages for f in (root / pkg).rglob("*.py")]
    stamp = max((f.stat().st_mtime for f in files), default=0.0)
    loaded = sys.modules.get(packages[0])
    if loaded is not None and getattr(loaded, "_reload_stamp", None) != stamp:
        for name in [n for n in sys.modules if n.split(".")[0] in packages]:
            del sys.modules[name]
        importlib.invalidate_caches()  # new files too
        print(f"[{root.name}] changed files: {', '.join(packages)} imported again")
    importlib.import_module(packages[0])._reload_stamp = stamp


_reload_if_changed(["lib_act_steer"], Path(__file__).resolve().parent.parent)
from lib_act_steer import core
from lib_act_steer.core import state
from lib_act_steer.text import focus_positions, word_positions

# tab -> (seed, subseed, prompt) of the image loaded into the editor
fix_source = {}


def _tab(p):
    return "img2img" if isinstance(p, processing.StableDiffusionProcessingImg2Img) else "txt2img"


def read_mask(editor):
    """Editor value (gradio sketch dict), PIL image or base64 PNG -> [H, W] float tensor, 1 = wrong area."""
    if editor is None:
        return None
    if isinstance(editor, str):
        if not editor.strip():
            return None
        img = Image.open(io.BytesIO(base64.b64decode(editor.split(",")[-1])))
    elif isinstance(editor, dict):
        img = editor.get("mask")
    else:
        img = editor
    if img is None:
        return None
    if img.mode in ("RGBA", "LA"):
        alpha = np.array(img.getchannel("A"), dtype=np.float32)
        lum = np.array(img.convert("L"), dtype=np.float32)
        # strokes are white; a fully opaque black canvas means nothing painted
        arr = lum if alpha.min() > 0 else np.minimum(lum, alpha)
    else:
        arr = np.array(img.convert("L"), dtype=np.float32)
    t = torch.from_numpy(arr / 255.0)
    return t if float(t.max()) > 0.05 else None


def parse_strengths(text):
    out = []
    for part in (text or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out or [1.0]


class FixAreasScript(scripts.Script):
    def title(self):
        return "Fix Wrong Areas"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        gr.Markdown("1. Generate an image. 2. **Load** it here (select it in the gallery first when there are "
                    "several) and **paint over the wrong areas**. 3. Press Generate: the image is rendered "
                    "again with the same seed, the prompt words that put something into the painted area are "
                    "found, and it is rendered once per strength with them removed there. Keep prompt and "
                    "settings unchanged.")
        with gr.Row(equal_height=True):
            load = gr.Button("⬅ Load last image", variant="primary", size="sm", scale=0, min_width=200)
            with gr.Column(scale=1):
                status = gr.Markdown("")
        gallery_idx = gr.Number(value=-1, visible=False)
        editor = gr.Image(label="Paint the wrong areas", source="upload", tool="sketch", type="pil",
                          interactive=True, height=512, elem_id=self.elem_id("editor"))
        mode = gr.Radio(list(core.FIX_MODES), value=core.FIX_MODES[0], label="Correction",
                        elem_id=self.elem_id("mode"))
        with gr.Row():
            words = gr.Textbox(label="Words to remove", placeholder="empty: found automatically",
                               elem_id=self.elem_id("words"))
            strengths = gr.Textbox(label="Strengths (one image each; 1 = word gone, more = pushed away)",
                                   value="1, 2, 3",
                                   elem_id=self.elem_id("strengths"))
        with gr.Row():
            start = gr.Slider(0.0, 1.0, value=0.15, step=0.05,
                              label="Start (0 = may also re-compose the image; later = only the area changes)",
                              elem_id=self.elem_id("start"))
            feather = gr.Slider(0, 6, value=2, step=1, label="Feather", elem_id=self.elem_id("feather"))
        with gr.Row():
            use_last_seed = gr.Checkbox(label="Use the loaded image's seed", value=True,
                                        elem_id=self.elem_id("use_last_seed"))
            include_original = gr.Checkbox(label="Show re-rendered original", value=True,
                                           elem_id=self.elem_id("include_original"))

        def do_load(g):
            last = core.last_images.get(tab)
            if not last or not last["items"]:
                return gr.update(), "⚠️ nothing generated on this tab yet"
            g = int(g if g is not None else -1)
            i = g - last["first"] if g >= last["first"] else 0
            i = min(max(i, 0), len(last["items"]) - 1)
            im, seed, subseed = last["items"][i]
            fix_source[tab] = (seed, subseed, last["prompt"])
            return im, f"loaded image {i} (seed {seed}): paint the wrong areas, then Generate"

        load.click(do_load, _js="(g) => [selected_gallery_index()]", inputs=[gallery_idx], outputs=[editor, status])
        return [editor, mode, words, strengths, start, feather, use_last_seed, include_original]

    def run(self, p, editor=None, mode=core.FIX_MODES[0], words="", strengths="1, 2, 3", start=0.15,
            feather=2, use_last_seed=True, include_original=True, *args):
        mask = read_mask(editor)
        if mask is None:
            raise ValueError("Fix Wrong Areas: paint the wrong areas on the image first")
        tab = _tab(p)
        source = fix_source.get(tab)
        if use_last_seed and source:
            p.seed, p.subseed = source[0], source[1]
            if source[2] != p.prompt:
                print("[Fix Wrong Areas] the prompt differs from the loaded image's; the re-render won't match it")
        p.seed = processing.get_fixed_seed(p.seed)
        p.subseed = processing.get_fixed_seed(p.subseed)
        p.n_iter, p.batch_size = 1, 1
        if abs(mask.shape[1] / mask.shape[0] - p.width / p.height) > 0.02:
            print("[Fix Wrong Areas] the painted image has a different aspect ratio than the generation")

        def render(fix, label=None):
            pc = copy.copy(p)
            pc.extra_generation_params = dict(p.extra_generation_params)
            if label:
                pc.extra_generation_params["Area fix"] = label
            state.set_fix(fix)
            try:
                return processing.process_images(pc)
            finally:
                state.set_fix(None)

        record = {"kind": "record", "mask": mask, "feather": feather}
        first = render(record)

        positions, label = [], mode
        if mode == core.FIX_MODES[0]:
            all_pos, text = word_positions(p.prompt, p.steps)
            if words and words.strip():
                positions = [int(t) for t in focus_positions(p.prompt, words, p.steps)]
                if not positions:
                    raise ValueError(f"Fix Wrong Areas: '{words}' is not in the prompt")
            else:
                tok_in = record["tok_in"] / record["tok_in_n"] if "tok_in" in record else None
                tok_out = record["tok_out"] / record["tok_out_n"] if "tok_out" in record else None
                positions = core.pick_words(tok_in, tok_out, all_pos.to(tok_in.device) if tok_in is not None
                                            else all_pos)
                if not positions:
                    raise RuntimeError("Fix Wrong Areas: no prompt word stands out in the painted area; "
                                       "type the words or use 'weaken prompt in area'")
            found = ", ".join(text.get(t, str(t)) for t in positions)
            print(f"[Fix Wrong Areas] removing in the painted area: {found}")
            label = f"remove words [{found}]"

        images, infotexts = [], []
        if include_original:
            images += first.images[:1]
            infotexts += first.infotexts[:1]
        for s in parse_strengths(strengths):
            fix = {"kind": "words" if mode == core.FIX_MODES[0] else "guidance", "positions": positions,
                   "w": s, "start": float(start), "mask": mask, "feather": feather}
            res = render(fix, label=f"{label}, strength {s:g}, start {float(start):g}")
            images += res.images[:1]
            infotexts += res.infotexts[:1]

        n = len(images)
        return Processed(p, images, p.seed, infotexts[0] if infotexts else "", subseed=p.subseed,
                         all_prompts=[p.prompt] * n, all_negative_prompts=[p.negative_prompt] * n,
                         all_seeds=[p.seed] * n, all_subseeds=[p.subseed] * n, infotexts=infotexts)
