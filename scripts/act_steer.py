import importlib
import json
import os
import queue
import re
import sys
import threading
from pathlib import Path

import gradio as gr

from modules import extra_networks, processing, script_callbacks, scripts, shared
from modules.ui_components import FormRow, ToolButton


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
from lib_act_steer import adherence, core, evaluate, learn, loratune, negative, order, promptedit, quality, \
    rules as rules_mod, seglayers, septrain, timing, wordmap
from lib_act_steer.text import focus_positions, tokenizer as text_tokenizer, word_groups
from lib_act_steer.core import state

AUTO_CHOICES = ["ask", "positive", "negative"]


def _model_meta():
    model = shared.sd_model
    info = getattr(model, "sd_checkpoint_info", None)
    return {
        "model": getattr(info, "name_for_extra", "") or "",
        "model_hash": getattr(info, "shorthash", "") or "",
        "arch": "sdxl" if getattr(model, "is_sdxl", False) else "sd1",
    }


_warned = set()


def _check_model(steer):
    want = steer.meta.get("model_hash") or steer.meta.get("model")
    have = _model_meta()
    if steer.meta.get("arch") and steer.meta["arch"] != have["arch"]:
        print(f"[ActSteer] '{steer.name}' was recorded on {steer.meta['arch']}, current model is {have['arch']}: "
              "channel counts won't match, most blocks will be skipped")
    elif want and want not in (have["model_hash"], have["model"]) and steer.name not in _warned:
        _warned.add(steer.name)
        print(f"[ActSteer] '{steer.name}' was recorded on {steer.meta.get('model')}, using it on {have['model']}")


def clean_focus(text):
    """'Eyes , blue eyes' -> 'Eyes, blue eyes' (one canonical spelling per profile)."""
    return ", ".join(w.strip() for w in (text or "").split(",") if w.strip())


def _tab(p):
    return "img2img" if isinstance(p, processing.StableDiffusionProcessingImg2Img) else "txt2img"


def merge(tab, profile, impact, image=-1, weight=1.0, create=False):
    """Merge image `image` (0-based, -1 = all) of the last recorded job on `tab` into a profile.

    impact: "positive" | "negative". create=True makes a new profile and refuses an existing name.
    Returns a status line.
    """
    rec = core.last_recording.get(tab)
    if not rec:
        raise ValueError(f"no image influence kept on {tab}: tick 'Keep influence' and generate first "
                         "(or it was skipped)")
    side = {"positive": "pos", "pos": "pos", "+": "pos", "negative": "neg", "neg": "neg", "-": "neg"}.get(impact)
    if side is None:
        raise ValueError("impact must be 'positive' or 'negative'")
    if not (profile or "").strip():
        if not create:
            raise ValueError("pick a profile to merge into, or create a new one")
        profile = rec["default_name"]
    name = core.safe_name(profile)
    imgs = sorted(rec["samples"]) if image is None or image < 0 else [image]
    missing = [i for i in imgs if i not in rec["samples"]]
    if missing:
        raise ValueError(f"image {missing[0]} not in the last job (has {len(rec['samples'])} images, 0-based)")
    done = rec["rated"].get(name, {})
    again = [i for i in imgs if i in done]
    if again:
        raise ValueError(f"image {again[0]} was already merged into '{name}' as {done[again[0]]}")
    name, pos_n, neg_n = core.add_samples(
        name, side, [rec["samples"][i] for i in imgs], rec["meta"], rec["buckets"],
        prompts=[rec["prompts"].get(i, "") for i in imgs], weight=weight, create=create)
    rec["rated"].setdefault(name, {}).update({i: impact for i in imgs})
    _last_merge[name] = (tab, imgs)
    which = "all images" if len(imgs) > 1 else f"image {imgs[0]}"
    verb = "created" if create else "merged into"
    focus = rec["meta"].get("focus")
    return (f"{which} {verb} **{name}**{f' (focus: {focus})' if focus else ''} as {impact} (weight {float(weight):g}). "
            f"Profile now has {pos_n} positive / {neg_n} negative.")


# profile -> (tab, images) of its last merge, so undo makes them mergeable again
_last_merge = {}


def skip(tab):
    rec = core.last_recording.pop(tab, None)
    if not rec:
        return "nothing to skip"
    return f"skipped: influence of {len(rec['samples'])} image(s) discarded"


def undo(profile):
    name = core.safe_name(profile or "")
    msg = core.undo_last(name)
    tab_imgs = _last_merge.pop(name, None)
    if tab_imgs:
        rec = core.last_recording.get(tab_imgs[0])
        if rec:
            for i in tab_imgs[1]:
                rec["rated"].get(name, {}).pop(i, None)
    return msg


def _dump_maps(directory, p):
    """Debug: save the largest attention map per image and focus as a greyscale PNG."""
    from PIL import Image
    os.makedirs(directory, exist_ok=True)
    for img, per_focus in state.maps.items():
        for focus in per_focus:
            m = state.norm_map(img, focus)
            arr = (m.float().clamp(0, 1) * 255).byte().cpu().numpy()
            name = f"{p.seed + img}_{core.safe_name(focus)}.png"
            Image.fromarray(arr, "L").resize((p.width, p.height), Image.NEAREST).save(os.path.join(directory, name))


class ExtraNetworkSteer(extra_networks.ExtraNetwork):
    """<steer:name:strength> in the prompt."""

    def __init__(self):
        super().__init__("steer")

    def activate(self, p, params_list):
        """<steer:name:strength> or <steer:name:strength:focus words> (focus overrides the profile's)."""
        steers, focus = [], {}
        for params in params_list:
            if not params.positional:
                continue
            name = params.positional[0]
            w = float(params.positional[1]) if len(params.positional) > 1 else 1.0
            if len(params.positional) > 2:
                focus[name] = clean_focus(params.positional[2])
            try:
                s = core.load_steer(name)
            except Exception as e:
                print(f"[ActSteer] {e}")
                continue
            _check_model(s)
            steers.append((s, w))
        state.prompt_steers = steers
        state.prompt_focus = focus

    def deactivate(self, p):
        state.prompt_steers = []
        state.prompt_focus = {}


def parse_edits(text):
    """'hat:0, red dress:1.5' -> [("hat", 0.0), ("red dress", 1.5)] (factor 1 entries dropped).

    Factor: 1 normal, 0 the word is ignored, < 0 pushed away, > 1 strengthened."""
    out = []
    for part in (text or "").split(","):
        if ":" not in part:
            continue
        word, _, val = part.rpartition(":")
        word = word.strip()
        try:
            factor = min(5.0, max(-1.0, float(val)))  # below -1 images break (tested: hat:-2)
        except ValueError:
            continue
        if word and abs(factor - 1.0) > 1e-6:
            out.append((word, factor))
    return out


def format_edits(edits):
    return ", ".join(f"{w}:{f:g}" for w, f in edits)


def _word_tokens(prompts, steps, prefix):
    """Word groups of each prompt -> ({key: [positions per prompt]}, [[(key or None, text)] per prompt])."""
    groups = [word_groups(pr, steps) for pr in prompts]
    tokens, texts = {}, []
    for i, words in enumerate(groups):
        row = []
        for n, (text, pos) in enumerate(words):
            key = f"{prefix}{n}" if pos is not None else None
            if key is not None:
                tokens.setdefault(key, [None] * len(groups))[i] = pos
            row.append((key, text))
        texts.append(row)
    return tokens, texts


NEW_PROMPT_SIM = 0.5  # word overlap (Jaccard) of two prompts below which they are different images


def is_new_prompt(old, new):
    """True when `new` is a prompt for another image, not an edit of `old` (Optimize, rephrasing and
    hand edits change a few words; a new prompt shares less than NEW_PROMPT_SIM of its words)."""
    a, b = promptedit.prompt_words(old or ""), promptedit.prompt_words(new or "")
    if not a or not b:
        return False
    return len(a & b) / len(a | b) < NEW_PROMPT_SIM


def learning(tab):
    """Optimize / auto record experiments (checkbox "Learn from each optimisation")."""
    return learn_settings.get(tab, True)


_eval_queue = queue.Queue()
_eval_thread = None


def _eval_worker():
    while True:
        exp_id, side, image, infotext = _eval_queue.get()
        try:
            ev = evaluate.evaluate(image, infotext, tag=f"learn_{side}")
            # only scores of the learned models are comparable (quality None without them)
            learn.set_quality(exp_id, side, evaluate.quality(ev), evaluate.adherence(ev))
        except Exception as e:
            print(f"[ActSteer] learning evaluation: {type(e).__name__}: {e}")


def evaluate_later(exp_id, side, image, infotext):
    """SD MCP quality score of an experiment's image, in the background: one worker thread, one
    evaluation at a time (several at once only cost RAM)."""
    global _eval_thread
    _eval_queue.put((exp_id, side, image, infotext))
    if _eval_thread is None or not _eval_thread.is_alive():
        _eval_thread = threading.Thread(target=_eval_worker, daemon=True, name="act-steer-learn-eval")
        _eval_thread.start()


def apply_learned(prompt, neg, limit=6, guard=()):
    """Changes with a good record (learn.proven) applied to a new prompt -> (prompt, negative, [what]).
    guard: words that are only weakened / strengthened by GUARD_BAND at most, never removed, rephrased,
    replaced or delayed."""
    st = learn.stats(model_name())
    if not st:
        return prompt, neg, []
    words = promptedit.prompt_words(prompt) | promptedit.prompt_phrases(prompt)
    neg_phrases = {negative.phrase_key(x) for x in promptedit._split(neg or "") if x.strip()}
    R = rules_mod.current().for_prompt(prompt)
    applied = []
    for s in learn.proven(st, words, neg_phrases):
        if len(applied) >= limit:
            break
        w, kind, d = s["word"], s["kind"], s["detail"]
        if R.is_kept(w) or any(R._match(x, w) for x in R.neg_kept()):
            continue
        if guard and not kind.startswith("neg_") and set(w.split()) & set(guard):
            if kind not in ("weaken", "strengthen", "boost_missed"):
                continue
            b = promptedit.prompt_weights(prompt, {w}).get(w, 1.0)
            d = round(min(max(float(d), b * (1 - GUARD_BAND)), b * (1 + GUARD_BAND)), 2)
        if kind in ("weaken", "strengthen", "remove", "boost_missed"):
            new = promptedit.apply_to_prompt(prompt, {w: 0.0 if kind == "remove" else float(d)})
        elif kind == "delay":
            new = promptedit.delay_piece(prompt, w, float(d or timing.DELAY))[0]
        elif kind in ("rephrase", "replace"):
            new = promptedit.dedupe_phrases(promptedit.replace_text(prompt, w, d)[0]) if d else prompt
        elif kind == "neg_add":
            neg = negative.add_required(neg, [(w, float(d) if d else None)])[0]
            applied.append(f"negative + {w}")
            continue
        elif kind in ("neg_remove", "neg_weaken", "neg_boost"):
            neg, changed = negative.set_phrase(neg, w, None if kind == "neg_remove" else float(d or 1.0))
            if changed:
                applied.append(f"negative {w} " + ("removed" if kind == "neg_remove" else f"→ {float(d):g}"))
            continue
        else:
            continue
        if new != prompt:
            prompt = new
            applied.append(f"{w} " + ("removed" if kind == "remove" else f"delayed to {float(d or timing.DELAY):g}"
                                      if kind == "delay" else f"→ {d:g}" if isinstance(d, (int, float))
                                      else f"→ {d}"))
    return prompt, neg, applied


RATING_SCORE = {1: 8.0, -1: 3.0}  # 👍 / 👎 as SD MCP ratings (0-10)


def send_rating(image, infotext, value, tab):
    """Your rating to the SD MCP too (rate_image: feeds token_quality_report / token_impact); in a thread."""
    if not evaluate.ENABLED:
        return

    def run():
        try:
            path = evaluate.save_for_eval(image, infotext, f"{tab}_rated")
            out = evaluate.call_tool("rate_image", {"image": path, "rating": RATING_SCORE[value],
                                                    "note": "act-steer " + ("👍" if value > 0 else "👎")}, timeout=300)
            if not out.get("ok", True):
                print(f"[ActSteer] SD MCP rating not stored: {out.get('reason') or out}")
        except Exception as e:
            print(f"[ActSteer] SD MCP not reachable for the rating ({type(e).__name__}: {e})")
    threading.Thread(target=run, daemon=True, name="act-steer-rate").start()


def model_name():
    """The checkpoint name as the SD MCP records it (infotext "Model")."""
    try:
        return shared.sd_model.sd_checkpoint_info.model_name or ""
    except Exception:
        return ""


def model_hint():
    """Checkpoint name and base-model metadata (for the rephrasing vocabulary: Illustrious, Pony ...)."""
    try:
        info = shared.sd_model.sd_checkpoint_info
        meta = getattr(info, "metadata", None) or {}
        keys = ("modelspec.architecture", "modelspec.title", "ss_base_model_version", "ss_sd_model_name")
        return " ".join([info.name or "", info.filename or ""] + [str(meta.get(k, "")) for k in keys])
    except Exception:
        return ""


_unsupported_warned = set()


def embedding_names():
    """Loaded textual inversion embeddings (never removed from a negative prompt)."""
    try:
        from modules import sd_hijack
        return set(sd_hijack.model_hijack.embedding_db.word_embeddings)
    except Exception:
        return set()


LAYER_TARGETS = ("background", "foreground", "left", "right", "top", "bottom", "center") + seglayers.NAMES


def parse_layers(text):
    """'sea -> background, hat -> on girl, cat -> right:0.6' -> [(words, target, strength)].
    Targets: background, foreground, left, right, top, bottom, center, on <word>, away from <word>, and the
    parts of the figure skin, clothes (cloth, clothing, outfit), hair, face, body (person)."""
    out = []
    for part in re.split(r"[,\n]", text or ""):
        if "->" not in part:
            continue
        words, _, target = part.partition("->")
        target, strength = target.strip().lower(), 0.85
        m = re.match(r"(.*?):\s*([\d.]+)$", target)
        if m:
            target = m.group(1).strip()
            try:
                strength = min(1.0, max(0.0, float(m.group(2))))
            except ValueError:
                pass
        words = words.strip()
        if words and (target in LAYER_TARGETS or re.match(r"(on|away from) \S", target)):
            out.append((words, target, strength))
    return out


def _take_out(prompt, words):
    """Take `words` out of the prompt for a region of its own -> (prompt, moved text). A comma phrase made of
    just these words, or a short one (up to 3 words) ending in them, moves whole with its weight ("blue sea"
    for sea, "(red lantern:1.2)"); inside a longer phrase only the words go. <lora:...> tags stay."""
    want = [w.lower() for w in re.findall(r"\w+", words)]
    if not want:
        return prompt, ""
    rx = re.compile(r"(?<![\w])" + r"\W+".join(map(re.escape, want)) + r"(?![\w])", re.I)
    out, moved = [], []
    for piece in promptedit._split(prompt):
        if piece.strip().startswith("<") or not rx.search(piece):
            out.append(piece)
            continue
        toks = re.findall(r"[A-Za-z][\w']*", promptedit._WEIGHT_NUM.sub(" ", piece))
        if promptedit.prompt_words(piece) <= set(want) or (len(toks) <= 3 and toks[-1].lower() == want[-1]):
            moved.append(piece.strip())
            continue
        out.append(re.sub(r"\s+:", ":", rx.sub("", piece, count=1)))
        moved.append(words.strip())
    return promptedit._tidy(", ".join(p.strip() for p in out if p.strip())), ", ".join(moved)


def format_layers(entries):
    return ", ".join(f"{w} -> {t}" + (f":{s:g}" if abs(s - 0.85) > 1e-6 else "") for w, t, s in entries)


def split_words(text):
    return [w.strip() for w in (text or "").split(",") if w.strip()]


# tab -> index of the image whose word map is shown (its seed is reused for word edits)
wordmap_shown = {}
# tab -> words taken out of the prompt from the word map menu (struck through until the next job)
removed_words = {}
# tab -> word map strength filter: None (all words), ("top", n) or ("below", fraction of the attention)
view_filter = {}
# tab -> number of finished jobs with a word map; javascript/act_steer.js polls it to refresh the view
wordmap_version = {}
# learning (lib_act_steer/learn.py): tab -> checkbox state; the changes of the current Optimize /
# auto step; (wordmap_version, experiment id) of the last finished experiment; learned changes
# applied to the last new prompt (shown after its job)
learn_settings = {}
step_changes = {}
last_finished = {}
learned_applied = {}
# (tab, wordmap_version, image) -> quality.word_quality (the view and Optimize share it)
quality_cache = {}
# tab -> True when the last job started a new prompt (before_process); the view then clears the old
# word edits and locked words
fresh_prompt = {}
# tab -> (wordmap_version, evaluate_image result) of the last evaluation (its defect tags feed the
# negative prompt check)
last_eval = {}
# tab -> {word: [rephrase suggestions already used]} (Optimize / auto try the next one)
rephrase_tried = {}
QUALITY_DROP = 0.08  # auto optimize stops when the quality score falls this far below round 0
GUARD_BAND = 0.15  # the N most important words (auto-lock slider) move at most this much, never removed
ADHERENCE_IN_BEST = 0.3  # weight of prompt adherence next to the quality score when auto picks its best round
# tab -> running auto optimize: {"round", "max", "thr", "to_prompt", "history": [{conflict, pair, prompt, edits}]}
auto_runs = {}
# tab -> True after Optimize: the next generation keeps that seed even without word edits
keep_seed = {}


def train_separation(status, n=40, steps=600, rank=4, keep_w=septrain.KEEP_WEIGHT, lines="", use_cur=False,
                     name="act_steer_separation", prompt=""):
    """Separation LoRA (lib_act_steer/septrain.py): generate n images, train, save. Writes its progress and
    result into status["md"] (markdown), the images into status["images"], status["done"] at the end, and
    returns status (status["path"] = the saved LoRA)."""
    from modules import call_queue

    def gen_progress(i, total):
        status["md"] = f"Generating training images: {i}/{total}"

    def train_progress(step, total, sep, kl, secs):
        status["md"] = f"Training: step {step}/{total}, overlap loss {sep:.3f}, keep (KL) {kl:.4f}, {secs:.0f} s"

    with call_queue.queue_lock:
        shared.state.begin(job="act_steer_train")
        try:
            net, _ = core.find_net(shared.sd_model)
            size = 1024 if getattr(shared.sd_model, "is_sdxl", False) or (net is not None and core.is_dit(net)) else 512
            prompts = septrain.make_prompts(lines, n, prompt or "", use_cur)
            status["md"] = f"Generating training images: 0/{len(prompts)}"
            septrain.check_ram("before generating")
            samples, images, n_images = septrain.generate(prompts, size, size, progress=gen_progress)
            status["images"] = images
            if shared.state.interrupted:
                status["md"] = "Stopped while generating; nothing trained."
                return status
            if not samples:
                status["md"] = ("No training data: the prompts need two nouns in different phrases (e.g. "
                                "`a red cat and a blue dog`).")
                return status
            status["md"] = f"Training on {len(samples)} steps of {n_images} images …"
            lora, res = septrain.train(samples, steps=int(steps), rank=int(rank), keep_weight=float(keep_w),
                                       progress=train_progress)
            n_samples = len(samples)
            del samples
            status["result"] = res
            if not res["steps"]:
                status["md"] = "Stopped before the first training step; nothing saved."
                return status
            path = septrain.save(lora, name or "act_steer_separation", {
                "ss_network_module": "networks.lora", "ss_network_dim": int(rank), "ss_network_alpha": int(rank),
                "act_steer_kind": "separation", "act_steer_images": n_images, "act_steer_steps": res["steps"],
                "act_steer_overlap_before": round(res["before"], 4), "act_steer_overlap_after": round(res["after"], 4)})
            status["path"] = path
            lname = os.path.splitext(os.path.basename(path))[0]
            early = ", stopped early: low RAM" if res["low_memory"] else ", stopped early" if res["steps"] < int(steps) else ""
            status["md"] = (f"✅ Saved `{path}` ({res['steps']} steps on {n_images} images{early}).\n\n"
                            f"Worst noun overlap on {min(septrain.EVAL_SAMPLES, n_samples)} training steps: "
                            f"**{res['before']:.3f}** without the LoRA, **{res['after']:.3f}** with it. "
                            f"Use it as `<lora:{lname}:1>`.")
        except septrain.LowMemory as e:
            status["md"] = f"⚠️ Training stopped: {e}. Nothing saved."
        except Exception as e:
            import traceback
            traceback.print_exc()
            status["md"] = f"❌ Training failed: {type(e).__name__}: {e}"
        finally:
            samples = None
            septrain.free_gpu()
            shared.state.end()
            status["done"] = True
    return status


class ActSteerScript(scripts.Script):
    def title(self):
        return "Activation Steering"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        with gr.Accordion(self.title(), open=False):
            view = gr.HTML("<p>Generate an image: its prompt words and the areas they influenced appear here. "
                           "Click a word to edit its influence.</p>")
            with gr.Row(equal_height=True):
                show_mode = gr.Radio(["all words", "top N", "below threshold"], value="all words", scale=2,
                                     min_width=260, label="Show words by influence (share of the attention)",
                                     elem_id=self.elem_id("show_mode"))
                top_n = gr.Number(value=10, precision=0, minimum=1, label="N", scale=0, min_width=80,
                                  elem_id=self.elem_id("top_n"))
                strength_thr = gr.Slider(0.05, 5.0, value=0.5, step=0.05, scale=2,
                                         label="Threshold (% of the attention)", elem_id=self.elem_id("strength_thr"))
                remove_weak = gr.Button("🗑 Remove words below the threshold from the prompt", size="sm", scale=1,
                                        min_width=200)
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, min_width=200):
                    selected_md = gr.Markdown("Click a word in the map above.")
                with gr.Column(scale=3, min_width=260):
                    with FormRow():
                        influence = gr.Slider(-1.0, 3.0, value=1.0, step=0.05,
                                              label="Influence of the selected word (1 = normal, 2 = double, "
                                                    "0 = ignore, below 0 = push away)",
                                              elem_id=self.elem_id("influence"))
                        lock_btn = ToolButton("\U0001f512")
            with gr.Row(equal_height=True):
                layers = gr.Textbox(label="Layers: move prompt words to a part of the image (right click a word: 📍 "
                                          "Move to; next generation keeps the seed)",
                                    placeholder="sea -> background, lantern -> left, hat -> on girl, cloud -> away from "
                                                "girl, cat -> right:0.6, lace -> clothes, glossy -> skin",
                                    elem_id=self.elem_id("layers"), scale=4)
                rule_layers = gr.Checkbox(label="Placement rules (📌 place … -> … in Rules, every generation)",
                                          value=True, scale=1, min_width=180, elem_id=self.elem_id("rule_layers"))
                rp_layers = gr.Checkbox(label="As Regional Prompter regions (BREAK): background / foreground / on …",
                                        value=True, scale=1, min_width=180, elem_id=self.elem_id("rp_layers"))
            with gr.Row():
                edits = gr.Textbox(label="Word edits (next generation keeps the seed of the shown image)",
                                   placeholder="hat:0, dress:2", elem_id=self.elem_id("edits"), scale=3)
                locked = gr.Textbox(label="Locked words (Optimize leaves them as they are; 🔒 adds the selected one)",
                                    placeholder="woman, red dress", elem_id=self.elem_id("locked"), scale=2)
                with gr.Column(scale=0, min_width=200):
                    to_prompt = gr.Checkbox(label="Optimize writes into the prompt", value=True,
                                            elem_id=self.elem_id("to_prompt"))
                    auto_clear = gr.Checkbox(label="Clear word edits and locked words for a new prompt", value=True,
                                             elem_id=self.elem_id("auto_clear"))
                    auto_lock = gr.Slider(0, 10, value=3, step=1, elem_id=self.elem_id("auto_lock"),
                                          label="Guard the N most important words (Optimize changes them only "
                                                f"by ±{GUARD_BAND:.0%}, never removes them)")
            with gr.Row():
                optimize_btn = gr.Button("✨ Optimize & regenerate", variant="primary", size="sm", min_width=160)
                learned_btn = gr.Button("🎓 Optimize prompt with what was learned", size="sm", min_width=160)
                regen = gr.Button("▶ Regenerate with edits", size="sm", min_width=160)
                show = gr.Button("🔄 Word map of selected image", size="sm", min_width=160,
                                 elem_id=f"act_steer_{tab}_show")
                clear = gr.Button("✖ Clear edits", size="sm", min_width=120)
                eval_btn = gr.Button("📊 Evaluate", size="sm", min_width=120)
                up_btn = gr.Button("👍", size="sm", min_width=50, scale=0)
                down_btn = gr.Button("👎", size="sm", min_width=50, scale=0)
                reset_btn = gr.Button("🧹 New prompt", size="sm", min_width=120)
            with gr.Row():
                auto_btn = gr.Button("🔁 Auto optimize", size="sm", min_width=160, scale=1)
                auto_thr = gr.Slider(0.05, 0.6, value=0.3, step=0.01, scale=2,
                                     label="Auto: stop when the worst word conflict (shared area) is below",
                                     elem_id=self.elem_id("auto_thr"))
                auto_rounds = gr.Slider(1, 500, value=5, step=1, label="Auto: max rounds", scale=1,
                                        elem_id=self.elem_id("auto_rounds"))
                auto_eval = gr.Checkbox(label="Auto: evaluate each round (SD MCP quality + prompt adherence)",
                                        value=True, scale=1, min_width=200, elem_id=self.elem_id("auto_eval"))
            with gr.Row():
                remove_thr = gr.Slider(0.3, 1.0, value=0.6, step=0.01, scale=2,
                                       label="Remove or rephrase a word when its overlap is at least (1 = never)",
                                       elem_id=self.elem_id("remove_thr"))
                use_rephrase = gr.Checkbox(label="Rephrase with the rhyme MCP before removing (merge synonyms, "
                                                 "place in background, wearing/holding, synonym, more specific)",
                                           value=True, scale=2, elem_id=self.elem_id("use_rephrase"))
                vocab = gr.Dropdown(["auto", "danbooru", "natural"], value="auto", scale=1, min_width=160,
                                    label="Model vocabulary for rephrasing (auto: from the checkpoint / prompt style)",
                                    elem_id=self.elem_id("vocab"))
            with gr.Row():
                neg_on = gr.Checkbox(label="Optimize the negative prompt too (phrases fighting the prompt, idle "
                                           "and duplicate phrases, defects the evaluation found)",
                                     value=True, scale=1, elem_id=self.elem_id("neg_on"))
                qual_on = gr.Checkbox(label="Consider image quality: strengthen words whose area is blurry / flat, "
                                            "weaken words with noisy areas and tags that made earlier images "
                                            "blurrier / noisier (SD MCP history)",
                                      value=True, scale=1, elem_id=self.elem_id("qual_on"))
                record_hist = gr.Checkbox(label="Add every generation to the SD MCP history (sharpness, noise, "
                                                "detail and tags per image, for the quality history)",
                                          value=evaluate.ENABLED, interactive=evaluate.ENABLED, scale=1,
                                          elem_id=self.elem_id("record_hist"))
            with gr.Row():
                learn_on = gr.Checkbox(label="Learn from each optimisation (what changed, did the next image get "
                                             "better: conflicts, blurry / noisy areas, SD MCP quality; saved)",
                                       value=True, scale=1, elem_id=self.elem_id("learn_on"))
                learn_apply = gr.Checkbox(label="Apply learned improvements to new prompts before generating",
                                          value=True, scale=1, elem_id=self.elem_id("learn_apply"))
            with gr.Row():
                adh_on = gr.Checkbox(label="Fix prompt adherence: strengthen / move up tags the image misses, keep a "
                                           "colour that landed on another object on its own (SD MCP tagger)",
                                     value=True, scale=1, elem_id=self.elem_id("adh_on"))
                order_on = gr.Checkbox(label="Order the prompt: subject first after the quality tags, weak phrases "
                                             "last, no phrase cut by a 75-token chunk",
                                       value=True, scale=1, elem_id=self.elem_id("order_on"))
                timing_on = gr.Checkbox(label="Timing: the weaker word of a layout conflict joins later ([word:0.3]) "
                                              "instead of being weakened",
                                        value=True, scale=1, elem_id=self.elem_id("timing_on"))
            auto_step_btn = gr.Button("auto step", elem_id=f"act_steer_{tab}_autostep",
                                      elem_classes="act-steer-hidden")
            opt_md = gr.Markdown("")
            word_box = gr.Textbox(elem_id=f"act_steer_{tab}_word", elem_classes="act-steer-hidden",
                                  show_label=False)
            # rules from the right-click menu ("allow|word|other|nonce"): own light event, see on_rule_action
            rule_in = gr.Textbox(elem_id=f"act_steer_{tab}_rule", elem_classes="act-steer-hidden", show_label=False)
            gallery_idx = gr.Number(value=-1, visible=False)
            selected = gr.State("")

            with gr.Accordion("Rules", open=False):
                cur_rules = rules_mod.current()  # loaded on start, default rules included
                with gr.Row():
                    rules_box = gr.Textbox(value=cur_rules.text, lines=8, label="Rules",
                                           placeholder="allow hair ~ girl\nprefer ocean over sea\nkeep straw hat",
                                           elem_id=self.elem_id("rules"))
                    default_box = gr.Textbox(value=cur_rules.defaults, lines=8,
                                             label="Default rules (saved, loaded on every start, always applied)",
                                             elem_id=self.elem_id("default_rules"))
                with gr.Row():
                    rules_save = gr.Button("💾 Save rules", size="sm", scale=0, min_width=140)
                    defaults_save = gr.Button("⭐ Save default rules", size="sm", scale=0, min_width=170)
                    defaults_builtin = gr.Button("↺ Built-in defaults", size="sm", scale=0, min_width=150)
                    rules_status = gr.Markdown(cur_rules.summary())
                gr.Markdown(
                    "One rule per line, for every prompt (saved in the profile folder); `when <word>` at the end "
                    "limits a rule to prompts with that word, `*` is a wildcard, `a|b|c` alternatives, `#` a "
                    "comment. Saving is instant, the word map follows in the background.\n\n"
                    "`allow hair ~ girl` they may share an area (no conflict) · `prefer ocean over sea` on overlap "
                    "sea changes · `keep straw hat` never changed · `never remove sky` weakened, not removed · "
                    "`limit dress 0.7..1.3` factor range · `replace sea -> ocean` · `negative keep lowres` · "
                    "`negative add blurry:1.2` · `place sea|ocean -> background` a fixed layer whenever the prompt "
                    "has the word (`:0.6` strength; off: the *Placement rules* checkbox). Right click a word: "
                    "*allow overlap* / *prefer* / *📌 Always move to* add rules too.")

            with gr.Accordion("Learned", open=False):
                learned_md = gr.Markdown("Open and press 🔄 to see what earlier optimisations taught.")
                with gr.Row():
                    learned_refresh = gr.Button("🔄 Refresh", size="sm", scale=0, min_width=120)
                    learned_rules = gr.Button("➕ Add suggested allow rules", size="sm", scale=0, min_width=200)
                    learned_reset = gr.Button("🗑 Forget everything learned", variant="stop", size="sm", scale=0,
                                              min_width=200)

            with gr.Accordion("Separation LoRA (train)", open=False):
                gr.Markdown("Trains a LoRA on the cross-attention (UNet to_q / to_k, Anima q_proj / k_proj) so nouns of different prompt "
                            "phrases take different areas by themselves, like the word edits of Optimize do per "
                            "prompt. It generates the images below with the loaded model, trains on their sampling "
                            "steps and saves the LoRA to the LoRA folder: use it as `<lora:name:1>`. "
                            "`allow a ~ b` rules are respected. SD 1.x / SDXL / Anima; nothing else can run meanwhile.")
                with gr.Row():
                    tr_images = gr.Slider(4, 500, value=40, step=1, label="Images to generate",
                                          elem_id=self.elem_id("tr_images"))
                    tr_steps = gr.Slider(50, 5000, value=600, step=50, label="Training steps",
                                         elem_id=self.elem_id("tr_steps"))
                    tr_rank = gr.Slider(1, 32, value=4, step=1, label="LoRA rank", elem_id=self.elem_id("tr_rank"))
                    tr_keep = gr.Slider(0.0, 5.0, value=septrain.KEEP_WEIGHT, step=0.1,
                                        label="Keep the model's attention (higher = smaller change)",
                                        elem_id=self.elem_id("tr_keep"))
                with gr.Row():
                    tr_lines = gr.Textbox(value=septrain.DEFAULT_LINES, lines=2, scale=3,
                                          label="Training prompts (one per line, cycled; {pair} = two or three random "
                                                "things, {place} = a random place, {person} = a person with hair "
                                                "and clothing colours, {animal}, {thing}, {colour}; no word twice "
                                                "in one prompt)",
                                          elem_id=self.elem_id("tr_lines"))
                    with gr.Column(scale=1, min_width=200):
                        tr_current = gr.Checkbox(label="Every other image uses the current prompt", value=False,
                                                 elem_id=self.elem_id("tr_current"))
                        tr_name = gr.Textbox(value="act_steer_separation", label="LoRA name",
                                             elem_id=self.elem_id("tr_name"))
                        tr_complex = gr.Button("Complex templates", size="sm")
                        tr_complex.click(lambda: septrain.COMPLEX_LINES, outputs=[tr_lines], show_progress="hidden")
                with gr.Row():
                    tr_btn = gr.Button("🎓 Train separation LoRA", variant="primary", size="sm", scale=0,
                                       min_width=220)
                    tr_stop = gr.Button("⏹ Stop", size="sm", scale=0, min_width=100)
                tr_md = gr.Markdown("")
                tr_gallery = gr.Gallery(label="Generated training images", columns=8, height="auto", visible=False)

                def on_train(n, steps, rank, keep_w, lines, use_cur, name, prompt):
                    status = {"md": "Waiting for the current job to finish …", "done": False, "images": []}
                    threading.Thread(target=train_separation, daemon=True,
                                     args=(status, n, steps, rank, keep_w, lines, use_cur, name, prompt)).start()
                    import time as _time
                    last = None
                    while not status["done"]:
                        if status["md"] != last:
                            last = status["md"]
                            yield last, gr.update()
                        _time.sleep(1.0)
                    imgs = status["images"]
                    yield status["md"], gr.update(value=imgs[:32], visible=bool(imgs))

                tr_btn.click(on_train,
                             _js=f"(...a) => {{ a[7] = gradioApp().querySelector('#{tab}_prompt textarea').value; "
                                 "return a; }",
                             inputs=[tr_images, tr_steps, tr_rank, tr_keep, tr_lines, tr_current, tr_name, tr_name],
                             outputs=[tr_md, tr_gallery])
                tr_stop.click(lambda: shared.state.interrupt(), show_progress="hidden")

            with gr.Accordion("Profiles", open=False):
                gr.Markdown("A profile stores what images activated inside the UNet. Merge the selected gallery "
                            "image (none selected = all of the last job) as positive or negative, then apply "
                            "profiles here or with `<steer:name:strength>` in the prompt.")
                with gr.Row():
                    with gr.Column(scale=3, min_width=240):
                        with FormRow():
                            steers = gr.Dropdown(label="Apply profiles", choices=core.list_steers(), value=[],
                                                 multiselect=True, elem_id=self.elem_id("steers"))
                            refresh = ToolButton("\U0001f504")
                    with gr.Column(scale=2, min_width=200):
                        strength = gr.Slider(-4.0, 6.0, value=1.0, step=0.05, label="Strength",
                                             elem_id=self.elem_id("strength"))
                info = gr.Markdown("")
                gr.Markdown("**Add the selected image to a profile**")
                with gr.Row():
                    with gr.Column(scale=1, min_width=240):
                        with FormRow():
                            profile = gr.Dropdown(label="Profile to merge into", choices=core.list_steers(),
                                                  value=None, elem_id=self.elem_id("profile"))
                            refresh_p = ToolButton("\U0001f504")
                    with gr.Column(scale=1, min_width=200):
                        new_name = gr.Textbox(label="New profile name", placeholder="empty: from prompt",
                                              elem_id=self.elem_id("new_name"))
                with gr.Row():
                    impact = gr.Radio(["positive", "negative"], value="positive", label="Impact",
                                      elem_id=self.elem_id("impact"))
                    weight = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Weight",
                                       elem_id=self.elem_id("weight"))
                with gr.Row():
                    merge_btn = gr.Button("➕ Merge", variant="primary", size="sm", min_width=100)
                    create_btn = gr.Button("✨ New profile", size="sm", min_width=100)
                    skip_btn = gr.Button("⏭ Skip", size="sm", min_width=100)
                    undo_btn = gr.Button("↩ Undo", size="sm", min_width=100)
                    delete_btn = gr.Button("🗑 Delete", variant="stop", size="sm", min_width=100)
                status = gr.Markdown("")

            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    wordmap_on = gr.Checkbox(label="Record word maps", value=True, elem_id=self.elem_id("wordmap"))
                    record = gr.Checkbox(label="Keep influence for profiles", value=True,
                                         elem_id=self.elem_id("record"))
                    cond_only = gr.Checkbox(label="Prompt pass only", value=False, elem_id=self.elem_id("cond_only"))
                with gr.Row():
                    edit_start = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                           label="Word edits start (0 = may re-compose, later = keeps layout)",
                                           elem_id=self.elem_id("edit_start"))
                    place_start = gr.Slider(0.0, 0.8, value=0.15, step=0.05,
                                            label="Layers start (the subject's area must have formed; later = "
                                                  "less re-composition)", elem_id=self.elem_id("place_start"))
                    buckets = gr.Slider(1, 10, value=4, step=1, label="Time buckets",
                                        elem_id=self.elem_id("buckets"))
                with gr.Row():
                    rec_focus = gr.Textbox(label="Profile focus words", placeholder="empty: whole image | e.g. eyes",
                                           elem_id=self.elem_id("rec_focus"))
                    focus_override = gr.Textbox(label="Apply profiles at words", placeholder="empty: profile's own",
                                                elem_id=self.elem_id("focus_override"))
                    blocks = gr.Textbox(label="Blocks", value="all", placeholder="all | IN4-8,M,OUT0-5",
                                        elem_id=self.elem_id("blocks"))
                with gr.Row():
                    mode = gr.Radio(["add", "gain"], value="add", label="Profile mode", elem_id=self.elem_id("mode"))
                    auto = gr.Radio(AUTO_CHOICES, value="ask", label="Merge automatically",
                                    elem_id=self.elem_id("auto"))
                with gr.Row():
                    start = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Profiles start",
                                      elem_id=self.elem_id("start"))
                    end = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Profiles end", elem_id=self.elem_id("end"))

            # ---- word map and word edits ----------------------------------------------

            def view_html(i=None, edit_text="", sel=None):
                last = core.last_wordmaps.get(tab)
                if not last or not last["items"]:
                    return "<p>No word map yet: generate an image (Advanced: 'Record word maps' on).</p>"
                i = wordmap_shown.get(tab, 0) if i is None else i
                i = min(max(i, 0), len(last["items"]) - 1)
                image, words = last["items"][i][:2]
                lora = (last.get("lora") or [None] * (i + 1))[i]
                used = (last.get("prompts") or [("", "")] * (i + 1))[i][0]
                return wordmap.render_html(image, words, dict(parse_edits(edit_text)), sel, lora,
                                           removed=removed_words.get(tab, set()),
                                           rules=rules_mod.current().for_prompt(used),
                                           show=view_filter.get(tab)) + neg_view(i) + \
                    quality.table_html(word_quality(i)) + history_view(used)

            def eval_sharpness():
                """The sharpness percentile (SD MCP, 100 = sharpest of the checkpoint) of the last
                evaluation when it was of the shown job, else None."""
                ev = last_eval.get(tab)
                if not ev or ev[0] != wordmap_version.get(tab) or not isinstance(ev[1], dict):
                    return None
                return (ev[1].get("percentiles") or {}).get("blur_value")

            def history_view(used):
                hist = quality.history_cached(model_name(), evaluate.call_tool) \
                    if evaluate.ENABLED else None  # never waits for the MCP
                if not hist or not used:
                    return ""
                return quality.history_html(hist, promptedit.prompt_phrases(used), eval_sharpness())

            def word_quality(i):
                """quality.word_quality of image i of the last job (cached per job)."""
                key = (tab, wordmap_version.get(tab), i)
                if key not in quality_cache:
                    last = core.last_wordmaps.get(tab)
                    image, words = last["items"][i][:2]
                    quality_cache.clear()
                    quality_cache[key] = quality.word_quality(image, words)
                return quality_cache[key]

            def neg_issues(i, neg_prompt=None, pos_prompt=None, lock_text=""):
                """Negative prompt check of image i of the last job; prompts default to the ones it was
                made with. Defect tags from the last evaluation when it was of this job."""
                last = core.last_wordmaps.get(tab)
                negs, prompts = last.get("neg") or [], last.get("prompts") or []
                if i >= len(negs) or negs[i] is None:
                    return None
                pos_used, neg_used = prompts[i] if i < len(prompts) else ("", "")
                ev = last_eval.get(tab)
                defects = (ev[1].get("defect_tags") or []) if ev and ev[0] == wordmap_version.get(tab) else []
                R = rules_mod.current().for_prompt(pos_used if pos_prompt is None else pos_prompt)
                return negative.check(neg_used if neg_prompt is None else neg_prompt, negs[i],
                                      pos_used if pos_prompt is None else pos_prompt, last["items"][i][1],
                                      defects, split_words(lock_text), embedding_names() | set(R.neg_kept()))

            def neg_view(i):
                issues = neg_issues(i)
                if issues is None:
                    return ""
                items = "".join(f"<li>{wordmap._md(line[2:])}</li>" for line in negative.lines(issues).splitlines())
                return (f'<details class="wm-check"><summary>Negative prompt check: {negative.summary(issues)}'
                        f'</summary><ul>{items}</ul><p class="wm-p" style="font-size:12px">Optimize changes these '
                        'when "Optimize the negative prompt too" is on. Defects come from 📊 Evaluate.</p></details>')

            def neg_step(neg_prompt, prompt, lock_text):
                """-> (new negative prompt or "" when unchanged, markdown lines of the changes)."""
                last = core.last_wordmaps.get(tab)
                if not last or not last["items"]:
                    return "", ""
                i = min(wordmap_shown.get(tab, 0), len(last["items"]) - 1)
                issues = neg_issues(i, neg_prompt or "", prompt or "", lock_text) or []
                st = learn.stats(model_name()) if learning(tab) else {}
                if st:  # skip negative changes that did not help before
                    issues = [x for x in issues if learn.verdict(st, neg_key(x)) != "bad"]
                new, done = negative.apply(neg_prompt or "", issues) if issues else (neg_prompt or "", [])
                new, added = negative.add_required(new, rules_mod.current().for_prompt(prompt).neg_added())
                done += added
                step_changes.setdefault(tab, []).extend(
                    learn.change("neg_" + ("add" if x["action"] == "add" else x["action"]),
                                 negative.phrase_key(x["phrase"]), x.get("weight")) for x in done)
                return (new, negative.lines(done)) if done and new != (neg_prompt or "") else ("", "")

            def neg_key(issue):
                return f"neg_{issue['action']}:{negative.phrase_key(issue['phrase'])}"

            def learn_begin(words, lock_text, prompt, neg_prompt, new_prompt, new_neg, ev=None):
                """An optimisation was applied: start the experiment (its result is the next image).
                The image before gets its SD MCP quality score from ev, the last evaluation of it,
                or a background evaluation."""
                chg = step_changes.pop(tab, [])
                if not learning(tab) or not chg:
                    return
                i = min(wordmap_shown.get(tab, 0), len(core.last_wordmaps[tab]["items"]) - 1)
                R = rules_mod.current().for_prompt(prompt)
                o = wordmap.max_conflict(words, split_words(lock_text), promptedit.prompt_words(prompt), R)[0]
                flags = sum(1 for q in word_quality(i).values() if q["flags"])
                q, adh = evaluate.quality(ev), evaluate.adherence(ev)
                le = last_eval.get(tab)
                if le and le[0] == wordmap_version.get(tab) and isinstance(le[1], dict):
                    q = evaluate.quality(le[1]) if q is None else q
                    adh = evaluate.adherence(le[1]) if adh is None else adh
                exp_id = learn.start(tab, model_name(), prompt, neg_prompt, new_prompt or prompt, new_neg or neg_prompt,
                                     {"conflict": round(o, 4), "flags": flags, "quality": q, "adherence": adh}, chg,
                                     version=wordmap_version.get(tab))
                if (q is None or adh is None) and exp_id:
                    evaluate_later(exp_id, "before", *shown_item())

            def clear_state():
                for d in (rephrase_tried, removed_words, last_eval, auto_runs, fresh_prompt):
                    d.pop(tab, None)

            def on_reset():
                clear_state()
                return ("", "", "🧹 Word edits, locked words, tried rephrasings and the last evaluation are cleared "
                        "for a new prompt.", view_html(None, ""), gr.update(value=1.0), gr.update(value="🔁 Auto optimize"))

            def do_show_clear(g, edit_text):
                """The word map of the selected image; after the first job of a new prompt also clears
                the old word edits and locked words (they belonged to the previous image)."""
                applied = learned_applied.pop(tab, None)
                learned_note = ("\n\n🎓 Learned improvements applied to this prompt: " + "; ".join(applied)
                                if applied else "")
                if fresh_prompt.pop(tab, False):
                    return (*do_show(g, ""), "", "", "🧹 New prompt: the old word edits and locked words were "
                            "cleared (they were not used for this image)." + learned_note)
                if applied:
                    return (*do_show(g, edit_text), gr.update(), gr.update(), learned_note.strip())
                return (*do_show(g, edit_text), gr.update(), gr.update(), gr.update())

            def do_show(g, edit_text):
                last = core.last_wordmaps.get(tab)
                g = int(g if g is not None else -1)
                i = g - last["first"] if last and g >= last["first"] else 0
                wordmap_shown[tab] = max(i, 0)
                return view_html(i, edit_text), ""

            def on_word(raw, edit_text, lock_text):
                """javascript/act_steer.js: 'action|word|value|nonce' from a click or the right-click menu
                (select, set <factor>, reset, lock, removed = taken out of the prompt, restored, allow / prefer
                <other word> = a new rule)."""
                parts = (raw or "").split("|")
                action, word, value = (parts + ["", ""])[:3] if len(parts) >= 4 else ("select", parts[0], "")
                word = word.strip()
                if not word:
                    return gr.update(), gr.update(), "", gr.update(), gr.update(), gr.update()
                edit_list = [(w, f) for w, f in parse_edits(edit_text) if w.lower() != word.lower()]
                lock = split_words(lock_text)
                unlocked = [w for w in lock if w.lower() != word.lower()]
                if action == "set":
                    try:
                        factor = min(5.0, max(-1.0, float(value)))
                    except ValueError:
                        factor = 1.0
                    if abs(factor - 1.0) > 1e-6:
                        edit_list.append((word, factor))
                    lock = unlocked + [word]  # set by hand: Optimize keeps it
                elif action == "reset":
                    pass
                elif action == "lock":
                    lock = unlocked if len(unlocked) < len(lock) else lock + [word]
                elif action == "removed":
                    lock = unlocked
                    removed_words.setdefault(tab, set()).add(word.lower())
                elif action == "restored":
                    removed_words.get(tab, set()).discard(word.lower())
                if action == "select" or action == "lock" or action == "restored":
                    edit_list = parse_edits(edit_text)
                edit_text = format_edits(edit_list)
                lock_text = ", ".join(lock)
                factor = dict((w.lower(), f) for w, f in edit_list).get(word.lower(), 1.0)
                text = f"Selected: **{word}**"
                if word.lower() in removed_words.get(tab, set()):
                    text += " (removed from the prompt)"
                elif any(w.lower() == word.lower() for w in lock):
                    text += " 🔒"
                last = core.last_wordmaps.get(tab)
                if last and last["items"]:
                    i = min(wordmap_shown.get(tab, 0), len(last["items"]) - 1)
                    rel = [(k, o) for k, o in wordmap.related_to(last["items"][i][1], word) if o >= 0.1]
                    text += (" - shares its area with " + ", ".join(f"**{k}** {o:.2f}" for k, o in rel)
                             if rel else " - acts on its own area (no strong relation to other words)")
                return (text, gr.update(value=factor), word, view_html(None, edit_text, word), edit_text,
                        lock_text)

            def on_influence(value, word, edit_text, lock_text):
                if not word:
                    return gr.update(), gr.update(), gr.update()
                rest = [(w, f) for w, f in parse_edits(edit_text) if w.lower() != word.lower()]
                if abs(value - 1.0) > 1e-6:
                    rest.append((word, float(value)))
                text = format_edits(rest)
                # a word set by hand is kept by Optimize
                lock = split_words(lock_text)
                if word.lower() not in {w.lower() for w in lock}:
                    lock.append(word)
                return text, view_html(None, text, word), ", ".join(lock)

            def on_lock(word, lock_text):
                if not word:
                    return gr.update()
                lock = split_words(lock_text)
                kept = [w for w in lock if w.lower() != word.lower()]
                return ", ".join(kept if len(kept) < len(lock) else lock + [word])

            def shown_words():
                last = core.last_wordmaps.get(tab)
                if not last or not last["items"]:
                    return None
                return last["items"][min(wordmap_shown.get(tab, 0), len(last["items"]) - 1)][1]

            def with_auto_lock(lock_text, auto):
                """Locked words text + the auto-locked words not in it."""
                lock = split_words(lock_text)
                have = {w.lower() for w in lock}
                return ", ".join(lock + [w for w in auto if w not in have])

            def auto_lock_note(auto):
                return (f"\n\n🛡 Guarded (most important, ±{GUARD_BAND:.0%} at most, never removed): "
                        f"{', '.join(f'**{w}**' for w in auto)}") if auto else ""

            def optimize_step(words, edit_text, lock_text, write_prompt, prompt, use_rephrase=False, vocabulary="auto",
                              use_quality=False, ev=None, adh=False, order_on=False, timing_on=False, layers_text="",
                              guard=(), guard_base=None, **kw):
                """-> (plan, new edits text, new prompt or "", where the changes went, notes, new layers text
                or "" when unchanged).
                Words removed for too much overlap (remove_above) are rephrased by the rhyme MCP
                instead when use_rephrase and the changes go into the prompt (suggestions already
                tried for a word are skipped, see rephrase_tried). adh: phrases the evaluation ev missed
                are strengthened / moved up / bound to their object (lib_act_steer/adherence.py);
                timing_on: the weaker word of a conflict that pulls at the layout early is delayed
                instead of weakened (lib_act_steer/timing.py); order_on: subject first, weak phrases
                last, no phrase cut by a 75-token chunk (lib_act_steer/order.py). guard: the most important
                words; their factor stays within GUARD_BAND of guard_base (word -> factor at the start, filled
                from this image when missing) and they are never removed, rephrased, replaced or delayed."""
                edit_list = parse_edits(edit_text)
                current = {w.lower(): f for w, f in edit_list}
                if write_prompt:  # the prompt weights the image was made with count as well
                    names = {t.lower() for t, m, _, _ in words if m is not None}
                    for w, pw in promptedit.prompt_weights(prompt or "", names).items():
                        current[w] = current.get(w, 1.0) * pw
                allowed = promptedit.prompt_words(prompt)
                R = rules_mod.current().for_prompt(prompt)
                guard_base = {} if guard_base is None else guard_base
                for w in guard:
                    guard_base.setdefault(w.lower(), current.get(w.lower(), 1.0))
                guard = {w.lower(): guard_base[w.lower()] for w in guard}
                plan, merged = wordmap.optimize(words, current, split_words(lock_text), allowed=allowed, rules=R, **kw)
                if use_quality:  # image quality per word, and what tags did to earlier images
                    phrases = promptedit.prompt_phrases(prompt)
                    if write_prompt:
                        current.update(promptedit.prompt_weights(prompt or "", phrases))
                    i = min(wordmap_shown.get(tab, 0), len(core.last_wordmaps[tab]["items"]) - 1)
                    local = {w: q for w, q in word_quality(i).items() if w in allowed}
                    hist = quality.history(model_name(), evaluate.call_tool) if evaluate.ENABLED else {}
                    qplan = quality.plan(local, hist, phrases, current, split_words(lock_text) + R.kept(),
                                         skip=[w for w, _, _ in plan], sharpness_pct=eval_sharpness())
                    qplan = [(w, round(R.limit(w, f), 2), why) for w, f, why in qplan]
                    plan = plan + qplan
                    merged.update({w: f for w, f, _ in qplan})
                notes = []
                st = learn.stats(model_name()) if learning(tab) else {}
                kind_of = lambda w, f: ("remove" if f == 0 else  # noqa: E731
                                        "weaken" if f < current.get(w, 1.0) else "strengthen")
                if st:  # what earlier experiments taught: skip changes that did not help
                    kept_plan = []
                    for w, f, why in plan:
                        if learn.verdict(st, f"{kind_of(w, f)}:{w}") != "bad":
                            kept_plan.append((w, f, why))
                        elif kind_of(w, f) == "weaken" and kw.get("remove_above") is not None:
                            kept_plan.append((w, 0.0, f"weakening **{w}** did not help in earlier images: {wordmap.REMOVED}"))
                        else:
                            notes.append(f"**{w}** → {f:g} skipped: it did not help in earlier images (learned)")
                    plan = kept_plan

                def guard_clamp(w, f):
                    b = guard[w]
                    return round(min(max(f, b * (1 - GUARD_BAND)), b * (1 + GUARD_BAND)), 2)

                if guard:  # the most important words: small steps only, never removed
                    kept_plan = []
                    for w, f, why in plan:
                        if w in guard:
                            g = guard_clamp(w, f)
                            if abs(g - current.get(w, 1.0)) < 1e-6:
                                notes.append(f"**{w}** → {f:g} skipped: guarded word, already at its "
                                             f"±{GUARD_BAND:.0%} limit")
                                continue
                            if g != f:
                                why = (why.replace(wordmap.REMOVED, "too high") + f"; guarded word: only "
                                       f"±{GUARD_BAND:.0%}, never removed")
                            f = g
                        kept_plan.append((w, f, why))
                    plan = kept_plan
                    planned_f = {w: f for w, f, _ in plan}
                    for w in guard:
                        if w in merged:
                            merged[w] = planned_f.get(w, current.get(w, 1.0))
                verdict = (lambda key: learn.verdict(st, key)) if st else None
                locked_l = split_words(lock_text)
                missed_tags = adherence.missed(ev) if adh else []
                if missed_tags:  # words the image misses are not weakened this round
                    prot = adherence.protected(ev, prompt)
                    spared = [w for w, f, _ in plan if w in prot and f < current.get(w, 1.0)]
                    plan = [(w, f, why) for w, f, why in plan if w not in spared]
                    notes += [f"**{w}** not weakened: the image misses it" for w in spared]
                seen = adherence.shown(ev, prompt) if adh else set()
                if seen:  # the image shows these: never removed, an overlap is no reason to weaken them, and a
                    # boost that made them appear is not taken back
                    pw = {w: max([promptedit.piece_weight(pc, x) for pc in promptedit.pieces(prompt)
                                  if w in promptedit.prompt_words(pc) for x in (None, w)] or [1.0])
                          for w in seen} if write_prompt else {}
                    spared = [(w, why) for w, f, why in plan if w in seen and f < current.get(w, 1.0)
                              and (f == 0 or "same area" in why or max(current.get(w, 1.0), pw.get(w, 1.0)) > 1.0)]
                    plan = [(w, f, why) for w, f, why in plan if w not in {x for x, _ in spared}]
                    notes += [f"**{w}** kept: the image shows it (SD MCP tagger), "
                              + ("the shared area is no problem" if "same area" in why else "so it works as it is")
                              for w, why in spared]
                    # (the weight it got is in a phrase: other phrases with that word were left alone)
                adh_plan = adherence.plan(ev, prompt, current, locked_l, R, verdict) if missed_tags else []
                for a in adh_plan:
                    if a["kind"] == "boost" and a["word"] in guard:
                        a["factor"] = guard_clamp(a["word"], a["factor"])
                planned = {w for w, _, _ in plan}
                boosts = [a for a in adh_plan if a["kind"] == "boost" and a["word"] not in planned]
                if not write_prompt:  # word edits act on every occurrence of the word
                    plan = plan + [(a["word"], a["factor"], a["why"]) for a in boosts]
                    merged.update({a["word"]: a["factor"] for a in boosts})
                binds = [a for a in adh_plan if a["kind"] == "bind"]
                ups = [a for a in adh_plan if a["kind"] == "move_up"] if write_prompt else []
                delays, undelays = [], []
                if timing_on and write_prompt:
                    i_t = min(wordmap_shown.get(tab, 0), len(core.last_wordmaps[tab]["items"]) - 1)
                    early = (core.last_wordmaps[tab].get("early") or [None] * (i_t + 1))[i_t] or []
                    guarded_plan = [x for x in plan if x[0] in guard]  # a guarded word is weakened, not delayed
                    delays, undelays, plan = timing.plan([x for x in plan if x[0] not in guard], words, early,
                                                         prompt, missed_tags, current, verdict)
                    plan = guarded_plan + plan
                    back = {w for w, _ in undelays}  # moving back in place is this round's fix for these
                    boosts = [a for a in boosts if a["word"] not in back]
                    if not early:
                        notes.append("timing: no early word maps for this image (generate once with the word map on)")
                new_layers = ""
                if binds:
                    entries = parse_layers(layers_text)
                    have = {e[0].lower() for e in entries}
                    for a in binds:
                        if a["word"].lower() not in have:
                            entries += parse_layers(a["layer"])
                            notes.append(f"layer `{a['layer']}`: {a['why']}")
                    new_layers = format_layers(entries) if len(entries) > len(parse_layers(layers_text)) else ""
                lora_ch = []
                i = min(wordmap_shown.get(tab, 0), len(core.last_wordmaps[tab]["items"]) - 1)
                lb = (core.last_wordmaps[tab].get("lora") or [None] * (i + 1))[i]
                if lb and write_prompt:  # LoRA tags: block weights (lbw=) and strength
                    le = last_eval.get(tab)
                    missed = (le[1].get("missed_tags") or []) if le and le[0] == wordmap_version.get(tab) \
                        and isinstance(le[1], dict) else []
                    for ch in loratune.plan(prompt or "", words, lb, R, split_words(lock_text),
                                            kw.get("same_area_min"), missed, allowed):
                        bad = [k for k, w, _ in ch["changes"] if learn.verdict(st, f"{k}:{w}") == "bad"] if st else []
                        if bad:
                            notes.append(f"LoRA **{ch['name']}** → `{ch['text']}` skipped: it did not "
                                         "help in earlier images (learned)")
                            continue
                        lora_ch.append(ch)
                    handled = loratune.pair_words(lora_ch, words, lb, R, kw.get("same_area_min"))
                    if handled:  # the LoRA change takes care of these overlaps this round
                        plan = [(w, f, why) for w, f, why in plan if not (w in handled and f < current.get(w, 1.0))]
                elif lb:
                    notes.append("LoRA tags are tuned only when *Optimize writes into the prompt* is on")
                other_of = lambda w: (wordmap.related_to(words, w, n=1) or [("", 0.0)])[0][0]  # noqa: E731
                boost_words = {a["word"] for a in boosts} if not write_prompt else set()
                chg = [learn.change("boost_missed" if w in boost_words else kind_of(w, f), w, f, other_of(w))
                       for w, f, why in plan if not (write_prompt and f == 0 and wordmap.REMOVED in why)]
                if write_prompt:
                    chg += [learn.change("boost_missed", a["word"], a["factor"]) for a in boosts]
                chg += [learn.change("bind", a["tag"], a["layer"]) for a in binds if new_layers]
                for ch in lora_ch:
                    chg += [learn.change(k, w, d) for k, w, d in ch["changes"]]
                    notes.append(f"LoRA **{ch['name']}** → `{ch['text']}`: " + "; ".join(ch["why"]))
                step_changes[tab] = chg
                base = loratune.apply(prompt or "", lora_ch)
                if write_prompt:  # replace rules
                    for old, new in R.replacements():
                        if R.is_kept(old) or promptedit.prompt_words(old) & set(guard):  # keep / guard win
                            continue
                        base, n = promptedit.replace_text(base, old, new)
                        if n:
                            base = promptedit.dedupe_phrases(base)
                            notes.append(f"**{old}** → '{new}' (rule)")
                            chg.append(learn.change("replace", old, new))
                if write_prompt:  # missed phrases: weight only inside the phrase (other phrases keep the word as it is)
                    for a in boosts:
                        base, piece = promptedit.weight_piece(base, a["tag"], a["factor"],
                                                              a["word"] if a["scope"] == "word" else None)
                        if piece:
                            notes.append(f"`{piece}`: {a['why']}")
                if write_prompt:  # timing and order: delay / un-delay, move up, subject first, weak last, chunks
                    for w, why in undelays:
                        base, piece = promptedit.undelay_piece(base, w)
                        if piece:
                            notes.append(why)
                            chg.append(learn.change("undelay", w))
                    for w, why in delays:
                        base, piece = promptedit.delay_piece(base, w, timing.DELAY)
                        if piece:
                            notes.append(f"**{w}** → `[{piece}:{timing.DELAY:g}]`: {why}")
                            chg.append(learn.change("delay", w, timing.DELAY, other_of(w)))
                    subject = set(rules_mod.PERSONS.split("|")) if order_on else ()
                    weak = []
                    if order_on:
                        weak = [w for w in wordmap.no_effect_words(words, locked_l + R.kept() + list(guard))
                                if w in allowed and w not in planned and w not in adherence.protected(ev, prompt)
                                and w not in adherence.shown(ev, prompt)]
                    base, moves = order.plan(base, subject, [a["word"] for a in ups], weak, locked_l + R.kept())
                    why_up = {a["word"]: a["why"] for a in ups}
                    for kind, piece, why in moves:
                        key = promptedit.phrase_key(piece)
                        notes.append(f"`{piece}` {why}" + (f" ({why_up[key]})" if key in why_up else ""))
                        chg.append(learn.change(kind, key))


                def chunk_fix(text):
                    """No phrase cut by a 75-token chunk boundary (on the finished prompt: weights and
                    rephrasings change the token counts)."""
                    emb = text_tokenizer() if order_on and write_prompt else None
                    if emb is None:
                        return text
                    count = lambda t: len(emb.tokenize([t])[0])  # noqa: E731
                    text, fixed = order.fix_splits(text, count, getattr(shared.opts, "comma_padding_backtrack", 20),
                                                   allow_break=not parse_layers(layers_text))
                    for piece, how in fixed:
                        notes.append(f"`{piece}`: {how}")
                        chg.append(learn.change("chunk", promptedit.phrase_key(piece)))
                    return text

                if not plan:
                    base = chunk_fix(base)
                if not plan and base == (prompt or ""):
                    step_changes[tab] = chg
                    return plan, edit_text, "", "", "\n".join(f"- {n}" for n in notes), new_layers
                if write_prompt:
                    changes = {w: f for w, f, _ in plan}
                    removals = [w for w, f, why in plan if f == 0 and wordmap.REMOVED in why]
                    new_prompt = promptedit.apply_to_prompt(
                        base, {w: f for w, f in changes.items() if w not in removals})
                    tried = rephrase_tried.setdefault(tab, {})
                    for w in removals:
                        sug = []
                        if use_rephrase:
                            rel = wordmap.related_to(words, w, n=1)
                            other, o = rel[0] if rel else ("", 0.0)
                            sug = evaluate.rephrase(new_prompt, w, other, o, tried.get(w, []), vocabulary, model_hint())
                            if isinstance(sug, dict):
                                notes.append(f"⚠️ rhyme MCP not reachable ({sug['error']}): **{w}** removed instead")
                                sug = []
                        if sug and st:  # learned: drop suggestions that hurt before, proven ones first
                            key = lambda x: f"rephrase:{w}->{(x['text'] or '').lower()}"  # noqa: E731
                            sug = [x for x in sug if learn.verdict(st, key(x)) != "bad"]
                            sug.sort(key=lambda x: learn.verdict(st, key(x)) != "good")
                        if sug:
                            pick = sug[0]
                            chg.append(learn.change("rephrase" if pick["text"] else "remove", w, pick["text"],
                                                    other_of(w)))
                            tried.setdefault(w, []).append(pick["text"] or pick["kind"])
                            new_prompt = pick["prompt"]
                            label = f"'{pick['text']}'" if pick["text"] else "removed"
                            notes.append(f"**{w}** → {label} ({pick['kind']}): {pick['why']}")
                        else:
                            new_prompt = promptedit.apply_to_prompt(new_prompt, {w: 0.0})
                            chg.append(learn.change("remove", w, 0.0, other_of(w)))
                    new_prompt = promptedit.dedupe_phrases(new_prompt)
                    if plan:
                        new_prompt = chunk_fix(new_prompt)
                    text = format_edits([(w, f) for w, f in edit_list if w.lower() not in changes])
                    where = "written into the prompt (0 = word removed)"
                else:
                    new_prompt = ""
                    text = format_edits(list(merged.items()))
                    where = "as word edits"
                return plan, text, new_prompt, where, "\n".join(f"- {n}" for n in notes), new_layers

            def shown_item():
                """(PIL image, generation info) of the image the word map shows, or None."""
                last = core.last_wordmaps.get(tab)
                if not last or not last["items"]:
                    return None
                i = min(wordmap_shown.get(tab, 0), len(last["items"]) - 1)
                infos = last.get("infotexts") or []
                return last["items"][i][0], infos[i] if i < len(infos) else ""

            def on_evaluate():
                item = shown_item()
                if item is None:
                    return "⚠️ Generate an image first."
                ev = evaluate.evaluate(*item, tag=f"{tab}_eval")
                last_eval[tab] = (wordmap_version.get(tab), ev)
                return evaluate.summary_md(ev), view_html()

            def plan_lines(plan):
                return "\n".join(f"{n}. **{w}** → {f:g}: {why}" for n, (w, f, why) in enumerate(plan, 1))

            def current_eval(adh_on):
                """The evaluation of the shown image for the adherence fixes: the last one when it is of this
                image, else a quick one (tagger only, no quality models) -> (result or None, note)."""
                if not adh_on:
                    return None, ""
                le = last_eval.get(tab)
                if le and le[0] == wordmap_version.get(tab) and isinstance(le[1], dict):
                    return le[1], ""
                ev = evaluate.evaluate(*shown_item(), tag=f"{tab}_adh", iqa=False)
                if ev.get("error") or not ev.get("ok", True):
                    return None, f"⚠️ prompt adherence not checked: {ev.get('error') or ev.get('reason', '?')}"
                last_eval[tab] = (wordmap_version.get(tab), ev)
                return ev, ""

            def on_optimize(edit_text, lock_text, write_prompt, prompt, remove_thr, use_rephrase, neg_on, neg_prompt,
                            vocabulary, qual_on, n_lock=0, adh_on=True, order_on=True, timing_on=True, layers_text=""):
                words = shown_words()
                if words is None:
                    return gr.update(), "⚠️ Generate an image first.", gr.update(), False, "", "", gr.update()
                auto = wordmap.important_words(words, int(n_lock or 0), promptedit.prompt_words(prompt))
                ev, ev_note = current_eval(adh_on)
                plan, text, new_prompt, where, notes, new_layers = optimize_step(
                    words, edit_text, lock_text, write_prompt, prompt, use_rephrase=use_rephrase, vocabulary=vocabulary,
                    use_quality=qual_on, remove_above=remove_thr if remove_thr < 1 else None, ev=ev, adh=adh_on,
                    order_on=order_on, timing_on=timing_on, layers_text=layers_text, guard=auto)
                lock_text = with_auto_lock(lock_text, auto)  # the negative prompt does not fight them
                if ev_note:
                    notes = "\n".join(x for x in (notes, f"- {ev_note}") if x)
                new_neg, neg_notes = neg_step(neg_prompt, new_prompt or prompt, lock_text) if neg_on else ("", "")
                if not plan and not new_neg and not new_prompt and not new_layers:
                    step_changes.pop(tab, None)
                    return (gr.update(), "Nothing to optimise: the prompt check found nothing (locked words "
                            "are left alone)." + (f"\n\n{notes}" if notes else "") + auto_lock_note(auto),
                            gr.update(), False, "", "", gr.update())
                keep_seed[tab] = True
                learn_begin(words, lock_text, prompt, neg_prompt, new_prompt, new_neg)
                md = "**Optimisation** (most impact first), regenerating with the same seed:"
                if plan:
                    md += f"\n\nPrompt, {where}:\n{plan_lines(plan)}"
                if notes:
                    md += f"\n\nAdherence / timing / order / rephrased / LoRA:\n{notes}"
                if neg_notes:
                    md += f"\n\nNegative prompt:\n{neg_notes}"
                md += auto_lock_note(auto)
                return (text, md, view_html(None, text), True, new_prompt, new_neg, new_layers or gr.update())

            # ---- auto mode: optimise, regenerate, measure, until the worst conflict is low ----
            # javascript/act_steer.js clicks the hidden auto step button after every finished job

            def rating_mark(h):
                return {1: "👍", -1: "👎"}.get(learn.rating_of(tab, h.get("version")), "")

            def auto_history(run):
                if not run["evaluate"]:
                    rows = [f"| {n} | {h['conflict']:.2f} | {h['pair'] or '-'} | {rating_mark(h)} |"
                            for n, h in enumerate(run["history"])]
                    return "| round | worst conflict | words | you |\n|---|---|---|---|\n" + "\n".join(rows)
                num = lambda v, f: "-" if v is None else format(v, f)  # noqa: E731
                rows = [f"| {n} | {h['conflict']:.2f} | {h['pair'] or '-'} | {num(h['quality'], '.2f')} | "
                        f"{num(h['adherence'], '.0%')} | {', '.join(h['missed'][:5]) or '-'} | {rating_mark(h)} |"
                        for n, h in enumerate(run["history"])]
                return ("| round | worst conflict | words | quality | adherence | not in image | you |\n"
                        "|---|---|---|---|---|---|---|\n" + "\n".join(rows))

            def auto_best(run):
                """Round to keep: a round you rated 👍 (the best measured of them); rounds you rated 👎 never.
                Otherwise among rounds below the threshold the best quality + adherence; else the lowest
                conflict among rounds whose quality did not drop much (QUALITY_DROP), else overall."""
                hist = list(enumerate(run["history"]))
                rated = lambda h: learn.rating_of(tab, h.get("version")) or 0  # noqa: E731
                q = lambda h: ((h["quality"] if h["quality"] is not None else -1.0)  # noqa: E731
                               + ADHERENCE_IN_BEST * (h.get("adherence") or 0.0))
                liked = [(n, h) for n, h in hist if rated(h) > 0]
                if liked:
                    return max(liked, key=lambda nh: (q(nh[1]), -nh[1]["conflict"]))[0]
                hist = [(n, h) for n, h in hist if rated(h) >= 0] or hist
                below = [(n, h) for n, h in hist if h["conflict"] < run["thr"]]
                if below:
                    return max(below, key=lambda nh: (q(nh[1]), -nh[1]["conflict"]))[0]
                q0 = run["history"][0]["quality"]
                ok = [(n, h) for n, h in hist if q0 is None or h["quality"] is None
                      or h["quality"] >= q0 - QUALITY_DROP] or hist
                return min(ok, key=lambda nh: (nh[1]["conflict"], -q(nh[1])))[0]

            def auto_finish(run, reason, edit_text):
                """Ends the run; the prompt boxes, word edits and layers keep what the last round wrote."""
                auto_runs.pop(tab, None)
                hist = run["history"]
                if not hist:
                    return (gr.update(), f"**Auto optimize** stopped: {reason}.", gr.update(), False, "", "",
                            gr.update(value="🔁 Auto optimize"), gr.update())
                best_n = auto_best(run)
                best = hist[best_n]
                md = f"**Auto optimize** done: {reason}.\n\n{auto_history(run)}\n\n"
                if best_n == len(hist) - 1:
                    md += f"The last round is the best (worst conflict {best['conflict']:.2f})."
                    if best.get("eval"):
                        md += "\n\n" + evaluate.summary_md(best["eval"])
                    return (gr.update(), md, gr.update(), False, "", "", gr.update(value="🔁 Auto optimize"),
                            gr.update())
                # a later round measured worse: the prompt, negative prompt, word edits and layers stay as the
                # optimisation left them (the boxes are not changed back); the best round is only reported
                md += (f"Round {best_n} measured best (worst conflict {best['conflict']:.2f}"
                       + (f", quality {best['quality']:.2f}" if best["quality"] is not None else "")
                       + (f", adherence {best['adherence']:.0%}" if best.get("adherence") is not None else "")
                       + (", rated 👍" if rating_mark(best) == "👍" else "")
                       + "); the prompt, negative prompt, layers and word edits of the last round are kept."
                       + (f"\n\nRound {best_n} prompt: `{best['prompt']}`" if run["to_prompt"] and best["prompt"] else "")
                       + (f"\n\nRound {best_n} negative prompt: `{best['neg']}`" if run["neg"] and best["neg"] else ""))
                return (gr.update(), md, gr.update(), False, "", "", gr.update(value="🔁 Auto optimize"), gr.update())

            def auto_step(edit_text, lock_text, write_prompt, prompt, neg_prompt, layers_text=""):
                run = auto_runs.get(tab)
                if run is None:  # not running: every finished job clicks this
                    return gr.update(), gr.update(), gr.update(), False, "", "", gr.update(), gr.update()
                # ⏹ pressed while this step was measuring / optimising: on_auto has already finished the run
                stopped = lambda: auto_runs.get(tab) is not run  # noqa: E731
                no_op = (gr.update(), gr.update(), gr.update(), False, "", "", gr.update(value="🔁 Auto optimize"),
                         gr.update())
                words = shown_words()
                if words is None:
                    return auto_finish(run, "generate an image first", edit_text)
                if core.last_wordmaps.get(tab, {}).get("interrupted"):
                    return auto_finish(run, "the generation was interrupted", edit_text)
                if run.get("auto_lock") is None:  # chosen once per run, from the first image
                    run["auto_lock"] = wordmap.important_words(words, run.get("n_lock", 0),
                                                               promptedit.prompt_words(prompt))
                own_lock = lock_text  # guarded words may still change a little: not locked for optimize_step
                lock_text = with_auto_lock(lock_text, run["auto_lock"])
                o, a, b = wordmap.max_conflict(words, split_words(lock_text), promptedit.prompt_words(prompt),
                                               rules_mod.current().for_prompt(prompt))
                ev = evaluate.evaluate(*shown_item(), tag=f"{tab}_auto{len(run['history'])}") \
                    if run["evaluate"] else None
                good = bool(ev) and not ev.get("error") and ev.get("ok", True)
                if good:
                    last_eval[tab] = (wordmap_version.get(tab), ev)
                    lf = last_finished.get(tab)  # this image is the result of the last round's experiment
                    if lf and lf[0] == wordmap_version.get(tab):
                        learn.set_quality(lf[1], "after", evaluate.quality(ev), evaluate.adherence(ev))
                run["history"].append({"conflict": o, "pair": f"{a} & {b}" if a else None, "neg": neg_prompt,
                                       "prompt": prompt, "edits": edit_text, "layers": layers_text,
                                       "version": wordmap_version.get(tab), "eval": ev if good else None,
                                       "quality": evaluate.quality(ev) if good else None,
                                       "adherence": ev.get("adherence") if good else None,
                                       "missed": (ev.get("missed_tags") or []) if good else []})
                if stopped():
                    return no_op
                if run["evaluate"] and not good and len(run["history"]) == 1:
                    run["evaluate"] = False  # MCP not reachable: go on with the conflicts only
                    run["note"] = evaluate.summary_md(ev)
                q0, q = run["history"][0]["quality"], run["history"][-1]["quality"]
                if q0 is not None and q is not None and q < q0 - QUALITY_DROP:
                    return auto_finish(run, f"quality dropped from {q0:.2f} to {q:.2f}", edit_text)
                missing = adherence.missed(ev) if run["adh"] and good else []
                if o < run["thr"] and not missing:
                    return auto_finish(run, f"worst conflict {o:.2f} is below {run['thr']:.2f}"
                                       + (" and the image shows every checkable prompt tag" if run["adh"] and good
                                          else ""), edit_text)
                if run["round"] >= run["max"]:
                    return auto_finish(run, f"{run['max']} rounds done", edit_text)
                plan, text, new_prompt, where, notes, new_layers = optimize_step(
                    words, edit_text, own_lock, run["to_prompt"], prompt, use_rephrase=run["rephrase"],
                    vocabulary=run["vocab"], use_quality=run["quality"],
                    same_area_min=run["thr"], weak=False, remove_above=run["remove"],
                    ev=ev if good else None, adh=run["adh"], order_on=run["order"], timing_on=run["timing"],
                    layers_text=layers_text, guard=run["auto_lock"], guard_base=run.setdefault("guard_base", {}))
                new_neg, neg_notes = neg_step(neg_prompt, new_prompt or prompt, lock_text) if run["neg"] else ("", "")
                if stopped():
                    return no_op
                if not plan and not new_neg and not new_prompt and not new_layers:
                    step_changes.pop(tab, None)
                    if o < run["thr"]:
                        return auto_finish(run, f"worst conflict {o:.2f} is below {run['thr']:.2f}; still missing "
                                                f"{', '.join(missing[:5])}, nothing more to try for them", edit_text)
                    return auto_finish(run, "nothing left to change (pairs of locked words, or words already "
                                            "weakened to the minimum)", edit_text)
                run["round"] += 1
                keep_seed[tab] = True
                learn_begin(words, lock_text, prompt, neg_prompt, new_prompt, new_neg, ev if good else None)
                head = (f"worst conflict {o:.2f} (**{a}** & **{b}**), target below {run['thr']:.2f}" if o >= run["thr"]
                        else f"conflicts below {run['thr']:.2f}, fixing tags the image misses: {', '.join(missing[:5])}")
                md = (f"**Auto optimize** round {run['round']}/{run['max']}: {head}; changes {where}:\n{plan_lines(plan)}"
                      + (f"\n\nAdherence / timing / order / rephrased / LoRA:\n{notes}" if notes else "")
                      + (f"\n\nNegative prompt:\n{neg_notes}" if neg_notes else "")
                      + f"\n\n{auto_history(run)}" + (f"\n\n{run['note']}" if run.get("note") else "")
                      + auto_lock_note(run["auto_lock"]))
                return (text, md, view_html(None, text), True, new_prompt, new_neg, gr.update(value="⏹ Stop auto"),
                        new_layers or gr.update())

            def on_auto(edit_text, lock_text, write_prompt, prompt, thr, rounds, do_eval, remove_thr, use_rephrase,
                        neg_on, neg_prompt, vocabulary, qual_on, n_lock=0, adh_on=True, order_on=True, timing_on=True,
                        layers_text="", button_label=""):
                run = auto_runs.get(tab)
                if run is not None or str(button_label).startswith("⏹"):
                    # stop: no further round and the image being rendered for this run is cancelled
                    shared.state.interrupt()
                    if run is None:  # already over (the label was stale): just reset the button
                        return (gr.update(), "**Auto optimize** stopped.", gr.update(), False, "", "",
                                gr.update(value="🔁 Auto optimize"), gr.update())
                    return auto_finish(run, "stopped by you", edit_text)
                rephrase_tried.pop(tab, None)
                auto_runs[tab] = {"round": 0, "max": int(rounds), "thr": float(thr), "to_prompt": bool(write_prompt),
                                  "evaluate": bool(do_eval), "history": [],
                                  "remove": float(remove_thr) if remove_thr < 1 else None,
                                  "rephrase": bool(use_rephrase), "neg": bool(neg_on), "vocab": vocabulary,
                                  "quality": bool(qual_on), "n_lock": int(n_lock or 0), "auto_lock": None,
                                  "adh": bool(adh_on) and bool(do_eval), "order": bool(order_on),
                                  "timing": bool(timing_on)}
                return auto_step(edit_text, lock_text, write_prompt, prompt, neg_prompt, layers_text)

            def on_rate(value):
                """👍 / 👎 for the shown image: into the experiments it belongs to (learn) and the SD MCP."""
                item = shown_item()
                if item is None:
                    return "⚠️ Generate an image first."
                n = learn.rate(tab, wordmap_version.get(tab), value)
                send_rating(*item, value, tab)
                return (f"{'👍' if value > 0 else '👎'} saved for the shown image"
                        + (f", counted for {n} optimisation step{'s' if n != 1 else ''} it belongs to (your rating "
                           "outweighs the measurements when Optimize decides what helped)" if n else
                           " (no optimisation step made or started from it yet: rate after ✨ / 🔁, or before "
                           "pressing them)")
                        + ". Also sent to the SD MCP (rate_image) for its tag statistics."
                        + (" Auto optimize keeps a 👍 round and never returns to a 👎 one." if tab in auto_runs else ""))

            def on_clear():
                rephrase_tried.pop(tab, None)
                return "", view_html(None, ""), gr.update(value=1.0)

            show.click(do_show_clear, _js="(g, e) => [selected_gallery_index(), e]", inputs=[gallery_idx, edits],
                       outputs=[view, selected, edits, locked, opt_md])
            reset_btn.click(on_reset, outputs=[edits, locked, opt_md, view, influence, auto_btn])
            # no progress overlay: the word map stays usable (right click again right away)
            word_box.change(on_word, inputs=[word_box, edits, locked],
                            outputs=[selected_md, influence, selected, view, edits, locked], show_progress="hidden")

            def on_rule_action(raw, layers_text):
                """A rule or a layer from the menu: only saved here (instant); the word map follows in the
                background."""
                parts = (raw or "").split("|")
                if len(parts) < 3 or not parts[1].strip() or not parts[2].strip():
                    return gr.update(), gr.update(), gr.update()
                action, word, other = parts[0], parts[1].strip(), parts[2].strip()
                if action == "place_rule":  # a fixed layer for every generation
                    line = f"place {word} -> {other}"
                    text = rules_mod.add_line(line)
                    return text, f"Added `{line}`: {rules_mod.current().summary()}", gr.update()
                if action == "place":  # other = target ("background", "on girl", "none")
                    entries = [e for e in parse_layers(layers_text) if e[0].lower() != word.lower()]
                    if other != "none":
                        entries.append((word, other, 0.85))
                    text = format_layers(entries)
                    msg = f"Layer: **{word}** → {other} (generate to see it)" if other != "none" else \
                        f"Layer of **{word}** removed"
                    return gr.update(), msg, text
                line = f"allow {word} ~ {other}" if action == "allow" else f"prefer {word} over {other}"
                text = rules_mod.add_line(line)
                return text, f"Added `{line}`: {rules_mod.current().summary()}", gr.update()

            # saving only parses and writes the file (instant); the word map is rendered again in a
            # follow-up event without a progress overlay, so the UI stays usable meanwhile
            def learned_report():
                n, st, good, bad = learn.summary(model_name())
                if not n:
                    return ("Nothing learned yet: use ✨ Optimize or 🔁 Auto optimize, generate, and the next image "
                            f"tells whether the changes helped. Stored in `{learn.PATH}`.")
                rows = sorted(st.items(), key=lambda kv: -kv[1]["score"])
                fmt = lambda k, s: (f"| {k} | {s['n']} | {s['score']:+.3f} | "  # noqa: E731
                                    f"{'-' if s['d_quality'] is None else format(s['d_quality'], '+.3f')} | "
                                    f"{('👍' * s['user'] if s.get('user', 0) > 0 else '👎' * -s['user']) or '-'} | "
                                    f"{learn.verdict(st, k) or '…'} |")
                top = [fmt(k, s) for k, s in rows[:8]] + ([fmt(k, s) for k, s in rows[-4:]] if len(rows) > 12 else [])
                sugg = learn.allow_suggestions(st)
                return (f"**{n} experiments**, {len(good)} helpful and {len(bad)} harmful changes (checkpoint "
                        f"{model_name() or '?'}; others count when it has none). Helpful ones are applied to new "
                        "prompts, harmful ones skipped by Optimize. *you* = your 👍 / 👎 (image after minus before), they "
                        "outweigh the measurements.\n\n| change | n | score | Δ quality | you | |\n"
                        "|---|---|---|---|---|---|\n" + "\n".join(top)
                        + ("\n\nSuggested rules (these overlaps never got better): " + ", ".join(f"`{x}`" for x in sugg)
                           if sugg else "") + f"\n\nStored in `{learn.PATH}`.")

            def on_learned_rules():
                st = learn.stats(model_name())
                for line in learn.allow_suggestions(st):
                    rules_mod.add_line(line)
                return rules_mod.current().text, learned_report()

            def on_learned_reset():
                learn.reset()
                return learned_report()

            def on_apply_learned(prompt, neg_prompt, n_lock=0):
                """What earlier optimisations proved, applied to the prompt boxes now (no generation, no word
                map needed). The guarded words come from the shown word map when it is of this prompt."""
                words = shown_words()
                last = core.last_images.get(tab)
                guard = []
                if words is not None and last and not is_new_prompt(last["prompt"], prompt or ""):
                    guard = wordmap.important_words(words, int(n_lock or 0), promptedit.prompt_words(prompt))
                n, _, good, _ = learn.summary(model_name())
                if not good:
                    return (f"🎓 Nothing to apply: {n} experiments, none with a helpful record yet (a change needs "
                            f"{learn.MIN_N} experiments or your 👍)." if n else
                            "🎓 Nothing learned yet: use ✨ Optimize or 🔁 Auto optimize first."), "", ""
                new_p, new_n, applied = apply_learned(prompt or "", neg_prompt or "", limit=12, guard=guard)
                if not applied or (new_p == (prompt or "") and new_n == (neg_prompt or "")):
                    return (f"🎓 The prompt already has what was learned ({len(good)} helpful changes known, none "
                            "fits this prompt or they are applied already)." + auto_lock_note(guard)), "", ""
                return ("🎓 **Learned improvements** written into the prompt (press Generate; Ctrl+Z in the box "
                        "undoes it):\n" + "\n".join(f"- {a}" for a in applied) + auto_lock_note(guard),
                        new_p if new_p != (prompt or "") else "", new_n if new_n != (neg_prompt or "") else "")

            learned_refresh.click(learned_report, outputs=[learned_md], show_progress="hidden")
            learned_rules.click(on_learned_rules, outputs=[rules_box, learned_md])
            learned_reset.click(on_learned_reset, outputs=[learned_md])
            learn_on.change(lambda v: learn_settings.__setitem__(tab, bool(v)), inputs=[learn_on])

            def on_rules_save(text):
                return f"Saved: {rules_mod.save(text).summary()}"

            def on_defaults_save(text):
                return f"Default rules saved (`{rules_mod.DEFAULT_PATH}`): {rules_mod.save_defaults(text).summary()}"

            def on_defaults_builtin():
                r = rules_mod.save_defaults(rules_mod.BUILTIN_DEFAULTS)
                return rules_mod.BUILTIN_DEFAULTS, f"Built-in default rules restored: {r.summary()}"

            def refresh_view(edit_text, sel):
                return view_html(None, edit_text, sel or None)

            def in_background(event):
                event.then(refresh_view, inputs=[edits, selected], outputs=[view], show_progress="hidden")

            in_background(rules_save.click(on_rules_save, inputs=[rules_box], outputs=[rules_status],
                                           show_progress="hidden"))
            in_background(rules_box.blur(on_rules_save, inputs=[rules_box], outputs=[rules_status],
                                         show_progress="hidden"))
            in_background(defaults_save.click(on_defaults_save, inputs=[default_box], outputs=[rules_status],
                                              show_progress="hidden"))
            in_background(default_box.blur(on_defaults_save, inputs=[default_box], outputs=[rules_status],
                                           show_progress="hidden"))
            in_background(defaults_builtin.click(on_defaults_builtin, outputs=[default_box, rules_status],
                                                 show_progress="hidden"))
            in_background(rule_in.change(on_rule_action, inputs=[rule_in, layers],
                                         outputs=[rules_box, rules_status, layers], show_progress="hidden"))
            influence.release(on_influence, inputs=[influence, selected, edits, locked], outputs=[edits, view, locked])
            lock_btn.click(on_lock, inputs=[selected, locked], outputs=[locked])
            clear.click(on_clear, outputs=[edits, view, influence])
            regen.click(None, _js=f"() => {{ gradioApp().getElementById('{tab}_generate').click(); }}")
            go = gr.Checkbox(value=False, visible=False)
            new_prompt = gr.Textbox(visible=False)
            new_neg = gr.Textbox(visible=False)
            prompt_js = f"gradioApp().querySelector('#{tab}_prompt textarea')"
            neg_js = f"gradioApp().querySelector('#{tab}_neg_prompt textarea')"
            learned_btn.click(on_apply_learned, _js=f"(p, n, k) => [{prompt_js}.value, {neg_js}.value, k]",
                              inputs=[new_prompt, new_neg, auto_lock], outputs=[opt_md, new_prompt, new_neg]
                              ).then(None, inputs=[new_prompt, new_neg], _js=f"""(pr, ng) => {{
                    if (pr) {{ const box = {prompt_js}; box.value = pr; updateInput(box); }}
                    if (ng) {{ const box = {neg_js}; box.value = ng; updateInput(box); }}
                }}""")
            # writes the changed prompt / negative prompt ("" = unchanged) into the webui, then generates
            then_js = f"""(go, pr, ng) => {{
                    if (pr) {{ const box = {prompt_js}; box.value = pr; updateInput(box); }}
                    if (ng) {{ const box = {neg_js}; box.value = ng; updateInput(box); }}
                    if (go) setTimeout(() => gradioApp().getElementById('{tab}_generate').click(), 100);
                }}"""
            # the prompt boxes are read in the browser (inputs 3 and 7 are placeholders for them)
            optimize_btn.click(on_optimize,
                               _js=f"(...a) => {{ a[3] = {prompt_js}.value; a[7] = {neg_js}.value; return a; }}",
                               inputs=[edits, locked, to_prompt, new_prompt, remove_thr, use_rephrase, neg_on, new_neg,
                                       vocab, qual_on, auto_lock, adh_on, order_on, timing_on, layers],
                               outputs=[edits, opt_md, view, go, new_prompt, new_neg, layers]).then(
                None, inputs=[go, new_prompt, new_neg], _js=then_js)

            def on_filter(mode, n, thr, edit_text, sel):
                view_filter[tab] = (("top", max(1, int(n or 10))) if mode == "top N" else
                                    ("below", float(thr) / 100) if mode == "below threshold" else None)
                return view_html(None, edit_text, sel or None)

            for comp in (show_mode, top_n, strength_thr):
                (comp.release if comp is strength_thr else comp.change)(
                    on_filter, inputs=[show_mode, top_n, strength_thr, edits, selected], outputs=[view],
                    show_progress="hidden")

            def on_remove_weak(thr, lock_text, prompt, edit_text):
                """Take the comma phrases whose words all have less than thr % of the attention in the shown
                word map out of the prompt; locked words and words a keep rule protects count as strong.
                Only writes the prompt box, does not generate."""
                last = core.last_wordmaps.get(tab)
                if not last or not last["items"]:
                    return "No word map yet: generate an image first.", gr.update(), ""
                words = last["items"][min(wordmap_shown.get(tab, 0), len(last["items"]) - 1)][1]
                R = rules_mod.current().for_prompt(prompt or "")
                allowed = promptedit.prompt_words(prompt or "")
                weak = [w for w in wordmap.weak_words(words, float(thr) / 100, split_words(lock_text) + R.kept())
                        if w in allowed]
                if not weak:
                    return (f"No word of the prompt is below {float(thr):g}% of the attention (locked and kept "
                            "words are left out).", gr.update(), "")
                new, gone, partial = promptedit.drop_weak_phrases(prompt or "", weak, wordmap.STOPWORDS)
                if not gone:
                    return (f"Words below {float(thr):g}% ({', '.join(weak)}) are only in phrases with stronger words: "
                            "nothing removed (a phrase goes only when all its words are weak).", gr.update(), "")
                removed_words.setdefault(tab, set()).update(w for w in weak if w not in promptedit.prompt_words(new))
                kept = (f"\n\nKept (phrase has stronger words): {', '.join(f'`{x}`' for x in partial[:8])}"
                        if partial else "")
                return (f"🗑 Removed {len(gone)} phrase{'s' if len(gone) != 1 else ''} whose words all have less than "
                        f"{float(thr):g}% of the attention: {', '.join(f'**{x}**' for x in gone)}.{kept}\n\nGenerate to "
                        "see the result (Ctrl+Z in the prompt box undoes it).", view_html(None, edit_text), new)

            remove_weak.click(on_remove_weak, _js=f"(t, l, p, e) => [t, l, {prompt_js}.value, e]",
                              inputs=[strength_thr, locked, new_prompt, edits], outputs=[opt_md, view, new_prompt]
                              ).then(None, inputs=[new_prompt], _js=f"""(pr) => {{
                    if (pr) {{ const box = {prompt_js}; box.value = pr; updateInput(box); }}
                }}""")
            auto_outs = [edits, opt_md, view, go, new_prompt, new_neg, auto_btn, layers]
            eval_btn.click(on_evaluate, outputs=[opt_md, view])
            up_btn.click(lambda: on_rate(1), outputs=[opt_md], show_progress="hidden")
            down_btn.click(lambda: on_rate(-1), outputs=[opt_md], show_progress="hidden")
            auto_btn.click(on_auto, _js=f"(...a) => {{ a[3] = {prompt_js}.value; a[10] = {neg_js}.value; return a; }}",
                           inputs=[edits, locked, to_prompt, new_prompt, auto_thr, auto_rounds, auto_eval,
                                   remove_thr, use_rephrase, neg_on, new_neg, vocab, qual_on, auto_lock, adh_on,
                                   order_on, timing_on, layers, auto_btn],
                           outputs=auto_outs).then(None, inputs=[go, new_prompt, new_neg], _js=then_js)
            auto_step_btn.click(auto_step, _js=f"(...a) => {{ a[3] = {prompt_js}.value; a[4] = {neg_js}.value; return a; }}",
                                inputs=[edits, locked, to_prompt, new_prompt, new_neg, layers],
                                outputs=auto_outs).then(None, inputs=[go, new_prompt, new_neg], _js=then_js)

            # ---- profiles ----------------------------------------------------------------

            def do_refresh(sel):
                names = core.list_steers()
                return gr.update(choices=names, value=[x for x in (sel or []) if x in names])

            def show_info(sel):
                lines = []
                for name in sel or []:
                    try:
                        st = core.load_steer(name)
                    except Exception as e:
                        lines.append(f"- {name}: {e}")
                        continue
                    m = st.meta
                    pos_n, neg_n = st.counts
                    prompts = json.loads(m.get("pos_prompts", "[]")) or [m.get("prompt", "")]
                    focus = m.get("focus") or "whole image"
                    lines.append(f"- **{name}** (focus: {focus}): {m.get('model', '?')} ({m.get('arch', '?')}), "
                                 f"{pos_n} positive / {neg_n} negative, {m.get('buckets', '?')} buckets, "
                                 f"prompt: _{(prompts[-1] if prompts else '')[:120]}_")
                return "\n".join(lines)

            def lists(sel, prof):
                names = core.list_steers()
                return (gr.update(choices=names, value=[x for x in (sel or []) if x in names]),
                        gr.update(choices=names, value=prof if prof in names else None))

            def image_of(g):
                rec = core.last_recording.get(tab)
                g = int(g if g is not None else -1)
                return -1 if not rec or g < rec["first"] else g - rec["first"]

            def run(fn):
                def wrapped(*a):
                    *a, sel, prof = a
                    try:
                        msg, prof = fn(*a, prof)
                    except Exception as e:
                        msg = f"⚠️ {e}"
                    return (msg, *lists(sel, prof))
                return wrapped

            def do_merge(g, imp, w, prof):
                return merge(tab, prof, imp, image_of(g), w), prof

            def do_create(g, imp, w, name, prof):
                msg = merge(tab, name, imp, image_of(g), w, create=True)
                return msg, msg.split("**")[1]

            def do_skip(prof):
                return skip(tab), prof

            def do_undo(prof):
                if not prof:
                    raise ValueError("pick the profile to undo")
                return undo(prof), prof

            def do_delete(prof):
                if not prof:
                    raise ValueError("pick the profile to delete")
                core.delete_profile(prof)
                return f"deleted '{prof}'", None

            outs = [status, steers, profile]
            js = "(g, ...rest) => [selected_gallery_index(), ...rest]"
            merge_btn.click(run(do_merge), _js=js, inputs=[gallery_idx, impact, weight, steers, profile], outputs=outs)
            create_btn.click(run(do_create), _js=js, inputs=[gallery_idx, impact, weight, new_name, steers, profile],
                             outputs=outs)
            skip_btn.click(run(do_skip), inputs=[steers, profile], outputs=outs)
            undo_btn.click(run(do_undo), inputs=[steers, profile], outputs=outs)
            delete_btn.click(run(do_delete), inputs=[steers, profile], outputs=outs)
            refresh_p.click(lambda sel, prof: lists(sel, prof), inputs=[steers, profile], outputs=[steers, profile])
            refresh.click(do_refresh, inputs=[steers], outputs=[steers])
            steers.change(show_info, inputs=[steers], outputs=[info])

        self.infotext_fields = [(strength, "ActSteer strength"), (edits, "ActSteer word edits")]
        # order kept for API callers: new arguments are appended
        return [record, profile, buckets, auto, steers, strength, mode, cond_only, start, end, blocks,
                rec_focus, focus_override, edits, wordmap_on, edit_start, auto_clear, record_hist, learn_apply,
                layers, place_start, rule_layers, rp_layers]

    def before_process(self, p, *args):
        self._seed_and_learn(p, *args)
        self._rp_regions(p, args)

    def _rp_regions(self, p, args):
        """Layers as Regional Prompter regions: a word moved to background / foreground / on <word> / a part
        (clothes, hair, face, skin, body) leaves the main prompt and gets its own BREAK region, "main prompt
        BREAK sea, background" / "... BREAK lace, dress", in RP's Prompt mode (the region acts where the last
        word of its part, the target, gets attention in the main prompt; "background" and part names without
        a word in the prompt are added to it). Runs in before_process, before any script's process() reads the
        prompt, and switches RP on through p.script_args. Other targets (left, away from ...) stay on
        act-steer's own attention masking (RP has one mode per generation), which masks parts the same way."""
        self._rp_orig, self._rp_words = None, set()
        if len(args) <= 21 or state.fix is not None:
            return
        prompt = p.prompt or ""
        layers = [] if fresh_prompt.get(_tab(p)) else parse_layers(args[19])
        if args[21]:
            own = {w.lower() for w, _, _ in layers}
            layers += [e for e in parse_layers(", ".join(rules_mod.current().for_prompt(prompt).placements(prompt)))
                       if e[0].lower() not in own]
        placed = {w for ws, _, _ in layers for w in re.findall(r"\w+", ws.lower())}
        parts = {seglayers.canonical(t) for _, t, _ in layers} - {None}
        if parts:  # a part (clothes, hair ...) needs a word of its own in the prompt, as RP needs "background"
            new, added = seglayers.ensure_words(prompt, parts, promptedit.prompt_words(prompt), placed)
            if added:
                self._rp_orig, p.prompt, prompt = prompt, new, new
                print(f"[ActSteer] layers: added {', '.join(added)} to the prompt (the area of that part)")
        if len(args) <= 22 or not args[22] or not p.scripts:
            return
        rp = next((s for s in p.scripts.alwayson_scripts if s.title() == "Regional Prompter"), None)
        if rp is None or rp.args_to - rp.args_from < 16:
            return
        if any(k in prompt for k in ("BREAK", "ADDCOL", "ADDROW", "ADDCOMM", "ADDBASE", "ADDP")):
            return  # a hand-written RP prompt: left as it is
        words_in = promptedit.prompt_words(prompt)
        in_order = re.findall(r"\w+", re.sub(r"<[^>]*>", " ", prompt).lower())
        persons = rules_mod.PERSONS.split("|")
        wanted = []
        for words, target, _ in layers:
            layer = seglayers.canonical(target)
            if layer:  # RP masks the region by the attention of its target word: the part's own word
                own = set(re.findall(r"\w+", words.lower()))
                tgt = next(iter(seglayers.words_in(layer, in_order, own)), None)
            elif target == "background":
                tgt = "background"
            elif target == "foreground":
                tgt = next((w for w in persons if w in words_in), None)
            elif target.startswith("on ") and set(re.findall(r"\w+", target[3:].lower())) <= words_in:
                tgt = target[3:].strip()
            else:
                tgt = None
            if tgt:
                wanted.append((words, tgt))
        # RP keys a region's mask by its target tokens (two regions on one target share a mask and the last
        # one is dropped) and skips a target missing from the main prompt (the masks of all later regions
        # shift by one): one region per target, and a word that leaves the main prompt can't be a target.
        # A word whose target left stays on act-steer's own masking.
        skip = set()
        while True:
            base, by_tgt = prompt, {}
            for k, (words, tgt) in enumerate(wanted):
                if k in skip:
                    continue
                base, moved = _take_out(base, words)
                if moved:
                    by_tgt.setdefault(tgt.lower(), [tgt, []])[1].append((k, moved))
            left = promptedit.prompt_words(base) | {"background"}
            lost = {k for _, items in by_tgt.values() for k, _ in items
                    if not set(re.findall(r"\w+", wanted[k][1].lower())) <= left}
            if not lost:
                break
            skip |= lost
        regions = [(", ".join(m for _, m in items), tgt) for tgt, items in by_tgt.values()]
        self._rp_words = {wanted[k][0].lower() for _, items in by_tgt.values() for k, _ in items}
        if not regions:
            return
        if any(t == "background" for _, t in regions) and "background" not in promptedit.prompt_words(base):
            base = f"{base}, background"
        self._rp_orig = self._rp_orig or prompt
        p.prompt = base + "".join(f" BREAK {moved}, {tgt}" for moved, tgt in regions)
        sa = list(p.script_args)
        i = rp.args_from
        # rp.ui order: active, debug, tab, mmode, xmode, pmode, ratios, bratios, usebase, usecom, usencom, calcmode,
        # options, lnter, lnur, threshold, ...
        sa[i], sa[i + 2], sa[i + 5] = True, "Prompt", "Prompt"
        sa[i + 8] = sa[i + 9] = sa[i + 10] = False
        sa[i + 11] = "Attention"
        if not str(sa[i + 15] or "").strip():
            sa[i + 15] = "0.4"
        p.script_args = type(p.script_args)(sa) if isinstance(p.script_args, tuple) else sa
        p.extra_generation_params["ActSteer RP regions"] = "; ".join(f"{m} -> {t}" for m, t in regions)
        print(f"[ActSteer] layers as Regional Prompter regions: {p.prompt}")

    def _seed_and_learn(self, p, *args):
        edits = parse_edits(args[13]) if len(args) > 13 else []
        once = keep_seed.pop(_tab(p), False)
        auto_clear = bool(args[16]) if len(args) > 16 else True
        apply_learn = bool(args[18]) if len(args) > 18 else True
        self._job_prompt = p.prompt  # what Optimize wrote (an experiment's result is checked against it)
        last = core.last_images.get(_tab(p))
        new_image = state.fix is None and (last is None or is_new_prompt(last["prompt"], p.prompt))
        if apply_learn and new_image and not getattr(p, "is_api", False):
            # the next image of a new prompt starts with what earlier optimisations proved
            p.prompt, p.negative_prompt, applied = apply_learned(p.prompt, p.negative_prompt)
            if applied:
                p.extra_generation_params["ActSteer learned"] = "; ".join(applied)
                learned_applied[_tab(p)] = applied
                print(f"[ActSteer] learned improvements applied: {'; '.join(applied)}")
        if auto_clear and state.fix is None and last and is_new_prompt(last["prompt"], self._job_prompt):
            # a new image: the word edits, locks and seed of the previous one don't belong to it
            tab = _tab(p)
            fresh_prompt[tab] = True
            for d in (rephrase_tried, removed_words, last_eval, auto_runs):
                d.pop(tab, None)
            if edits:
                print("[ActSteer] new prompt: the word edits of the previous image are not used")
            return
        layered = bool(parse_layers(args[19])) if len(args) > 19 else False
        if not (edits or once or layered) or state.fix is not None or (p.seed is not None and int(p.seed) != -1):
            return
        last = core.last_images.get(_tab(p))
        if last and last["items"]:
            i = min(wordmap_shown.get(_tab(p), 0), len(last["items"]) - 1)
            p.seed, p.subseed = last["items"][i][1], last["items"][i][2]
            print(f"[ActSteer] word edits: keeping the seed of the shown image ({p.seed})")

    def _finish_experiment(self, p, images, infos, first):
        """The image of an optimisation is there: measure it and store the experiment (learn)."""
        tab = _tab(p)
        if learn.pending_id(tab) is None or not self._words:
            return
        i = sorted(self._words)[0]
        if i >= len(images):
            return
        prompt = getattr(self, "_job_prompt", p.prompt)
        words = self._words[i]
        R = rules_mod.current().for_prompt(prompt)
        o = wordmap.max_conflict(words, [], promptedit.prompt_words(prompt), R)[0]
        flags = sum(1 for q in quality.word_quality(images[i], words).values() if q["flags"])
        exp_id = learn.finish(tab, prompt, {"conflict": round(o, 4), "flags": flags, "quality": None, "adherence": None},
                              version=wordmap_version.get(tab))
        if exp_id is None:
            return
        last_finished[tab] = (wordmap_version.get(tab), exp_id)
        run = auto_runs.get(tab)
        if not (run and run.get("evaluate")):  # auto mode evaluates this image itself
            evaluate_later(exp_id, "after", images[i], infos[first + i] if first + i < len(infos) else "")

    def process(self, p, record=False, steer_name="", buckets=4, auto="ask", steers=None, strength=1.0,
                mode="add", cond_only=False, start=0.0, end=1.0, blocks="all", rec_focus="", focus_override="",
                edits_text="", wordmap_on=True, edit_start=0.0, auto_clear=True, record_hist=True, learn_apply=True,
                layers_text="", place_start=0.15, rule_layers=True, rp_layers=True, *args):
        state.reset()
        self._paused = None
        # normal generations go into the SD MCP history (API jobs: the SD MCP records its own)
        self._record_hist = bool(record_hist) and evaluate.ENABLED and state.fix is None \
            and not getattr(p, "is_api", False)
        saved_images.clear()
        state.last_applied = None
        self._job = None
        self._words = {}
        self._neg_words = {}
        self._early_words = {}
        self._lora_maps = {}
        if state.fix is not None:  # area-fix passes: don't replace the kept influence or the word map
            record = False
            wordmap_on = False
        edits = [] if fresh_prompt.get(_tab(p)) else parse_edits(edits_text)
        state.wordmap_on = bool(wordmap_on)
        state.word_tokens, state.word_maps, state.word_texts, state.block_maps = {}, {}, [], {}
        state.neg_word_tokens, state.neg_word_maps, state.neg_word_texts = {}, {}, []
        self._edits = edits
        self._layers = [] if fresh_prompt.get(_tab(p)) or state.fix is not None else parse_layers(layers_text)
        if rule_layers and state.fix is None:  # global place rules; the Layers field wins for its words
            own = {w.lower() for w, _, _ in self._layers}
            ruled = [e for e in parse_layers(", ".join(rules_mod.current().for_prompt(p.prompt).placements(p.prompt)))
                     if e[0].lower() not in own]
            if ruled:
                print(f"[ActSteer] placement rules: {format_layers(ruled)}")
            self._layers += ruled
        rp_words = getattr(self, "_rp_words", set())
        if rp_words:
            self._layers = [e for e in self._layers if e[0].lower() not in rp_words]
        prompt_has_tag = any("<steer:" in pr for pr in (p.all_prompts or [p.prompt]))
        if not (record or steers or prompt_has_tag or state.fix is not None or wordmap_on or edits or self._layers):
            return

        unet, forge = core.find_net(shared.sd_model)
        if unet is None:  # Flux, Qwen-Image ...: no UNet / Anima cross-attention to measure or steer
            name = type(shared.sd_model).__name__
            if name not in _unsupported_warned:
                _unsupported_warned.add(name)
                print(f"[ActSteer] {name} models are not supported (SD 1.x / SDXL / Anima): Activation "
                      "Steering is off for them")
            state.wordmap_on = False
            self._edits, self._layers = [], []
            return
        if core.is_dit(unet):  # Anima: the word map (and Optimize, which writes into the prompt) only
            dropped = [n for n, on in (("word edits", edits), ("layers", self._layers), ("profiles", steers or
                       prompt_has_tag), ("profile recording", record)) if on]
            if dropped and "dit" not in _unsupported_warned:
                _unsupported_warned.add("dit")
                print(f"[ActSteer] Anima: only the word map works ({', '.join(dropped)} need a SD 1.x / SDXL UNet)")
            self._edits, self._layers, edits, steers, record, prompt_has_tag = [], [], [], [], False, False
            state.prompt_steers = []
            if not wordmap_on:
                return
        state.install(unet, forge)
        state.edit_start = float(edit_start)
        if edits:
            p.extra_generation_params["ActSteer word edits"] = format_edits(edits)
        state.place_start = float(place_start)
        if self._layers:
            p.extra_generation_params["ActSteer layers"] = format_layers(self._layers)
        state.mode = mode if mode in ("add", "gain") else "add"
        state.focus_override = clean_focus(focus_override)
        state.cond_only = bool(cond_only)
        state.start, state.end = float(start), float(end)
        state.blocks_spec = blocks or "all"
        try:
            state.block_pred = core.parse_blocks(blocks)
        except ValueError as e:
            print(f"[ActSteer] {e}; using all blocks")

        loaded = []
        for name in steers or []:
            try:
                s = core.load_steer(name)
            except Exception as e:
                print(f"[ActSteer] {e}")
                continue
            _check_model(s)
            loaded.append((s, float(strength)))
        state.ui_steers = loaded
        if loaded or prompt_has_tag:
            p.extra_generation_params["ActSteer"] = ", ".join(f"{s.name}:{w:g}" for s, w in loaded) or "prompt"
            p.extra_generation_params["ActSteer mode"] = (
                f"{state.mode}, {'cond' if state.cond_only else 'both'}, {state.start:g}-{state.end:g}, {blocks}"
                + (f", focus {state.focus_override}" if state.focus_override else ""))

        if record:
            default_name = core.safe_name(re.sub(r"<[^>]*>", "", p.prompt)[:40])
            self._job = {"name": steer_name, "default_name": default_name, "auto": auto, "buckets": int(buckets)}
            self._job["focus"] = clean_focus(rec_focus)
            state.recorder = core.Recorder(int(buckets), focus=self._job["focus"])
            if auto in ("positive", "negative"):
                p.extra_generation_params["ActSteer added"] = f"{core.safe_name(steer_name or default_name)} ({auto})"

    def _pause(self):
        """ADetailer's early postprocess call: off during its inpaint passes (word edits, word maps, layers
        and recording belong to the full image, not to a face crop with another prompt)."""
        if getattr(self, "_paused", None) is None:
            self._paused = {k: getattr(state, k) for k in PAUSED_FIELDS}
            state.recorder, state.ui_steers, state.wordmap_on, state.word_edits, state.placements = \
                None, [], False, [], []

    def _resume(self):
        paused = getattr(self, "_paused", None)
        for k, v in (paused or {}).items():
            setattr(state, k, v)
        self._paused = None

    def process_batch(self, p, *args, **kwargs):
        self._resume()  # the next batch after ADetailer's passes
        state.image_base = p.iteration * p.batch_size
        if not state.active:
            return
        prompts = kwargs.get("prompts") or p.prompts or [p.prompt]

        tokens = {}
        for focus in state.needed_focus():
            tokens[focus] = [focus_positions(pr, focus, p.steps) for pr in prompts]
            if not any(len(t) for t in tokens[focus]) and focus not in state._warned_focus:
                state._warned_focus.add(focus)
                print(f"[ActSteer] focus '{focus}' is not in the prompt: profiles focused on it do nothing")
        state.set_focus_tokens(tokens)

        state.word_edits = []
        for word, factor in getattr(self, "_edits", []):
            per_img = [focus_positions(pr, word, p.steps) for pr in prompts]
            if any(len(t) for t in per_img):
                state.word_edits.append((per_img, factor))
            else:
                print(f"[ActSteer] word edit '{word}' is not in the prompt, ignored")

        layers = getattr(self, "_layers", [])
        if state.wordmap_on or layers:  # layers need the live word maps for subject / on / away from
            state.word_tokens, state.word_texts = _word_tokens(prompts, p.steps, "#w")
            state.word_maps, state.block_maps = {}, {}
            state.early_word_maps, state._early_done = {}, False
        if state.wordmap_on:
            negs = list(getattr(p, "negative_prompts", None) or []) or [p.negative_prompt] * len(prompts)
            state.neg_word_tokens, state.neg_word_texts = _word_tokens(negs, p.steps, "#n")
            state.neg_word_maps = {}
        state.placements = []
        persons = set(rules_mod.PERSONS.split("|"))
        for words, target, strength in layers:
            per_img = [focus_positions(pr, words, p.steps) for pr in prompts]
            if not any(len(t) for t in per_img):
                print(f"[ActSteer] layer: '{words}' is not in the prompt, ignored")
                continue
            kind, ref = target, ""
            if target.startswith("on "):
                kind, ref = "on", target[3:]
            elif target.startswith("away from "):
                kind, ref = "away", target[10:]
            own = {w.lower() for w in re.findall(r"\w+", words)}
            keys_of = lambda row, names: [k for k, t in row if k and t.lower() in names]  # noqa: E731
            if ref:
                names = {w.lower() for w in re.findall(r"\w+", ref)}
            elif kind in ("background", "foreground"):
                names = persons  # the subject: person words, else the strongest word (core)
            else:
                names = set()
            refs = [keys_of(row, names) for row in state.word_texts] if names else None
            layer = seglayers.canonical(target)
            minus = None
            if layer:  # the part's own words, like Regional Prompter's Prompt mode (lib_act_steer/seglayers.py)
                kind = "seg"
                refs = [[k for k, t in row if k and t.lower() not in own and seglayers.words_in(layer, [t])]
                        for row in state.word_texts]
                minus = [[k for k, t in row if k and any(seglayers.words_in(m, [t]) for m in seglayers.minus(layer))]
                         for row in state.word_texts] if seglayers.minus(layer) else None
            state.placements.append({"per_img": per_img, "target": kind, "ref": refs, "layer": layer, "minus": minus,
                                     "exclude": {k for row in state.word_texts for k in keys_of(row, own)},
                                     "strength": float(strength)})

    def postprocess_batch(self, p, *args, **kwargs):
        if not state.wordmap_on:
            return
        base = p.iteration * p.batch_size
        for img_local in range(p.batch_size):
            self._words[base + img_local] = wordmap.harvest(img_local)
            self._neg_words[base + img_local] = wordmap.harvest(img_local, negative=True)
            self._early_words[base + img_local] = wordmap.harvest(img_local, early=True)
            self._lora_maps[base + img_local] = wordmap.harvest_blocks(img_local)

    def postprocess(self, p, processed, *args):
        if processed is None or not processed.images:
            # ADetailer calls postprocess with an empty result before its inpaint passes
            # (need_call_postprocess): the job is not over, the real call comes after them
            self._pause()
            return
        self._resume()
        job = getattr(self, "_job", None)
        if state.fix is None and processed is not None and processed.images:
            first = getattr(processed, "index_of_first_image", 0)
            images = processed.images[first:]
            seeds = lambda i: (p.all_seeds[i] if i < len(p.all_seeds) else p.seed,  # noqa: E731
                               p.all_subseeds[i] if i < len(p.all_subseeds) else p.subseed)
            core.last_images[_tab(p)] = {
                "first": first, "prompt": getattr(self, "_rp_orig", None) or p.prompt,
                "items": [(im, *seeds(i)) for i, im in enumerate(images[:len(p.all_seeds) or 1])],
            }
            if state.wordmap_on and self._words:
                infos = processed.infotexts or []
                core.last_wordmaps[_tab(p)] = {
                    "first": first,
                    "items": [(images[i], self._words[i], *seeds(i)) for i in sorted(self._words) if i < len(images)],
                    "lora": [self._lora_maps.get(i) for i in sorted(self._words) if i < len(images)],
                    "infotexts": [infos[first + i] if first + i < len(infos) else ""
                                  for i in sorted(self._words) if i < len(images)],
                    "neg": [self._neg_words.get(i) for i in sorted(self._words) if i < len(images)],
                    "early": [self._early_words.get(i) for i in sorted(self._words) if i < len(images)],
                    "prompts": [(p.all_prompts[i] if i < len(p.all_prompts or []) else p.prompt,
                                 p.all_negative_prompts[i] if i < len(p.all_negative_prompts or [])
                                 else p.negative_prompt) for i in sorted(self._words) if i < len(images)],
                    "interrupted": bool(getattr(shared.state, "interrupted", False)
                                        or getattr(shared.state, "stopping_generation", False)),
                }
                wordmap_shown[_tab(p)] = 0
                removed_words.pop(_tab(p), None)
                wordmap_version[_tab(p)] = wordmap_version.get(_tab(p), 0) + 1
                self._finish_experiment(p, images, infos, first)
        if getattr(self, "_record_hist", False) and state.fix is None and processed is not None and processed.images:
            first = getattr(processed, "index_of_first_image", 0)
            paths = list(saved_images)
            if not paths:  # "Save images" is off: record copies with their generation info
                infos = processed.infotexts or []
                paths = [evaluate.save_for_eval(im, infos[first + i] if first + i < len(infos) else "",
                                                f"{_tab(p)}_hist{i}")
                         for i, im in enumerate(processed.images[first:])]
            saved_images.clear()
            threading.Thread(target=record_history, args=(paths,), daemon=True, name="act-steer-history").start()
        dump = os.environ.get("SD_ACT_STEER_DUMP_MAPS")
        if dump and state.maps:
            _dump_maps(dump, p)
        try:
            if job and state.recorder is not None:
                tab = _tab(p)
                samples = state.recorder.finish()
                core.last_recording[tab] = {
                    "samples": samples, "buckets": job["buckets"], "default_name": job["default_name"],
                    "first": getattr(processed, "index_of_first_image", 0),
                    "prompts": {i: (p.all_prompts[i] if i < len(p.all_prompts) else p.prompt) for i in samples},
                    "meta": dict(_model_meta(), focus=job["focus"], negative_prompt=p.negative_prompt, steps=p.steps,
                                 cfg_scale=p.cfg_scale, width=p.width, height=p.height),
                    "rated": {},
                }
                if job["auto"] in ("positive", "negative"):
                    print("[ActSteer] " + merge(tab, job["name"] or job["default_name"], job["auto"]).replace("**", ""))
        except Exception as e:
            print(f"[ActSteer] recording not kept: {e}")
        finally:
            self._job = None
            state.reset()
            state.wordmap_on = False
            state.word_tokens, state.word_maps, state.word_texts, state.block_maps = {}, {}, [], {}
            state.neg_word_tokens, state.neg_word_maps, state.neg_word_texts = {}, {}, []


# what makes state.active, kept aside while ADetailer inpaints (ActSteer._pause)
PAUSED_FIELDS = ("recorder", "ui_steers", "wordmap_on", "word_edits", "placements")

# files the webui saved during the current job (not grids), for the SD MCP history
saved_images = []


def _on_image_saved(params):
    fn = getattr(params, "filename", "") or ""
    if fn and not os.path.basename(fn).startswith("grid") and params.p is not None:
        saved_images.append(os.path.abspath(fn))


def record_history(paths):
    """Send generated images to the SD MCP (record_images: metrics + tags); runs in a thread."""
    try:
        out = evaluate.call_tool("record_images", {"paths": paths, "source": "webui"}, timeout=300)
        if out.get("errors"):
            print(f"[ActSteer] history: {out['errors']}")
        quality._history_cache.clear()  # the next Optimize sees the new images
    except Exception as e:
        print(f"[ActSteer] SD MCP history not reachable ({type(e).__name__}: {e}); image not recorded")


def _before_ui():
    extra_networks.register_extra_network(ExtraNetworkSteer())


def _app_started(demo, app):
    from fastapi import Body, HTTPException

    @app.get("/actsteer/list")
    def list_steers():
        out = []
        for name in core.list_steers():
            try:
                s = core.load_steer(name)
                pos_n, neg_n = s.counts
                out.append({"name": name, **s.meta, "positive": pos_n, "negative": neg_n})
            except Exception as e:
                out.append({"name": name, "error": str(e)})
        return {"dir": core.STEER_DIR, "steers": out}

    def call(fn, *a, **k):
        try:
            return {"status": fn(*a, **k).replace("**", "")}
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/actsteer/version")
    def version_api():
        """Counter per tab, raised when a job's word map is ready (the UI polls it to refresh the view)."""
        return {"txt2img": wordmap_version.get("txt2img", 0), "img2img": wordmap_version.get("img2img", 0)}

    @app.get("/actsteer/wordmap")
    def wordmap_api(tab: str = "txt2img", image: int = 0, html_view: bool = False):
        """Word maps of image `image` (0-based) of the last job: per word colour, share, PNG overlay."""
        last = core.last_wordmaps.get(tab)
        if not last or not (0 <= image < len(last["items"])):
            raise HTTPException(status_code=404, detail="no word map for that image")
        img, words = last["items"][image][:2]
        lora = (last.get("lora") or [None] * (image + 1))[image]
        if html_view:
            return {"html": wordmap.render_html(img, words, lora=lora)}
        plan, merged = wordmap.optimize(words)
        keys, ov, _ = wordmap.relations(words)
        groups = wordmap.color_words(words)
        return {"words": [{"word": t, "share": round(sh, 4),
                           "color": wordmap._css(groups[t.lower()][0]) if m is not None else None,
                           "overlay": wordmap.overlay_png(m, groups[t.lower()][0]) if m is not None else None}
                          for t, m, sh, _ in words],
                "relations": {"words": keys, "overlap": [[round(float(v), 3) for v in row] for row in ov]},
                "check": [{"kind": k, "message": msg} for k, msg in wordmap.check_prompt(words)],
                "lora_blocks": [{**r, "lbw": {n: wordmap.lbw_suggestion([r], n, lora["blocks"]) for n in r["added"]}}
                                for r in wordmap.lora_blocks(words, lora)],
                "lora_tune": [{"lora": c["name"], "old": c["old"], "tag": c["text"], "why": c["why"]}
                              for c in loratune.plan(last["prompts"][image][0], words, lora)],
                "optimize": {"plan": [{"word": w, "factor": f, "reason": why} for w, f, why in plan],
                             "edits": format_edits(list(merged.items()))}}

    @app.post("/actsteer/train_separation")
    def train_separation_api(images: int = Body(40), steps: int = Body(600), rank: int = Body(4),
                             keep: float = Body(septrain.KEEP_WEIGHT), prompts: str = Body(""),
                             name: str = Body("act_steer_separation")):
        """Separation LoRA: generate `images` images from `prompts` (lines, {pair} / {place}), train, save."""
        st = train_separation({}, images, steps, rank, keep, prompts, False, name, "")
        return {"status": st.get("md", "").replace("**", ""), "path": st.get("path"), "result": st.get("result")}

    @app.post("/actsteer/merge")
    def merge_api(profile: str = Body(""), impact: str = Body("positive"), image: int = Body(-1),
                  weight: float = Body(1.0), create: bool = Body(False), tab: str = Body("txt2img")):
        """Merge image `image` (0-based, -1 = all) of the last job into `profile`; create=True for a new one."""
        return call(merge, tab, profile, impact, image, weight, create)

    @app.post("/actsteer/skip")
    def skip_api(tab: str = Body("txt2img", embed=True)):
        return call(skip, tab)

    @app.post("/actsteer/undo")
    def undo_api(profile: str = Body(..., embed=True)):
        return call(undo, profile)

    @app.post("/actsteer/delete")
    def delete_api(profile: str = Body(..., embed=True)):
        return call(lambda: (core.delete_profile(profile), f"deleted '{profile}'")[1])


def _model_loaded(sd_model):
    if state.active and getattr(state, "hooked", False):  # the refiner was loaded during a job
        unet, forge = core.find_net(sd_model)
        if unet is not None:
            state.switch_unet(unet, forge)
            return
        print(f"[ActSteer] the refiner ({type(sd_model).__name__}) is not supported: act-steer stops at the "
              "switch")
        state.refined = True
    state.remove()


script_callbacks.on_before_ui(_before_ui)
script_callbacks.on_app_started(_app_started)
script_callbacks.on_image_saved(_on_image_saved)
script_callbacks.on_cfg_denoiser(state.on_cfg_denoiser)
script_callbacks.on_cfg_denoised(state.on_cfg_denoised)
script_callbacks.on_model_loaded(_model_loaded)
script_callbacks.on_script_unloaded(state.remove)
