"""Separation LoRA: a small LoRA on the cross-attention to_q / to_k of the UNet, trained so nouns of different
prompt phrases attend to different areas by themselves (what the word edits of Optimize do per prompt).

1. generate: n images from training prompts (built-in noun pairs and / or your lines); while they are
   sampled, the UNet inputs of the prompt rows (latent, timestep, context, y) are kept at a few steps.
2. train: the UNet runs on those inputs (no gradients through it); a hook on every cross-attention computes
   its attention with the LoRA added to to_q / to_k from the layer's detached input, so each layer gets its
   own gradient and nothing has to be kept for a backward pass through the UNet. Loss per sample:
     overlap: the word maps (head-averaged attention of each word, all layers, half latent resolution)
       -> concentration areas (smooth version of wordmap.concentration) -> soft Jaccard of every pair of
       nouns of different phrases (wordmap.runs: "red dress" is one phrase; rule `allow a ~ b` pairs and
       attention sinks left out); hinge above MARGIN, so a part inside a whole (hair / girl) costs little
     keep: KL divergence of the attention from the model's own, so the rest of the prompt stays as it was
   Steps up to EARLY_END of the sampling, where the layout forms, are drawn 1 / LATE_WEIGHT times as often.
3. save: kohya-style lora_unet_* keys in the LoRA folder, usable as <lora:name:1>.

SD 1.x / SDXL UNets (attn2 to_q / to_k) and Anima's DiT (cross_attn q_proj / k_proj of every block, word positions
= its T5 tokens, text.word_groups).
"""

import gc
import math
import os
import random
import re
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from lib_act_steer import core, rules as rules_mod
from lib_act_steer.text import word_groups
from lib_act_steer.wordmap import STOPWORDS

NOUNS = ["cat", "dog", "horse", "fox", "owl", "rabbit", "bird", "girl", "boy", "old man", "woman", "knight",
         "robot", "car", "bicycle", "boat", "house", "tree", "lamp", "chair", "teapot", "guitar", "umbrella",
         "castle", "flower vase", "book", "apple", "clock", "cake", "lantern", "backpack", "mushroom", "dragon"]
COLOURS = ["red", "blue", "green", "yellow", "white", "black", "orange", "purple", "pink", "brown"]
PLACES = ["in a park", "on a street", "in a kitchen", "on a beach", "in a forest", "in a living room",
          "on a table", "in the snow", "in a garden", "in a library", "on a hill at sunset", "in a city at night"]
PERSONS = ["girl", "boy", "woman", "old man", "child", "lady"]
GARMENTS = ["dress", "jacket", "coat", "hoodie", "sweater", "kimono", "uniform", "apron", "cape", "scarf"]
ANIMALS = ["cat", "dog", "horse", "fox", "owl", "rabbit", "bird", "deer", "parrot", "turtle"]
THINGS = ["lamp", "umbrella", "guitar", "teapot", "book", "clock", "lantern", "backpack", "vase", "bicycle",
          "suitcase", "kite", "balloon", "basket", "candle", "mirror"]
DEFAULT_LINES = "{pair}, {place}"
# several subjects with attributes; every phrase ends in its noun (the noun of a phrase is its last word,
# so no verbs like "holding" between them) and joins with "with" / "and" / commas
COMPLEX_LINES = """{person}, {person}, {place}
{person} with a {colour} {thing}, {animal}, {place}
{person} and {person} with a {colour} {thing}, {colour} {animal}, {place}
{colour} {animal}, {colour} {animal}, {colour} {thing}, {place}
{person}, {person}, {colour} {thing} in the background, {place}
{person}, {colour} {animal}, {colour} {thing}, {colour} {thing}, {place}
{person} with a {colour} {thing}, {person} with a {colour} {thing}, {place}
{pair}, {colour} {thing} in the background, {place}"""

# the words of the places: the setting the subjects stand in, so they are never a pair to separate
PLACE_WORDS = {w for p in PLACES for w in p.split()} - {"a", "an", "the", "on", "in", "at"}

MARGIN = 0.15        # pair overlap below this costs nothing
KEEP_WEIGHT = 1.0    # weight of the attention KL against the model's own attention
SINK_SHARE = 0.3     # a word with more of the attention is an attention sink (wordmap.color_words)
STEP_FRACS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6)  # sampling progress at which the UNet inputs are kept
EARLY_END = 0.3      # the layout forms up to here ...
LATE_WEIGHT = 0.25   # ... later steps are drawn this much less often
EVAL_SAMPLES = 16
MIN_FREE_GB = 4.0    # stop when less RAM is available (memory_watch / systemd-oomd kill the webui below ~1 GB)


# ---- memory -----------------------------------------------------------------------------------------------

class LowMemory(RuntimeError):
    pass


def available_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float("inf")


def check_ram(what):
    free = available_gb()
    if free < MIN_FREE_GB:
        raise LowMemory(f"only {free:.1f} GB RAM available {what} (needs {MIN_FREE_GB:.0f} GB); "
                        "close other programs or train with fewer images")


def free_gpu():
    """Gives the training's cached GPU memory back, so the next generation loads the model fully into VRAM
    instead of offloading it to RAM."""
    gc.collect()
    try:
        from backend import memory_management
        memory_management.soft_empty_cache(force=True)
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---- prompts ----------------------------------------------------------------------------------------------

def _pair(rng):
    n = 3 if rng.random() < 0.25 else 2
    things = rng.sample(NOUNS, n)
    parts = [f"a {rng.choice(COLOURS)} {t}" if rng.random() < 0.7 else f"a {t}" for t in things]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


class _Draw:
    """Random words for one prompt, none twice (a noun in two phrases would be one word map)."""

    def __init__(self, rng):
        self.rng, self.used = rng, set()

    def pick(self, pool):
        free = [w for w in pool if w not in self.used] or pool
        w = self.rng.choice(free)
        self.used.add(w)
        return w

    def person(self):
        return (f"a {self.pick(PERSONS)} with {self.pick(COLOURS)} hair in a {self.pick(COLOURS)} "
                f"{self.pick(GARMENTS)}")


def fill(line, rng):
    """{pair}, {place}, {person} (a person with hair and clothing colours), {animal}, {thing}, {colour}."""
    d = _Draw(rng)
    subs = {"pair": lambda: _pair(rng), "place": lambda: rng.choice(PLACES), "person": d.person,
            "animal": lambda: d.pick(ANIMALS), "thing": lambda: d.pick(THINGS), "colour": lambda: d.pick(COLOURS)}
    return re.sub(r"\{(pair|place|person|animal|thing|colour)\}", lambda m: subs[m.group(1)](), line)


def make_prompts(lines, n, current="", use_current=False, seed=None):
    """n training prompts: the lines (one per line, cycled; placeholders see fill); empty -> DEFAULT_LINES.
    use_current: every other prompt is the current prompt."""
    rng = random.Random(seed)
    lines = [ln.strip() for ln in (lines or "").splitlines() if ln.strip()] or [DEFAULT_LINES]
    out = []
    for i in range(int(n)):
        if use_current and current.strip() and i % 2 == 1:
            out.append(current.strip())
            continue
        ln = lines[(i // (2 if use_current and current.strip() else 1)) % len(lines)]
        out.append(fill(ln, rng))
    return out


def noun_columns(prompt, steps=20):
    """Content words of the prompt -> ([(word, positions LongTensor)], [(i, j)] noun pairs to separate).
    Words of the training places (PLACE_WORDS) are never in a pair.

    Words of one phrase run ("red dress") are kept together: only its last word is a noun and the run is one
    group; pairs are nouns of different runs, without the pairs a rule allows."""
    words = word_groups(prompt, steps)
    runs, cur = [], []
    for text, pos in words:
        key = text.lower()
        if pos is not None and key not in STOPWORDS and not key.startswith("<"):
            cur.append((key, pos))
            continue
        if cur:
            runs.append(cur)
        cur = []
    if cur:
        runs.append(cur)
    cols, run_of = [], {}
    for r, run in enumerate(runs):
        for key, pos in run:
            if key in run_of:  # same word twice: its positions are one map
                c = [i for i, (k, _) in enumerate(cols) if k == key][0]
                cols[c] = (key, torch.cat([cols[c][1], pos]))
                continue
            run_of[key] = r
            cols.append((key, pos))
    nouns = {run[-1][0] for run in runs}
    R = rules_mod.current().for_prompt(prompt)
    pairs = []
    for i, (a, _) in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j][0]
            if a in nouns and b in nouns and run_of[a] != run_of[b] and not R.allowed(a, b) \
                    and a not in PLACE_WORDS and b not in PLACE_WORDS:
                pairs.append((i, j))
    return cols, pairs


# ---- data: UNet inputs captured while the training images are sampled ---------------------------------------

class Capture:
    """Forward pre-hook on the UNet: keeps the prompt rows' inputs at STEP_FRACS of the sampling."""

    def __init__(self, unet):
        self.samples = []
        self.meta = None  # (prompt, columns, pairs) of the image being sampled
        self.seen = set()
        self.ctx = None   # (context on the device, its CPU copy) of the image being sampled
        self.handle = unet.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _pre(self, module, args, kwargs):
        if self.meta is None:
            return
        from modules import shared
        steps = max(1, shared.state.sampling_steps)
        step = shared.state.sampling_step
        slot = next((i for i, f in enumerate(STEP_FRACS) if step == min(steps - 1, round(f * steps))), None)
        if slot is None or slot in self.seen:
            return
        x = args[0] if args else kwargs.get("x")
        t = args[1] if len(args) > 1 else kwargs.get("timesteps")
        ctx, y = kwargs.get("context"), kwargs.get("y")
        if not torch.is_tensor(x) or not torch.is_tensor(t) or not torch.is_tensor(ctx):
            return
        cu = (kwargs.get("transformer_options") or {}).get("cond_or_uncond") or [0]
        if 0 not in cu or x.shape[0] % len(cu):
            return
        m = x.shape[0] // len(cu)
        r = cu.index(0) * m  # the first prompt row of this call (one image per job)
        self.seen.add(slot)
        half = lambda a: a[r:r + 1].detach().to("cpu", torch.float16)  # half the RAM of fp32; _run casts back
        c = ctx[r:r + 1]
        if self.ctx is None or self.ctx[0].shape != c.shape or not torch.equal(self.ctx[0], c):
            self.ctx = (c.detach().clone(), half(ctx))  # one copy per image (Anima: 512 x 1024 per sample)
        self.samples.append({"x": half(x), "t": t[r:r + 1].detach().cpu(), "context": self.ctx[1],
                             "y": half(y) if torch.is_tensor(y) else None,
                             "prompt": self.meta[0], "cols": self.meta[1], "pairs": self.meta[2],
                             "frac": STEP_FRACS[slot]})

    def start(self, prompt, cols, pairs):
        self.meta, self.seen, self.ctx = (prompt, cols, pairs), set(), None

    def close(self):
        self.handle.remove()
        self.meta = self.ctx = None


THUMB = 256
MAX_THUMBS = 32


def generate(prompts, width, height, steps=20, cfg=None, sampler=None, negative="lowres, worst quality",
             progress=None):
    """Samples one image per prompt with the loaded model -> (samples, [PIL thumbnails of the first MAX_THUMBS
    images], number of images). progress(i, n) after each. Raises LowMemory when RAM runs short."""
    from modules import processing, sd_models, shared
    sd_models.forge_model_reload()  # the checkpoint is loaded on the first job
    unet, _ = core.find_net(shared.sd_model)
    if unet is None:
        raise ValueError("training needs an SD 1.x / SDXL or Anima model")
    dit = core.is_dit(unet)  # Anima (flow model): plain Euler, lower CFG
    cfg = cfg if cfg is not None else 4.0 if dit else 5.0
    sampler = sampler or ("Euler" if dit else "Euler a")
    cap = Capture(unet)
    images, n_done = [], 0
    try:
        for i, prompt in enumerate(prompts):
            if shared.state.interrupted:
                break
            check_ram(f"after {n_done} training images")
            cols, pairs = noun_columns(prompt, steps)
            if not pairs:
                print(f"[ActSteer] training prompt without two separate nouns, skipped: {prompt}")
                continue
            cap.start(prompt, cols, pairs)
            p = processing.StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model, prompt=prompt, negative_prompt=negative, steps=steps, cfg_scale=cfg,
                width=width, height=height, sampler_name=sampler, seed=-1, batch_size=1, n_iter=1,
                do_not_save_samples=True, do_not_save_grid=True)
            p.scripts = None
            try:
                res = processing.process_images(p)
                if res.images:
                    n_done += 1
                    if len(images) < MAX_THUMBS:
                        im = res.images[0].copy()
                        im.thumbnail((THUMB, THUMB))
                        images.append(im)
                del res
            finally:
                p.close()
            if progress:
                progress(i + 1, len(prompts))
    finally:
        cap.close()
    return cap.samples, images, n_done


# ---- LoRA ---------------------------------------------------------------------------------------------------

def _proj_names(module):
    """(query, key, value, out) projection names of a cross-attention: UNet attn2 or Anima's cross_attn."""
    if hasattr(module, "to_q"):
        return "to_q", "to_k", "to_v", "to_out"
    return "q_proj", "k_proj", "v_proj", "output_proj"


def attn_layers(unet):
    """[(module path, cross-attention)]: every attn2 of a UNet, or the cross_attn of every Anima block (the
    adapter's cross-attention belongs to the text encoder, not to this network)."""
    if core.is_dit(unet):
        return [(f"blocks.{i}.cross_attn", b.cross_attn) for i, b in enumerate(unet.blocks)]
    return [(name, m) for name, m in unet.named_modules() if name.endswith(".attn2") and hasattr(m, "to_q")]


class SepLora(torch.nn.Module):
    def __init__(self, layers, rank, device):
        super().__init__()
        self.rank = rank
        self.names = [n for n, _ in layers]
        self.down = torch.nn.ParameterDict()
        self.up = torch.nn.ParameterDict()
        for name, m in layers:
            for lin in _proj_names(m)[:2]:
                mod = getattr(m, lin)
                key = f"{name}.{lin}".replace(".", "_")
                d = torch.empty(rank, mod.in_features, device=device)
                torch.nn.init.kaiming_uniform_(d, a=math.sqrt(5))
                self.down[key] = torch.nn.Parameter(d)
                self.up[key] = torch.nn.Parameter(torch.zeros(mod.out_features, rank, device=device))

    def delta(self, name, lin, x):
        key = f"{name}.{lin}".replace(".", "_")
        return (x @ self.down[key].t()) @ self.up[key].t()

    def state(self, alpha=None):
        """kohya-style keys (Forge: modules_forge/packages/comfy/lora.py model_lora_keys_unet maps
        lora_unet_<path with _> to diffusion_model.<path>, for the UNet and for Anima's blocks.N.cross_attn)."""
        out = {}
        for key in self.down.keys():
            base = f"lora_unet_{key}"
            out[f"{base}.lora_down.weight"] = self.down[key].detach().half().cpu().contiguous()
            out[f"{base}.lora_up.weight"] = self.up[key].detach().half().cpu().contiguous()
            out[f"{base}.alpha"] = torch.tensor(float(alpha or self.rank), dtype=torch.float16)
        return out


def _rms(t, norm):
    """Anima's per-head RMSNorm in fp32 (the module's weight may be bf16 and the input fp32)."""
    w = getattr(norm, "weight", None)
    t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + norm.eps)
    return t if w is None else t * w.to(t.device, torch.float32)


class Trainer:
    """Hooks on every cross-attention. Pass 1 (record): the network runs without gradients; each hook computes
    its attention with the LoRA added to the query / key projections, adds its word maps to the sum and gives
    the layer's output with the LoRA to the rest of the network; the layer's input is kept (detached).
    Pass 2 (backward): the loss gradient with respect to the summed maps is spread over the layers by
    recomputing one layer at a time with gradients, so only one layer's attention is ever in memory (Anima:
    52 layers x 4096 positions x 512 tokens x 16 heads would not fit at once)."""

    def __init__(self, unet, lora):
        self.unet, self.lora = unet, lora
        self.layers = attn_layers(unet)
        self.handles = [m.register_forward_hook(self._hook(name), with_kwargs=True) for name, m in self.layers]
        self.sel = None        # [T, W] word selection of the current sample
        self.hw = None         # latent size of the current sample
        self.scale_lora = 1.0  # 0: the model alone (evaluation baseline)
        self.keep = False      # keep each layer's input for pass 2
        self.reset()

    def reset(self):
        self.maps, self.n_layers, self.kept = None, 0, []

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _hook(self, name):
        def hook(module, args, kwargs, out):
            x = args[0] if args else kwargs.get("x")
            ctx = kwargs.get("context", args[1] if len(args) > 1 else None)
            if self.sel is None or not torch.is_tensor(ctx) or not torch.is_tensor(x) or x.dim() != 3 \
                    or kwargs.get("value") is not None:
                return None
            if ctx.dim() > 3:  # Anima: context [B, 1, T, D]
                ctx = ctx.flatten(1, -2)
            hw = self._hw(x.shape[1])
            if hw is None:
                return None
            x, ctx = x.detach(), ctx.detach()
            with torch.no_grad():
                logp, v = self._logp(name, module, x, ctx, self.scale_lora, value=bool(self.scale_lora))
                m = self._maps(logp.exp(), hw)
            self.maps = m if self.maps is None else self.maps + m
            self.n_layers += 1
            if self.keep:
                self.kept.append((name, module, x, ctx, hw))
            if not self.scale_lora:
                return None
            with torch.no_grad():  # the rest of the network sees the attention with the LoRA
                B, h, L, _ = logp.shape
                o = logp.exp().to(v.dtype) @ v
                o = o.transpose(1, 2).reshape(B, L, -1)
                return getattr(module, _proj_names(module)[3])(o).to(out.dtype)
        return hook

    def _hw(self, n):
        H0, W0 = self.hw
        for f in (1, 2, 4, 8, 16):
            H, W = -(-H0 // f), -(-W0 // f)
            if H * W == n:
                return H, W
        return None

    def _logp(self, name, module, x, ctx, scale_lora, value=False):
        """log attention [B, heads, L, T] with the LoRA at scale_lora (and the split values if value)."""
        qn, kn, vn, _ = _proj_names(module)
        with torch.no_grad():
            q, k = getattr(module, qn)(x).float(), getattr(module, kn)(ctx).float()
            v = getattr(module, vn)(ctx) if value else None
        if scale_lora:
            q = q + scale_lora * self.lora.delta(name, qn, x.float())
            k = k + scale_lora * self.lora.delta(name, kn, ctx.float())
        B, L, C = q.shape
        heads = getattr(module, "heads", None) or module.n_heads
        D = C // heads
        q = q.reshape(B, L, heads, D)
        k = k.reshape(B, k.shape[1], heads, D)
        if not hasattr(module, "to_q"):  # Anima: RMSNorm per head on q and k (no rope in the cross-attention)
            q, k = _rms(q, module.q_norm), _rms(k, module.k_norm)
        logp = (q.transpose(1, 2) @ k.permute(0, 2, 3, 1)).mul(D ** -0.5).log_softmax(-1)
        if v is not None:
            v = v.reshape(B, -1, heads, D).transpose(1, 2)
        return logp, v

    def _maps(self, probs, hw):
        """head-averaged attention of each selected word -> [B, W, H/2, W/2] (half latent resolution)."""
        B = probs.shape[0]
        sel = self.sel.to(device=probs.device)
        T = probs.shape[-1]
        m = (probs.mean(1) @ sel[:T]).transpose(1, 2).reshape(B, -1, *hw)
        target = (-(-self.hw[0] // 2), -(-self.hw[1] // 2))
        if tuple(hw) != target:
            m = F.interpolate(m, size=target, mode="bilinear", align_corners=False)
        return m

    def backward(self, grad_maps, keep_weight):
        """Pass 2: d loss / d LoRA from the gradient of the summed maps plus the KL to the model's own
        attention (averaged over the layers). -> mean KL."""
        n = max(1, len(self.kept))
        kl_sum = 0.0
        for name, module, x, ctx, hw in self.kept:
            with torch.enable_grad():
                logp = self._logp(name, module, x, ctx, 1.0)[0]
                with torch.no_grad():
                    logp0 = self._logp(name, module, x, ctx, 0.0)[0]
                kl = (logp0.exp() * (logp0 - logp)).sum(-1).mean()
                m = self._maps(logp.exp(), hw)
                surrogate = (m * grad_maps).sum() + keep_weight * kl / n
            surrogate.backward()
            kl_sum += float(kl.detach())
            del logp, logp0, m, surrogate
        self.kept = []
        return kl_sum / n


def soft_concentration(m):
    """Smooth wordmap.concentration: attention 1.2x .. 2.5x the word's own image average -> 0 .. 1."""
    lift = m / (m.mean((-1, -2), keepdim=True) + 1e-8)
    return torch.sigmoid((lift - 1.85) * 4.0)


def pair_overlaps(maps, pairs):
    """maps [W, H, W] summed attention -> (overlap per pair [P], sink mask of the words)."""
    mass = maps.sum((-1, -2))
    share = mass / (mass.sum() + 1e-8)
    sink = (share > SINK_SHARE) & (maps.shape[0] > 3)
    conc = soft_concentration(maps)
    ov = []
    for i, j in pairs:
        a, b = conc[i], conc[j]
        ov.append(torch.minimum(a, b).sum() / (torch.maximum(a, b).sum() + 1e-8))
    return torch.stack(ov), sink


def _run(unet, trainer, s, device, dtype):
    """Pass 1 on one sample -> (summed maps [W, h, w] of the image, number of layers)."""
    cols = s["cols"]
    T = s["context"].shape[-2]
    sel = torch.zeros(T, len(cols))
    for c, (_, pos) in enumerate(cols):
        pos = pos[pos < T]
        sel[pos, c] = 1.0
    trainer.sel = sel
    trainer.hw = tuple(s["x"].shape[-2:])
    trainer.reset()
    kw = {"context": s["context"].to(device, dtype)}
    if s["y"] is not None:
        kw["y"] = s["y"].to(device, dtype)
    with torch.no_grad():
        unet(s["x"].to(device, dtype), s["t"].to(device), transformer_options={}, **kw)
    return trainer.maps[0], max(1, trainer.n_layers)


def _overlap(maps_sum, n_layers, pairs):
    ov, sink = pair_overlaps(maps_sum / n_layers, pairs)
    keep = torch.tensor([not (sink[i] or sink[j]) for i, j in pairs], device=ov.device)
    return ov, keep


def evaluate(unet, trainer, samples, device, dtype):
    """Mean worst pair overlap of the samples with the LoRA (scale 1) and without (scale 0)."""
    out = {}
    trainer.keep = False
    for name, sc in (("before", 0.0), ("after", 1.0)):
        trainer.scale_lora = sc
        worst = []
        for s in samples:
            maps, n = _run(unet, trainer, s, device, dtype)
            ov, keep = _overlap(maps, n, s["pairs"])
            if keep.any():
                worst.append(float(ov[keep].max()))
        out[name] = sum(worst) / max(1, len(worst))
    trainer.scale_lora = 1.0
    trainer.sel = None
    trainer.reset()
    return out


def train(samples, steps=600, rank=4, lr=3e-4, margin=MARGIN, keep_weight=KEEP_WEIGHT, progress=None, seed=None):
    """-> (SepLora, {"before", "after", "steps", "low_memory"}). progress(step, steps, sep, kl, secs) every few
    steps; stops early when shared.state.interrupted or less than MIN_FREE_GB RAM is available."""
    from backend import memory_management
    from modules import shared
    unet, _ = core.find_net(shared.sd_model)
    if unet is None:
        raise ValueError("training needs an SD 1.x / SDXL or Anima model")
    memory_management.load_model_gpu(shared.sd_model.forge_objects.unet)
    device = memory_management.get_torch_device()
    dtype = shared.sd_model.forge_objects.unet.model.computation_dtype
    if seed is not None:
        torch.manual_seed(seed)
    rng = random.Random(seed)
    lora = SepLora(attn_layers(unet), rank, device)
    trainer = Trainer(unet, lora)
    held = rng.sample(samples, min(EVAL_SAMPLES, len(samples)))
    weights = [1.0 if s.get("frac", 0.0) <= EARLY_END else LATE_WEIGHT for s in samples]
    opt = torch.optim.AdamW(lora.parameters(), lr=lr, weight_decay=0.0)
    hist, t0, low = [], time.time(), False
    try:
        before = evaluate(unet, trainer, held, device, dtype)["before"]
        for step in range(int(steps)):
            if shared.state.interrupted:
                break
            if step % 25 == 0 and available_gb() < MIN_FREE_GB:
                low = True  # stop like Stop: what is trained so far is kept
                break
            s = rng.choices(samples, weights)[0]
            opt.zero_grad(set_to_none=True)
            trainer.keep = True
            maps, n = _run(unet, trainer, s, device, dtype)
            trainer.keep = False
            maps = maps.detach().requires_grad_(True)
            ov, keep = _overlap(maps, n, s["pairs"])
            sep = F.relu(ov - margin)[keep].mean() if keep.any() else ov.sum() * 0
            sep.backward()
            g = maps.grad.unsqueeze(0)  # pass 1 summed the maps of image 0 of a batch of 1
            kl = trainer.backward(g, keep_weight)
            opt.step()
            hist.append((float(sep), kl))
            trainer.reset()
            if progress and (step % 10 == 9 or step + 1 == steps):
                k = hist[-50:]
                progress(step + 1, steps, sum(a for a, _ in k) / len(k), sum(b for _, b in k) / len(k),
                         time.time() - t0)
        ev = evaluate(unet, trainer, held, device, dtype)
    finally:
        trainer.close()
        trainer.lora = trainer.unet = None
        trainer.reset()
        del opt, trainer
        lora.requires_grad_(False)
        lora.cpu()  # tiny; keeps nothing on the GPU
        free_gpu()
    return lora, {"before": before, "after": ev["after"], "steps": len(hist), "low_memory": low}


def save(lora, name, meta):
    from modules import shared
    folder = shared.cmd_opts.lora_dir
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{core.safe_name(name)}.safetensors")
    save_file(lora.state(), path, metadata={k: str(v) for k, v in meta.items()})
    try:  # the Lora tab lists it without a manual refresh
        import networks
        networks.list_available_networks()
    except Exception:
        pass
    return path
