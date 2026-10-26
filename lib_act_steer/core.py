"""Activation steering for the UNet.

Record: during a generation, every UNet block output (input_blocks.N,
middle_block, output_blocks.N) is averaged over its spatial positions, per
channel, per image, separately for the prompt (cond) and negative (uncond)
rows of the CFG batch and per denoising-progress bucket.

Rate: a recorded image is added to a steer file as a positive or a negative
example. The file keeps running means per side, so it can be built up from
many images over time.

Steer direction per block/bucket: the push of an image is cond - uncond
(what its prompt switched on, measured on the same latent). The steer is
mean push of the positives minus mean push of the negatives, so positives
strengthen and negatives weaken what they activated. Gain mode uses the
matching ratio of RMS values (pos ratio / neg ratio) instead.

Apply: the direction is added back onto the block outputs, scaled by a
strength (gain: each channel scaled by ratio ** strength).

Attention-guided (focus) profiles: with focus words, e.g. "eyes", the
statistics are weighted by the cross-attention map of those words, so the
profile describes only that region. When applied, the offset is weighted by
the same words' attention map in the new image, so it lands wherever the
eyes are in that image. The words have to be in the prompt.
"""

import json
import os
import re
import shutil
import time
import weakref

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from lib_act_steer import tokenmaps

STEER_DIR = os.environ.get("SD_ACT_STEER_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "models", "ActSteer")

FORMAT = "act_steer/2"
SIDES = ("pos", "neg")
STATS = ("mc", "qc", "mu", "qu")  # mean / mean-square of cond rows, of uncond rows
MAX_PROMPTS = 20
# attention map -> region weights: normalised map below LOW counts 0, above HIGH counts 1
MAP_LOW, MAP_HIGH = 0.1, 0.5
MAP_DECAY = 0.6
EARLY_FRAC = 0.3  # word maps are also kept at this fraction of the steps (the layout has formed)


def _row_slices(rows):
    """Contiguous runs of row indices -> [slice]: indexing a GPU tensor with a Python list copies the
    indices from host memory, and on ROCm that copy waits for the GPU to finish all queued work (one
    stall per attention layer and step, 2026-10-24); slices need no copy."""
    out, start = [], None
    for i, j in enumerate(rows):
        if start is None:
            start = j
        if i + 1 == len(rows) or rows[i + 1] != j + 1:
            out.append(slice(start, j + 1))
            start = None
    return out


def _take(t, rows):
    parts = [t[s] for s in _row_slices(rows)]
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def _put(t, rows, value):
    i = 0
    for s in _row_slices(rows):
        n = s.stop - s.start
        t[s] = value[i:i + n]
        i += n


def list_steers():
    if not os.path.isdir(STEER_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(STEER_DIR) if f.endswith(".safetensors"))


def steer_path(name):
    return os.path.join(STEER_DIR, f"{name}.safetensors")


def safe_name(name):
    name = re.sub(r"[^\w\-. ]+", "_", name).strip(" ._")
    return name[:80] or "steer"


def bucket_of(step, steps, buckets):
    if buckets <= 1 or not steps:
        return 0
    return min(buckets - 1, max(0, int(step * buckets / steps)))


def parse_blocks(spec):
    """'all' | 'IN0-8,M,OUT0-11' (case-insensitive) -> predicate on block name."""
    spec = (spec or "all").strip().lower()
    if spec in ("", "all", "*"):
        return lambda name: True
    wanted = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if part in ("m", "mid", "middle"):
            wanted.add("middle_block")
            continue
        m = re.fullmatch(r"(in|out)(\d+)?(?:-(\d+))?", part)
        if not m:
            raise ValueError(f"block spec '{part}' not understood (use IN0-8, M, OUT3, ...)")
        prefix = "input_blocks" if m.group(1) == "in" else "output_blocks"
        if m.group(2) is None:
            wanted.update(f"{prefix}.{i}" for i in range(32))
        else:
            a = int(m.group(2))
            b = int(m.group(3)) if m.group(3) else a
            wanted.update(f"{prefix}.{i}" for i in range(a, b + 1))
    return lambda name: name in wanted


def _read_meta(path):
    with safe_open(path, framework="pt") as f:
        return f.metadata() or {}


def _derive(side_stats):
    """side_stats[side][stat] = tensor or None -> (delta, gain) or (None, None)."""
    # Each side's push is cond - uncond on the same latent; comparing raw cond
    # activations of different images instead mostly measures the noisy input
    # and steers the wrong way.
    pushes = []
    for side, sign in (("pos", 1.0), ("neg", -1.0)):
        s = side_stats.get(side, {})
        if all(s.get(k) is not None for k in STATS):
            pushes.append((sign * (s["mc"] - s["mu"]),
                           (s["qc"].sqrt() / s["qu"].sqrt().clamp_min(1e-6)).pow(sign)))
    if not pushes:
        return None, None
    delta = sum(d for d, _ in pushes)
    gain = pushes[0][1] if len(pushes) == 1 else pushes[0][1] * pushes[1][1]
    return delta, gain


class Steer:
    """One loaded steer file."""

    def __init__(self, name):
        self.name = name
        self.path = steer_path(name)
        tensors = load_file(self.path, device="cpu")
        self.meta = _read_meta(self.path)
        self.buckets = int(self.meta.get("buckets", 1))
        # delta[block][bucket] / gain[block][bucket] -> [C]
        self.delta, self.gain = {}, {}
        self.mean = {}  # mean block output per channel (for baking gain mode)
        raw = {}
        for key, t in tensors.items():
            parts = key.split("|")
            if parts[0] in ("delta", "gain"):  # act_steer/1: precomputed
                kind, block, b = parts
                (self.delta if kind == "delta" else self.gain).setdefault(block, {})[int(b)] = t.float()
            elif parts[0] in SIDES and not parts[1].endswith("#n"):
                side, stat, block, b = parts
                raw.setdefault((block, int(b)), {}).setdefault(side, {})[stat] = t.float()
        for (block, b), side_stats in raw.items():
            delta, gain = _derive(side_stats)
            if delta is not None:
                self.delta.setdefault(block, {})[b] = delta
                self.gain.setdefault(block, {})[b] = gain
                side = side_stats.get("pos") or side_stats.get("neg")
                if side.get("mc") is not None:
                    self.mean.setdefault(block, {})[b] = side["mc"]
        self._cache = {}

    @property
    def focus(self):
        return self.meta.get("focus", "")

    @property
    def counts(self):
        return int(self.meta.get("pos_n", 0)), int(self.meta.get("neg_n", 0))

    def get(self, kind, block, bucket, device, dtype):
        table = self.delta if kind == "delta" else self.gain
        per_block = table.get(block)
        if not per_block:
            return None
        if bucket not in per_block:
            bucket = min(per_block, key=lambda b: abs(b - bucket))
        key = (kind, block, bucket, device, dtype)
        t = self._cache.get(key)
        if t is None:
            t = per_block[bucket].to(device=device, dtype=dtype)
            self._cache[key] = t
        return t


FIX_MODES = ("remove responsible words", "weaken prompt in area")


def pick_words(tok_in, tok_out, word_pos, max_words=6):
    """Prompt tokens the painted area attends to much more than the rest of the image.

    tok_in / tok_out: mean attention per context position inside / outside the mask.
    word_pos: LongTensor of positions holding real prompt words (no BOS/EOS/padding).
    Returns positions sorted by score.
    """
    if tok_in is None or not len(word_pos):
        return []
    a_in, a_out = tok_in[word_pos], tok_out[word_pos]
    score = a_in * (a_in / a_out.clamp_min(1e-6) - 1).clamp_min(0)
    if float(score.max()) <= 0:
        return []
    keep = (score >= 0.3 * score.max()).nonzero().flatten()
    keep = keep[score[keep].argsort(descending=True)][:max_words]
    return [int(word_pos[i]) for i in keep]


_steer_cache = {}


def load_steer(name):
    path = steer_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"steer '{name}' not found in {STEER_DIR}")
    mtime = os.path.getmtime(path)
    hit = _steer_cache.get(name)
    if hit and hit[0] == mtime:
        return hit[1]
    s = Steer(name)
    _steer_cache[name] = (mtime, s)
    return s


class Recorder:
    """Per-image activation statistics of one generation job."""

    def __init__(self, buckets, focus=""):
        self.buckets = buckets
        self.focus = focus  # focus words: statistics weighted by their attention map
        # stats[image][block][bucket] = [sum [4, C], n_cond, n_unc]
        self.stats = {}

    def add(self, block, bucket, h, n_cond, n_unc, vrows, image_base, maps=None):
        """maps: per row an [H, W] weight map or None (focus recording; rows without a map are skipped)."""
        hf = h.detach().float()
        if maps is None:
            mean = hf.mean(dim=(2, 3))            # [k, C]
            msq = hf.pow(2).mean(dim=(2, 3))      # [k, C]
        for j in range(h.shape[0]):
            if maps is not None:
                m = maps[j]
                if m is None or float(m.sum()) <= 0:
                    continue
                wsum = m.sum()
                mean_j = (hf[j] * m).sum(dim=(1, 2)) / wsum
                msq_j = (hf[j].pow(2) * m).sum(dim=(1, 2)) / wsum
            else:
                mean_j, msq_j = mean[j], msq[j]
            r = vrows[j]
            if r < n_cond:
                cond, img = True, (r * n_unc // n_cond if n_unc else r)
            else:
                cond, img = False, r - n_cond
            per = self.stats.setdefault(image_base + img, {}).setdefault(block, {})
            s = per.get(bucket)
            if s is None:
                s = per[bucket] = [torch.zeros(4, h.shape[1], device=h.device), 0, 0]
            if cond:
                s[0][0] += mean_j
                s[0][1] += msq_j
                s[1] += 1
            else:
                s[0][2] += mean_j
                s[0][3] += msq_j
                s[2] += 1

    def finish(self):
        """-> {image: {(block, bucket): {stat: cpu tensor or None}}}, frees GPU memory."""
        out = {}
        for img, blocks in self.stats.items():
            sample = {}
            for block, per in blocks.items():
                for b, (s, nc, nu) in per.items():
                    if not nc:
                        continue
                    s = s.cpu()
                    sample[(block, b)] = {
                        "mc": s[0] / nc, "qc": s[1] / nc,
                        "mu": s[2] / nu if nu else None, "qu": s[3] / nu if nu else None,
                    }
            if sample:
                out[img] = sample
        self.stats = {}
        return out


def add_samples(name, side, samples, meta, buckets, prompts=(), weight=1.0, create=False):
    """Merge recorded images into profile (steer file) `name` as positive or negative examples.

    samples: list of {(block, bucket): {stat: tensor or None}}, one per image.
    weight: how much each image counts against what the profile already holds.
    create: refuse if the profile already exists.
    The previous version is kept as <name>.safetensors.undo for undo_last().
    Returns (name, pos_n, neg_n).
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    if not samples:
        raise ValueError("no recorded images to add")
    weight = float(weight)
    if weight <= 0:
        raise ValueError("weight must be > 0")
    os.makedirs(STEER_DIR, exist_ok=True)
    name = safe_name(name)
    path = steer_path(name)
    tensors, old_meta = {}, {}
    existed = os.path.exists(path)
    if existed and create:
        raise ValueError(f"profile '{name}' already exists: merge into it or pick another name")
    if existed:
        tensors = load_file(path, device="cpu")
        old_meta = _read_meta(path)
        if any(k.split("|")[0] in ("delta", "gain") for k in tensors):
            raise ValueError(f"'{name}' is an old single-recording steer; rate into a new name")
        have, new = old_meta.get("focus", ""), str(meta.get("focus", ""))
        if have != new:
            raise ValueError(f"'{name}' is focused on '{have or 'whole image'}', this image was recorded "
                             f"with focus '{new or 'whole image'}': merge into a matching profile or create a new one")
    file_buckets = int(old_meta.get("buckets", buckets))

    for sample in samples:
        for (block, b), stats in sample.items():
            fb = b if file_buckets == buckets else min(file_buckets - 1, int((b + 0.5) * file_buckets / buckets))
            for stat in STATS:
                x = stats.get(stat)
                if x is None:
                    continue
                key = f"{side}|{stat}|{block}|{fb}"
                nkey = f"{side}|{stat}#n|{block}|{fb}"
                n = float(tensors[nkey]) if nkey in tensors else 0.0
                old = tensors.get(key)
                tensors[key] = (x.float() if old is None or n == 0
                                else old + (x.float() - old) * (weight / (n + weight))).contiguous()
                tensors[nkey] = torch.tensor(n + weight)

    new_meta = dict(old_meta)
    for k, v in meta.items():
        new_meta.setdefault(k, str(v))
    counts = {s: int(old_meta.get(f"{s}_n", 0)) for s in SIDES}
    counts[side] += len(samples)
    hist = json.loads(old_meta.get(f"{side}_prompts", "[]"))
    for p in prompts:
        if p and p not in hist:
            hist.append(p)
    new_meta.update({
        "format": FORMAT, "buckets": str(file_buckets),
        "pos_n": str(counts["pos"]), "neg_n": str(counts["neg"]),
        f"{side}_prompts": json.dumps(hist[-MAX_PROMPTS:]),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "blocks": str(len({k.split('|')[2] for k in tensors})),
    })
    new_meta.setdefault("created", new_meta["updated"])
    undo = path + ".undo"
    if existed:
        shutil.copy2(path, undo)
    else:
        open(undo, "wb").close()  # empty: undo deletes the new profile
    save_file(tensors, path, metadata=new_meta)
    _steer_cache.pop(name, None)
    return name, counts["pos"], counts["neg"]


def undo_last(name):
    """Restore profile `name` to before its last merge. Returns a status line."""
    name = safe_name(name)
    path, undo = steer_path(name), steer_path(name) + ".undo"
    if not os.path.exists(undo):
        raise ValueError(f"nothing to undo for '{name}'")
    if os.path.getsize(undo) == 0:
        os.remove(undo)
        if os.path.exists(path):
            os.remove(path)
        msg = f"removed '{name}' (it was created by the last merge)"
    else:
        os.replace(undo, path)
        msg = f"'{name}' restored to before the last merge"
    _steer_cache.pop(name, None)
    return msg


def delete_profile(name):
    name = safe_name(name)
    for p in (steer_path(name), steer_path(name) + ".undo"):
        if os.path.exists(p):
            os.remove(p)
    _steer_cache.pop(name, None)


def soft_map(acc):
    """Accumulated attention -> [H, W] in 0..1, lightly blurred, background level removed."""
    # light blur: single-pixel attention spikes become regions
    m = F.avg_pool2d(acc[None, None].float(), 3, stride=1, padding=1, count_include_pad=False)
    m = F.avg_pool2d(m, 3, stride=1, padding=1, count_include_pad=False)[0, 0]
    m = (m - m.mean()).clamp_min(0)
    return m / m.max().clamp_min(1e-8)


class SteerState:
    def __init__(self):
        self.unet = None
        self.handles = []
        # what the last generation applied; kept after it ends so it can be baked
        self.last_applied = None
        # area fix (scripts/fix_areas.py), set around its own process_images calls, with
        # "mask" ([H, W] 0..1, image space) and "feather":
        #   {"kind": "record"}: collects "tok_in"/"tok_out", mean cross-attention per context
        #       position inside/outside the mask
        #   {"kind": "words", "positions", "w", "start"}: those positions' attention scaled by
        #       (1 - w) inside the mask, in every cross-attention layer
        #   {"kind": "guidance", "w", "start"}: prompt guidance scaled by (1 - w) inside the mask
        self.fix = None
        self._fix_masks = {}
        # word map (scripts/word_map.py): word key -> [positions per image of the batch],
        # word_maps[img in batch][word key] = accumulated attention like self.maps
        self.wordmap_on = False
        self.word_tokens = {}
        self.word_maps = {}
        self.word_texts = []  # per image of the batch: [(word key or None, text)]
        # the same for the negative prompt, from the uncond rows (sd-webui-act-steer optimises it too)
        self.neg_word_tokens = {}
        self.neg_word_maps = {}
        self.neg_word_texts = []
        # LoRA block analysis (word maps on and a LoRA loaded): per image of the batch
        # block_maps[img][(LoRA name or "" for all LoRAs, block id)][word key] = attention like
        # word_maps, in the first cross-attention of each block; "<name>" = to_q/to_k without that LoRA
        self.block_maps = {}
        self.attn_block = {}  # id(attn2 module) -> block id in LoRA Block Weight naming (IN04, M00, OUT02)
        self.reset()

    def reset(self):
        self.recorder = None
        self.ui_steers = []      # [(Steer, strength)]
        self.prompt_steers = []  # from <steer:name:w>
        self.mode = "add"
        self.cond_only = False
        self.start = 0.0
        self.end = 1.0
        self.block_pred = lambda name: True
        self.blocks_spec = "all"
        self.image_base = 0
        self.focus_override = ""
        # word edits: [([positions per image of the batch], influence factor)], from step fraction edit_start
        self.word_edits = []
        self.edit_start = 0.0
        # layers: [{"per_img": [positions per image], "target": background | foreground | left | right | top |
        # bottom | center | on | away | seg, "ref": [word map keys per image] (on / away / the subject; seg: the
        # part's words) or None, "layer": skin / clothes / hair / face / body (seg), "minus": [keys per image]
        # taken out of the area (skin: clothes, hair), "exclude": keys of placed words, "strength": 0..1}]:
        # the word's attention outside the target region is scaled by 1 - strength, from step fraction place_start
        self.placements = []
        self.place_start = 0.15
        self._place_cache = {}
        self.prompt_focus = {}   # steer name -> focus word(s) from <steer:name:w:focus>
        # focus words -> [token positions LongTensor per image of the batch]
        self.focus_tokens = {}
        # maps[img in batch][focus] = [H, W] attention of the focus words, summed over all
        # cross-attention layers at half latent resolution, decayed every step
        self.maps = {}
        self._map_cache = {}
        self._stacks = {}     # id(map dict of one image) -> its stacked buffer (_stack_add)
        self._sel_cache = {}  # (id(token groups), img, T, device) -> word columns (_columns)
        self._plan_cache = {}  # row layout -> (identity refs, plan, selection, row index) (_reduce_maps)
        self._pos_cache = {}   # (id(positions), T, device) -> positions on the GPU (_on_device)
        self.latent_hw = None
        self._warned_focus = set()
        self.lora_weights = {}  # LoRA name -> its prompt weight, for the lbw suggestion
        self._lora_w = {}  # (id(attn2), loaded LoRAs) -> [(LoRA name, to_q weight, to_k weight) without it]
        # per CFG-denoiser call
        self.n_cond = 1 << 30
        self.n_unc = 0
        self.row0 = 0
        # per model call: the A1111-style row of each row of the call ("virtual rows": prompt rows 0..n_cond-1,
        # then the negative ones). A1111 stacks them in that order across calls; Forge batches the negative
        # first or in separate calls and says which in transformer_options["cond_or_uncond"]
        self.vrows = []
        self.call_rows = 0
        self.step = 0
        self.steps = 0
        # word maps at EARLY_FRAC of the steps (lib_act_steer/timing.py): {img: {key: map}}, taken once per job
        self.early_word_maps = {}
        self._early_done = False
        # the refiner (or another model) took over during the job: the hooks moved to its UNet
        self.refined = False

    # the UNet the hooks are on, as a weak reference: act-steer must never keep a model alive (Forge frees the
    # base model when the refiner loads; a strong reference here kept ~5 GB of SDXL UNet in RAM)
    @property
    def unet(self):
        ref = getattr(self, "_unet_ref", None)
        return ref() if ref is not None else None

    @unet.setter
    def unet(self, unet):
        self._unet_ref = weakref.ref(unet) if unet is not None else None

    @property
    def active(self):
        return (self.recorder is not None or bool(self.ui_steers) or bool(self.prompt_steers)
                or self.fix is not None or self.wordmap_on or bool(self.word_edits) or bool(self.placements))

    def set_fix(self, fix):
        self.fix = fix
        self._fix_masks = {}

    def _fix_frac(self):
        return self.step / self.steps if self.steps else 0.0

    def _fix_mask(self, H, W, device):
        m = self._fix_masks.get((H, W))
        if m is None:
            m = F.interpolate(self.fix["mask"][None, None].float(), size=(H, W), mode="area")
            for _ in range(int(self.fix.get("feather", 0))):
                m = F.avg_pool2d(m, 3, stride=1, padding=1, count_include_pad=False)
            m = m[0, 0].clamp(0, 1).to(device)
            self._fix_masks[(H, W)] = m
        return m

    # ---- hooks -------------------------------------------------------------

    def install(self, unet, forge=False):
        if self.unet is unet and self.handles:
            return
        self.remove()
        self.unet = unet
        self.hooked = True
        self.forge = forge
        self.dit = is_dit(unet)
        if self.dit:  # Anima: word maps from the cross-attention of every other block (no UNet blocks)
            self.handles.append(unet.register_forward_pre_hook(self._unet_pre, with_kwargs=True))
            self.attn_block = {}
            for i, block in enumerate(unet.blocks):
                if i % 2 == 1:
                    self.handles.append(block.cross_attn.register_forward_hook(self._attn_hook, with_kwargs=True))
            return
        self.handles.append(unet.register_forward_pre_hook(self._unet_pre, with_kwargs=True))
        self.attn_block = {}
        for name, module in _blocks(unet):
            self.handles.append(module.register_forward_hook(self._make_block_hook(name)))
            for sub in module.modules():
                blocks = getattr(sub, "transformer_blocks", None)
                if blocks is not None and len(blocks) and hasattr(blocks[0], "attn2"):
                    self.attn_block.setdefault(id(blocks[0].attn2), lbw_block(name))
        # one cross-attention per SpatialTransformer is enough for the word maps;
        # the area fix acts on all of them
        for module in unet.modules():
            blocks = getattr(module, "transformer_blocks", None)
            if blocks is not None and len(blocks) and hasattr(blocks[0], "attn2"):
                self.handles.append(blocks[0].attn2.register_forward_hook(self._attn_hook, with_kwargs=True))
                for block in blocks:
                    self.handles.append(block.attn2.register_forward_hook(self._attn_fix_hook, with_kwargs=True))
                    self.handles.append(block.attn2.register_forward_hook(self._attn_edit_hook, with_kwargs=True))

    def steer_focus(self, steer):
        return self.focus_override or self.prompt_focus.get(steer.name) or steer.focus

    def needed_focus(self):
        groups = {self.steer_focus(s) for s, _ in self.ui_steers + self.prompt_steers}
        if self.recorder is not None:
            groups.add(self.recorder.focus)
        groups.discard("")
        return groups

    def set_focus_tokens(self, focus_tokens):
        """focus words -> [LongTensor of context positions per image of the batch]."""
        self.focus_tokens = focus_tokens
        self.maps = {}
        self._map_cache = {}

    # ---- attention maps ----------------------------------------------------

    def _vrows(self, k):
        """Virtual rows of the k rows of the current call (see self.vrows)."""
        if len(self.vrows) >= k:
            return self.vrows
        return list(range(self.row0, self.row0 + k))

    @staticmethod
    def _qk(module, x, ctx):
        """(q, k, heads, scale) of a cross-attention: ldm / Forge UNet (to_q, to_k) or Anima's DiT (q_proj,
        k_proj with a per-head RMSNorm on q and k)."""
        if hasattr(module, "to_q"):
            q, k = module.to_q(x), module.to_k(ctx)
            heads = module.heads
            return q, k, heads, getattr(module, "scale", (q.shape[-1] // heads) ** -0.5)
        H, D = module.n_heads, module.head_dim
        q = module.q_norm(module.q_proj(x).unflatten(-1, (H, D))).flatten(-2)
        k = module.k_norm(module.k_proj(ctx).unflatten(-1, (H, D))).flatten(-2)
        return q, k, H, D ** -0.5

    def _img_of_row(self, r):
        if r < self.n_cond:
            return r * self.n_unc // self.n_cond if self.n_unc else r
        return r - self.n_cond

    def _hw_for(self, n):
        if not self.latent_hw:
            return None
        H0, W0 = self.latent_hw
        for f in (1, 2, 4, 8, 16, 32):
            H, W = -(-H0 // f), -(-W0 // f)
            if H * W == n:
                return H, W
        return None

    def _attn_hook(self, module, args, kwargs, out):
        if not self.active or not (self.focus_tokens or self.word_tokens or self.neg_word_tokens):
            return None
        x = args[0] if args else kwargs.get("x")
        ctx = kwargs.get("context", args[1] if len(args) > 1 else None)
        if not torch.is_tensor(x) or not torch.is_tensor(ctx) or x.dim() != 3:
            return None
        if ctx.dim() > 3:  # Anima: context [B, 1, T, D]
            ctx = ctx.flatten(1, -2)
        try:
            return self._attn_maps(module, x, ctx)
        except Exception as e:  # word maps are a by-product: never stop the generation for them
            if not getattr(self, "_attn_warned", False):
                self._attn_warned = True
                print(f"[ActSteer] word maps skipped ({type(e).__name__}: {e}); x {tuple(x.shape)}, context {tuple(ctx.shape)}")
            return None

    def _attn_maps(self, module, x, ctx):
        vr = self._vrows(x.shape[0])
        rows_c = [j for j in range(x.shape[0]) if vr[j] < self.n_cond]
        rows_u = [j for j in range(x.shape[0]) if vr[j] >= self.n_cond] if self.neg_word_tokens else []
        rows = rows_c + rows_u
        hw = self._hw_for(x.shape[1])
        if not rows or hw is None:
            return None
        with torch.no_grad():
            q, k, heads, scale = self._qk(module, _take(x, rows), _take(ctx, rows))
            # one fused reduction for every row and word (lib_act_steer/tokenmaps.py), then one add per
            # image into the stacked maps: no per-word launches
            groups = [(rows_c, [(self.focus_tokens, self.maps), (self.word_tokens, self.word_maps)])]
            if rows_u:
                groups.append((rows_u, [(self.neg_word_tokens, self.neg_word_maps)]))
            self._reduce_maps(q, k, rows, groups, heads, scale, hw)
            blk = self.attn_block.get(id(module))
            variants = self._lora_variants(module) if blk is not None and self.wordmap_on and self.word_tokens \
                and rows_c else None
            if variants is not None:  # a LoRA is loaded: every block gets its map, LoRAs acting on it more
                nc = len(rows_c)
                xs, cs = _take(x, rows_c), _take(ctx, rows_c)
                for name, wq, wk in [("", None, None)] + variants:
                    qv = q[:nc] if wq is None else F.linear(xs, wq.to(xs.dtype))
                    kv = k[:nc] if wk is None else F.linear(cs, wk.to(cs.dtype))
                    store = {img: self.block_maps.setdefault(img, {}).setdefault((name, blk), {})
                             for img in {self._img_of_row(vr[j]) for j in rows_c}}
                    self._reduce_maps(qv, kv, rows_c, [(rows_c, [(self.word_tokens, store)])], heads, scale, hw)
            self._map_cache = {}
        return None

    def _columns(self, tokens, img, T, device):
        """(keys, [has positions], [T, K] 0/1 selection) of a token group dict for one image, cached."""
        ck = (id(tokens), img, T, device)
        hit = self._sel_cache.get(ck)
        if hit is not None and hit[0] is tokens:
            return hit[1]
        keys, valid = list(tokens), []
        sel = torch.zeros(T, len(keys), device=device)
        for c, key in enumerate(keys):
            per_img = tokens[key]
            pos = per_img[img % len(per_img)] if per_img else None
            pos = pos[pos < T] if pos is not None and len(pos) else None
            valid.append(pos is not None and len(pos) > 0)
            if valid[-1]:
                sel[pos.to(device), c] = 1.0
        self._sel_cache[ck] = (tokens, (keys, valid, sel))
        return keys, valid, sel

    def _reduce_maps(self, q, k, rows, groups, heads, scale, hw):
        """q / k: to_q / to_k outputs of `rows` (call rows); groups: [(subset of rows, [(token group dict,
        store {img: {key: map}})])]. Adds every word's attention of these rows to its store."""
        T, dev = k.shape[1], q.device
        # the plan and selection matrix only depend on the rows and token groups: the same for every layer
        ck = (tuple(self._vrows(max(rows) + 1)), tuple(rows), T, dev,
              tuple((tuple(sub), tuple((id(t), id(st)) for t, st in dests)) for sub, dests in groups))
        hit = self._plan_cache.get(ck)
        if hit is None or any(a is not b for a, b in zip(hit[0], [x for _, d in groups for p in d for x in p])):
            hit = (([x for _, d in groups for p in d for x in p]), *self._plan(rows, groups, T, dev))
            if len(self._plan_cache) > 32:  # a few row layouts per job (cond / uncond rows, LoRA blocks)
                self._plan_cache = {}
            self._plan_cache[ck] = hit
        _, plan, sel, it = hit
        if not plan:
            return
        qs, ks = (q, k) if it is None else (q.index_select(0, it), k.index_select(0, it))
        out = tokenmaps.token_maps(qs, ks, sel, heads, scale)  # [rows, L, C]
        C = sel.shape[2]
        m = out.transpose(1, 2).reshape(len(plan), C, *hw)
        target = self._map_hw()
        if tuple(hw) != tuple(target):
            m = F.interpolate(m, size=target, mode="bilinear", align_corners=False)
        for r, (_, cols) in enumerate(plan):
            for store, img, keys, valid, sl in cols:
                self._stack_add(store.setdefault(img, {}), keys, valid, m[r, sl])

    def _plan(self, rows, groups, T, dev):
        """-> (plan [(index into rows, [(store, img, keys, valid, column slice)])], selection [n, T, C],
        row index tensor or None when all rows take part)."""
        index = {j: i for i, j in enumerate(rows)}
        plan, sels = [], []  # per row: [(store, keys, valid, column slice)]
        for sub, dests in groups:
            for j in sub:
                img = self._img_of_row(self._vrows(j + 1)[j])
                cols, parts, c0 = [], [], 0
                for tokens, store in dests:
                    if not tokens:
                        continue
                    keys, valid, sel = self._columns(tokens, img, T, dev)
                    if not any(valid):
                        continue
                    parts.append(sel)
                    cols.append((store, img, keys, valid, slice(c0, c0 + len(keys))))
                    c0 += len(keys)
                if cols:
                    plan.append((index[j], cols))
                    sels.append(torch.cat(parts, 1) if len(parts) > 1 else parts[0])
        if not plan:
            return [], None, None
        C = max(s.shape[1] for s in sels)
        sel = torch.zeros(len(plan), T, C, device=dev)
        for i, s in enumerate(sels):
            sel[i, :, :s.shape[1]] = s
        idx = [i for i, _ in plan]
        it = None if idx == list(range(len(rows))) else torch.tensor(idx, device=dev)
        return plan, sel, it

    def _stack_add(self, per, keys, valid, m):
        """per[key] are views into one [K, H, W] buffer: adding a layer's maps (and the per-step fade) is a
        single op. A new key set or a new resolution (hires pass) rebuilds the buffer from what per holds."""
        st = self._stacks.get(id(per))
        if st is None or st["per"] is not per or st["keys"] != keys:
            buf = torch.zeros_like(m)
            for c, key in enumerate(keys):
                old = per.get(key)
                if old is not None:
                    buf[c] = old if old.shape == m.shape[1:] else \
                        F.interpolate(old[None, None], size=m.shape[1:], mode="bilinear", align_corners=False)[0, 0]
            live = {id(d) for d in self._map_stores()}
            self._stacks = {i: v for i, v in self._stacks.items() if i in live}
            st = self._stacks[id(per)] = {"per": per, "keys": list(keys), "buf": buf, "shown": None}
        elif st["buf"].shape[1:] != m.shape[1:]:  # hires pass: new resolution
            st["buf"] = F.interpolate(st["buf"][None], size=m.shape[1:], mode="bilinear", align_corners=False)[0]
            st["shown"] = None
        st["buf"] += m
        shown = tuple(v or key in per for key, v in zip(keys, valid))
        if shown != st["shown"]:
            for c, key in enumerate(keys):
                if shown[c]:
                    per[key] = st["buf"][c]
            st["shown"] = shown

    def _map_stores(self):
        return [*self.maps.values(), *self.word_maps.values(), *self.neg_word_maps.values(),
                *(d for b in self.block_maps.values() for d in b.values())]

    def _lora_variants(self, attn):
        """[(LoRA name, to_q weight, to_k weight with that LoRA taken out)] for the loaded LoRAs that
        change attn's to_q or to_k (built-in Lora extension; block weights are already in its deltas).
        None when no LoRA is loaded."""
        try:
            import networks
        except ImportError:
            return None
        nets = list(getattr(networks, "loaded_networks", []))
        if not nets:
            return None
        self.lora_weights = {n.name: n.te_multiplier for n in nets}
        key = (id(attn), tuple((n.name, str(n.unet_multiplier)) for n in nets))
        if key in self._lora_w:
            return self._lora_w[key]
        per = []
        for lin in (attn.to_q, attn.to_k):
            name = getattr(lin, "network_layer_name", None)
            w0 = getattr(lin, "network_weights_backup", None)  # original weight when LoRAs are merged in
            w0 = (w0 if torch.is_tensor(w0) else lin.weight).to(lin.weight.device)
            ups = {}
            for net in nets:
                mod = net.modules.get(name) if name else None
                if mod is None:
                    continue
                try:
                    up, _ = mod.calc_updown(w0)
                    ups[net.name] = up.to(device=w0.device, dtype=torch.float32).reshape(w0.shape)
                except Exception as e:
                    print(f"[ActSteer] LoRA block analysis: {net.name} {name}: {e}")
            w_all = w0.float() + sum(ups.values()) if ups else w0.float()
            per.append((w_all, ups))
        out = []
        for n in sorted(set(per[0][1]) | set(per[1][1])):
            out.append((n, (per[0][0] - per[0][1].get(n, 0)).to(w0.dtype),
                        (per[1][0] - per[1][1].get(n, 0)).to(w0.dtype)))
        self._lora_w[key] = out
        return out

    def _attn_fix_hook(self, module, args, kwargs, out):
        fix = self.fix
        if fix is None or fix["kind"] not in ("record", "words"):
            return None
        if fix["kind"] == "words" and (self._fix_frac() < fix.get("start", 0.0) or not fix["positions"]):
            return None
        x = args[0] if args else kwargs.get("x")
        ctx = kwargs.get("context", args[1] if len(args) > 1 else None)
        if not torch.is_tensor(x) or not torch.is_tensor(ctx) or x.dim() != 3:
            return None
        vr = self._vrows(x.shape[0])
        rows = [j for j in range(x.shape[0]) if vr[j] < self.n_cond]
        hw = self._hw_for(x.shape[1])
        if not rows or hw is None:
            return None
        with torch.no_grad():
            m = self._fix_mask(hw[0], hw[1], x.device).flatten()  # [L]
            q = module.to_q(_take(x, rows))
            k = module.to_k(_take(ctx, rows))
            n, L, D = q.shape
            T, heads = k.shape[1], module.heads
            dh = D // heads
            q = q.view(n, L, heads, dh).transpose(1, 2)
            k = k.view(n, T, heads, dh).transpose(1, 2)
            scale = getattr(module, "scale", dh ** -0.5)
            probs = (q @ k.transpose(-1, -2)).float().mul_(scale).softmax(-1)  # [n, heads, L, T]
            if fix["kind"] == "record":
                p = probs.mean(1)  # [n, L, T]
                a_in = (p * m[None, :, None]).sum(1).sum(0) / (m.sum() * n).clamp_min(1e-6)
                a_out = (p * (1 - m)[None, :, None]).sum(1).sum(0) / ((1 - m).sum() * n).clamp_min(1e-6)
                for key, val in (("tok_in", a_in), ("tok_out", a_out)):
                    old = fix.get(key)
                    if old is None or old.shape != val.shape:
                        fix[key], fix[key + "_n"] = val, 1
                    else:
                        fix[key] = old + val
                        fix[key + "_n"] += 1
                return None
            # words: scale the chosen tokens' attention by (1 - w) and renormalise, per head
            pos = torch.tensor([t for t in fix["positions"] if t < T], dtype=torch.long, device=x.device)
            if not len(pos):
                return None
            v = module.to_v(_take(ctx, rows)).view(n, T, heads, dh).transpose(1, 2).float()  # [n, heads, T, dh]
            raw = probs @ v                                                            # [n, heads, L, dh]
            ps = probs[..., pos]                                                       # [n, heads, L, S]
            c = ps @ v[:, :, pos, :]
            w = float(fix["w"])
            denom = (1 - w * ps.sum(-1, keepdim=True)).clamp_min(0.05)
            change = (raw - w * c) / denom - raw                                       # [n, heads, L, dh]
            change = change.transpose(1, 2).reshape(n, L, D) * m[None, :, None]
            proj = module.to_out[0] if isinstance(module.to_out, torch.nn.Sequential) else module.to_out
            out = out.clone()
            _put(out, rows, _take(out, rows) + F.linear(change.to(out.dtype), proj.weight))
        return out

    # ---- layers: where in the image a word may act -------------------------------------------

    def _ref_map(self, img, keys, H, W):
        """[H, W] 0..1: where the words `keys` look in this generation so far (None before they do)."""
        per = self.word_maps.get(img, {})
        accs = [per[k] for k in keys if k in per]
        if not accs:
            return None
        acc = accs[0]
        for a in accs[1:]:
            acc = acc + (a if a.shape == acc.shape else
                         F.interpolate(a[None, None], size=acc.shape, mode="bilinear", align_corners=False)[0, 0])
        m = ((soft_map(acc) - MAP_LOW) / (MAP_HIGH - MAP_LOW)).clamp(0, 1)
        m = F.avg_pool2d(m[None, None], 5, stride=1, padding=2, count_include_pad=False)  # soft edge
        return F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)[0, 0]

    def _subject_keys(self, img, exclude):
        """The subject when no word is named: the word map word with the most attention (not placed)."""
        per = self.word_maps.get(img, {})
        items = [(k, a) for k, a in per.items() if k not in exclude]
        if not items:
            return []
        sums = torch.stack([a.sum() for _, a in items]).tolist()  # one GPU sync, not one per word
        return [max(zip(sums, (k for k, _ in items)))[1]]

    def _place_mask(self, pl, img, H, W, device):
        """[H*W] target region of a placement, 1 = the word may act there; None = not known yet."""
        key = (id(pl), img, H, W)
        if key in self._place_cache:
            return self._place_cache[key]
        t = pl["target"]
        ys = torch.linspace(0, 1, H, device=device)[:, None].expand(H, W)
        xs = torch.linspace(0, 1, W, device=device)[None, :].expand(H, W)
        ramp = lambda v: ((0.6 - v) / 0.2).clamp(0, 1)  # noqa: E731  1 up to 0.4, 0 from 0.6
        m = None
        if t == "left":
            m = ramp(xs)
        elif t == "right":
            m = ramp(1 - xs)
        elif t == "top":
            m = ramp(ys)
        elif t == "bottom":
            m = ramp(1 - ys)
        elif t == "center":
            m = ramp(((xs - 0.5) ** 2 + (ys - 0.5) ** 2).sqrt() + 0.25)
        elif t == "seg":  # like Regional Prompter: where the part's own words look (lib_act_steer/seglayers.py)
            refs = pl["ref"][img % len(pl["ref"])] if pl.get("ref") else None
            if not refs and pl["layer"] in ("body", "skin"):
                refs = self._subject_keys(img, pl.get("exclude", ()))
            r = self._ref_map(img, refs, H, W) if refs else None
            if r is not None:
                m = r.to(device)
                sub = pl["minus"][img % len(pl["minus"])] if pl.get("minus") else None
                s = self._ref_map(img, sub, H, W) if sub else None
                if s is not None:
                    m = m * (1 - s.to(device))
        else:
            refs = pl["ref"][img % len(pl["ref"])] if pl.get("ref") else None
            refs = refs or self._subject_keys(img, pl.get("exclude", ()))
            r = self._ref_map(img, refs, H, W) if refs else None
            if r is not None:
                r = r.to(device)
                m = r if t in ("foreground", "on") else 1 - r
        if m is not None:
            m = m.flatten()
        self._place_cache[key] = m
        return m

    def _on_device(self, pos, T, device):
        """Token positions (CPU LongTensor) below T on the GPU, cached: a host-to-device copy per layer
        and step would stall the GPU queue on ROCm (see _row_slices). None when none is below T."""
        key = (id(pos), T, device)
        hit = self._pos_cache.get(key)
        if hit is None or hit[0] is not pos:
            p = pos[pos < T]
            hit = (pos, p.to(device) if len(p) else None)
            self._pos_cache[key] = hit
        return hit[1]

    def _attn_edit_hook(self, module, args, kwargs, out):
        """Word edits: each edited word's attention scaled by its factor, renormalised per head.
        Layers: a placed word's attention outside its target region scaled by 1 - strength."""
        frac = self._fix_frac()
        edits = self.word_edits if frac >= self.edit_start else []
        places = self.placements if frac >= self.place_start else []
        if not edits and not places:
            return None
        x = args[0] if args else kwargs.get("x")
        ctx = kwargs.get("context", args[1] if len(args) > 1 else None)
        if not torch.is_tensor(x) or not torch.is_tensor(ctx) or x.dim() != 3:
            return None
        vr = self._vrows(x.shape[0])
        rows = [j for j in range(x.shape[0]) if vr[j] < self.n_cond]
        if not rows:
            return None
        with torch.no_grad():
            heads = module.heads
            T = ctx.shape[1]
            proj = module.to_out[0] if isinstance(module.to_out, torch.nn.Sequential) else module.to_out
            out = out.clone()
            hw = self._hw_for(x.shape[1])
            for j in rows:
                img = self._img_of_row(vr[j])
                scale = None
                for per_img, factor in edits:
                    pos = per_img[img % len(per_img)] if per_img else None
                    if pos is None or not len(pos):
                        continue
                    pos = self._on_device(pos, T, x.device)
                    if pos is None:
                        continue
                    if scale is None:
                        scale = torch.ones(T, device=x.device)
                    scale[pos] = float(factor)
                if scale is not None:
                    scale = scale[None, :]  # [1, T], the same at every position
                for pl in places if hw is not None else ():
                    pos = pl["per_img"][img % len(pl["per_img"])] if pl["per_img"] else None
                    if pos is None or not len(pos):
                        continue
                    pos = self._on_device(pos, T, x.device)
                    m = self._place_mask(pl, img, hw[0], hw[1], x.device) if pos is not None else None
                    if m is None:
                        continue
                    if scale is None:
                        scale = torch.ones(1, T, device=x.device)
                    if scale.shape[0] == 1:
                        scale = scale.expand(m.shape[0], T).clone()  # [L, T]
                    scale[:, pos] = scale[:, pos] * (1 - float(pl["strength"]) * (1 - m))[:, None]
                if scale is None:
                    continue
                q = module.to_q(x[j:j + 1])
                k = module.to_k(ctx[j:j + 1])
                v = module.to_v(ctx[j:j + 1])
                L, D = q.shape[1], q.shape[2]
                dh = D // heads
                q = q.view(1, L, heads, dh).transpose(1, 2)
                k = k.view(1, T, heads, dh).transpose(1, 2)
                v = v.view(1, T, heads, dh).transpose(1, 2).float()
                probs = (q @ k.transpose(-1, -2)).float().mul_(getattr(module, "scale", dh ** -0.5)).softmax(-1)
                # negative factors push the word away; the clamp keeps the renormalisation finite
                edited = probs * scale
                edited = edited / edited.sum(-1, keepdim=True).clamp_min(0.05)
                change = ((edited - probs) @ v).transpose(1, 2).reshape(1, L, D)  # [1, L, D]
                out[j:j + 1] = out[j:j + 1] + F.linear(change.to(out.dtype), proj.weight)
        return out

    def on_cfg_denoised(self, params):
        fix = self.fix
        if fix is None or fix["kind"] != "guidance" or self._fix_frac() < fix.get("start", 0.0):
            return
        x = params.x
        if not self.n_unc or x.shape[0] < self.n_cond + self.n_unc:
            return
        m = self._fix_mask(x.shape[2], x.shape[3], x.device).to(x.dtype)
        keep = 1 - float(fix["w"]) * m
        for r in range(self.n_cond):
            u = self.n_cond + self._img_of_row(r)
            x[r] = x[u] + (x[r] - x[u]) * keep

    def _map_hw(self):
        H0, W0 = self.latent_hw
        return -(-H0 // 2), -(-W0 // 2)

    def norm_map(self, img, focus):
        """[H, W] weights in 0..1: where the focus words look, background level removed."""
        acc = self.maps.get(img, {}).get(focus)
        if acc is None:
            return None
        return ((soft_map(acc) - MAP_LOW) / (MAP_HIGH - MAP_LOW)).clamp(0, 1)

    def _map(self, focus, r, H, W):
        img = self._img_of_row(r)
        key = (img, focus, H, W)
        if key not in self._map_cache:
            m = self.norm_map(img, focus)
            if m is not None and m.shape != (H, W):
                m = F.interpolate(m[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            self._map_cache[key] = m
        return self._map_cache[key]

    def _row_maps(self, focus, k, H, W):
        vr = self._vrows(k)
        return [self._map(focus, vr[j], H, W) for j in range(k)]

    def switch_unet(self, unet, forge=False):
        """Another model took over during the job (the refiner, from "Refiner switch at"): the hooks move to
        its UNet, so word maps, word edits, layers and profiles go on with it (profiles only in blocks with
        the same channel count). Profile recording stops: one profile must not mix two models."""
        self.install(unet, forge)
        self.refined = True
        self._map_cache, self._place_cache, self._plan_cache, self._lora_w = {}, {}, {}, {}
        print("[ActSteer] the refiner took over: word maps, edits and layers go on with it"
              + ("; profile recording stopped at the switch" if self.recorder is not None else ""))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.unet = None
        self.hooked = False
        self._lora_w = {}  # weight copies of the old model's attention layers

    def on_cfg_denoiser(self, params):
        if not self.active:
            return
        if getattr(self, "hooked", False):  # the refiner switch happens just before this callback, in the same step
            unet, forge = _current_unet()
            if unet is not None and unet is not self.unet:
                self.switch_unet(unet, forge)
        unc = params.text_uncond
        if getattr(self, "forge", False):  # Forge: x is the image batch; one prompt and one negative row each
            self.n_cond = params.x.shape[0]
            self.n_unc = params.x.shape[0] if unc is not None else 0
        else:
            self.n_unc = unc.shape[0] if torch.is_tensor(unc) else next(iter(unc.values())).shape[0]
            self.n_cond = params.x.shape[0] - self.n_unc
        self.call_rows = 0
        if params.sampling_step != self.step:
            # older steps fade out, later (cleaner) ones dominate
            for per in self._map_stores():
                st = self._stacks.get(id(per))
                stacked = st["keys"] if st is not None and st["per"] is per else ()
                if stacked:
                    st["buf"].mul_(MAP_DECAY)  # the views in per follow
                for key in per:
                    if key not in stacked:
                        per[key] = per[key] * MAP_DECAY
            self._map_cache = {}
            self._place_cache = {}  # the subject's area sharpens step by step
        self.step = params.sampling_step
        self.steps = params.total_sampling_steps
        if not self._early_done and self.word_maps and self.steps and self.step >= EARLY_FRAC * self.steps:
            # the layout has formed: what each word did so far (once per job, not again in a hires pass)
            self.early_word_maps = {img: {k: v.clone() for k, v in per.items()} for img, per in self.word_maps.items()}
            self._early_done = True
        steers = self.ui_steers + self.prompt_steers
        if steers:
            self.last_applied = {"steers": list(steers), "mode": self.mode, "cond_only": self.cond_only,
                                 "start": self.start, "end": self.end, "block_pred": self.block_pred,
                                 "blocks": self.blocks_spec,
                                 "focus": {s.name: self.steer_focus(s) for s, _ in steers}}

    def _unet_pre(self, module, args, kwargs):
        if not self.active:
            return
        x = args[0] if args else kwargs.get("x")
        k = x.shape[0] if torch.is_tensor(x) else 0
        if torch.is_tensor(x) and x.dim() in (4, 5):  # [B, C, H, W]; DiT video latents [B, C, T, H, W]
            self.latent_hw = (x.shape[-2], x.shape[-1])
        self.row0 = self.call_rows
        self.call_rows += k
        cu = (kwargs.get("transformer_options") or {}).get("cond_or_uncond")
        if cu and k % len(cu) == 0:  # Forge: one chunk of the batch per entry, 0 = prompt, 1 = negative
            m = k // len(cu)
            self.vrows = [(i if c == 0 else self.n_cond + i) for c in cu for i in range(m)]
        else:
            self.vrows = list(range(self.row0, self.row0 + k))

    def _make_block_hook(self, name):
        def hook(module, args, out):
            if not self.active or not torch.is_tensor(out) or out.dim() != 4:
                return None
            frac = self.step / self.steps if self.steps else 0.0
            k, C, H, W = out.shape
            if self.recorder is not None and not self.refined:
                b = bucket_of(self.step, self.steps, self.recorder.buckets)
                maps = self._row_maps(self.recorder.focus, k, H, W) if self.recorder.focus else None
                self.recorder.add(name, b, out, self.n_cond, self.n_unc, self._vrows(k), self.image_base, maps)
            steers = self.ui_steers + self.prompt_steers
            if not steers or not (self.start <= frac <= self.end) or not self.block_pred(name):
                return None
            row_mask = None
            if self.cond_only:
                rows = torch.tensor(self._vrows(k), device=out.device)
                row_mask = (rows < self.n_cond).to(out.dtype).view(k, 1, 1, 1)
                if not row_mask.any():
                    return None
            for steer, w in steers:
                b = bucket_of(self.step, self.steps, steer.buckets)
                focus = self.steer_focus(steer)
                region = None
                if focus:
                    maps = self._row_maps(focus, k, H, W)
                    if all(m is None for m in maps):
                        continue
                    zero = torch.zeros(H, W, device=out.device)
                    region = torch.stack([zero if m is None else m for m in maps]).to(out.dtype)[:, None]
                    if row_mask is not None:
                        region = region * row_mask
                if self.mode == "gain":
                    g = steer.get("gain", name, b, out.device, torch.float32)
                    if g is None or g.shape[0] != C:
                        continue
                    scale = g.clamp(0.1, 10.0).pow(w).to(out.dtype).view(1, C, 1, 1)
                    if region is not None:
                        scale = 1 + (scale - 1) * region
                    elif row_mask is not None:
                        scale = 1 + (scale - 1) * row_mask
                    out = out * scale
                else:
                    d = steer.get("delta", name, b, out.device, out.dtype)
                    if d is None or d.shape[0] != C:
                        continue
                    add = d.view(1, C, 1, 1) * w
                    if region is not None:
                        add = add * region
                    elif row_mask is not None:
                        add = add * row_mask
                    out = out + add
            return out
        return hook


def find_unet(model):
    """The SD 1.x / SDXL UNet of a webui model: A1111's ldm model or Forge's backend UNet (same block layout);
    None for other model types (Flux, Anima ... have no UNet cross-attention)."""
    for get in (lambda: model.model.diffusion_model, lambda: model.forge_objects.unet.model.diffusion_model):
        try:
            unet = get()
        except Exception:
            continue
        if hasattr(unet, "input_blocks") and hasattr(unet, "output_blocks"):
            return unet
    return None


def is_dit(net):
    """Anima (Cosmos-style DiT): blocks with a cross_attn of q_proj / k_proj."""
    blocks = getattr(net, "blocks", None)
    return blocks is not None and len(blocks) > 0 and hasattr(getattr(blocks[0], "cross_attn", None), "q_proj")


def find_net(model):
    """(network, forge) of a webui model: a SD 1.x / SDXL UNet, or an Anima DiT (word maps only); (None, False)
    for other model types."""
    unet = find_unet(model)
    forge = unet is not None and getattr(getattr(model, "model", None), "diffusion_model", None) is not unet
    if unet is not None:
        return unet, forge
    try:
        net = model.forge_objects.unet.model.diffusion_model
    except Exception:
        return None, False
    return (net, True) if is_dit(net) else (None, False)


def _current_unet():
    try:
        from modules import shared
    except ImportError:
        return None, False
    return find_net(shared.sd_model)


def lbw_block(name):
    """input_blocks.4 -> IN04, middle_block -> M00, output_blocks.2 -> OUT02 (LoRA Block Weight names)."""
    if name == "middle_block":
        return "M00"
    side, _, i = name.partition(".")
    return f"{'IN' if side == 'input_blocks' else 'OUT'}{int(i):02d}"


def _blocks(unet):
    out = [(f"input_blocks.{i}", m) for i, m in enumerate(unet.input_blocks)]
    out.append(("middle_block", unet.middle_block))
    out += [(f"output_blocks.{i}", m) for i, m in enumerate(unet.output_blocks)]
    return out


def final_bias_names(unet):
    """block -> name (relative to unet) of the bias added last to that block's output.

    ResBlock returns skip(x) + out_layers(h), SpatialTransformer proj_out(x) + x_in,
    Down/Upsample end in a conv, so a constant added to that bias is a constant
    added to the block output.
    """
    names = {id(m): n for n, m in unet.named_modules()}
    out = {}
    for block, module in _blocks(unet):
        last = list(module.children())[-1] if len(list(module.children())) else module
        cls = type(last).__name__
        layer = {"ResBlock": lambda m: m.out_layers[-1], "SpatialTransformer": lambda m: m.proj_out,
                 "Downsample": lambda m: m.op, "Upsample": lambda m: getattr(m, "conv", None)}.get(cls)
        layer = layer(last) if layer else (last if isinstance(last, (torch.nn.Conv2d, torch.nn.Linear)) else None)
        if layer is not None and getattr(layer, "bias", None) is not None and id(layer) in names:
            out[block] = names[id(layer)] + ".bias"
    return out


def bake_deltas(unet, applied):
    """Turn what a generation applied into constant bias offsets.

    Returns ({bias name relative to unet: delta [C] float32 cpu}, [notes]).
    Exact for add mode on both passes over all steps, apart from the time
    buckets being averaged into one value. Step ranges are folded in as
    strength x covered fraction, gain mode as (gain ** w - 1) x mean output.
    """
    notes = []
    if applied["cond_only"]:
        notes.append("'prompt rows only' can't be baked; the offset now applies to the negative pass too")
    start, end = applied["start"], applied["end"]
    coverage = max(0.0, end - start)
    if coverage < 1.0:
        notes.append(f"step range {start:g}-{end:g} baked as {coverage:g} x strength for all steps")
    if applied["mode"] == "gain":
        notes.append("gain mode baked approximately, as an offset of (gain ** strength - 1) x mean output")
    for steer, _ in applied["steers"]:
        focus = applied.get("focus", {}).get(steer.name)
        if focus:
            notes.append(f"'{steer.name}' is attention-guided ('{focus}'): it depends on where the words are "
                         "in each image and can't be baked into fixed weights, skipped")
    biases = final_bias_names(unet)
    out = {}
    for block, bias in biases.items():
        if not applied["block_pred"](block):
            continue
        total = None
        for steer, w in applied["steers"]:
            if applied.get("focus", {}).get(steer.name):
                continue
            table = steer.gain if applied["mode"] == "gain" else steer.delta
            per = table.get(block)
            if not per:
                continue
            nb = max(steer.buckets, 1)
            picked = [b for b in per if start <= (b + 0.5) / nb <= end] or \
                [min(per, key=lambda b: abs((b + 0.5) / nb - (start + end) / 2))]
            if applied["mode"] == "gain":
                means = steer.mean.get(block, {})
                parts = [(per[b].clamp(0.1, 10.0).pow(w) - 1) * means[b] for b in picked if b in means]
                if not parts:
                    continue
                d = torch.stack(parts).mean(0)
            else:
                d = torch.stack([per[b] for b in picked]).mean(0) * w
            d = d * coverage
            total = d if total is None else total + d
        if total is not None:
            out[bias] = total.float().cpu()
    if not out:
        notes.append("no block of the loaded model matched the profiles (different architecture?)")
    return out, notes


state = SteerState()

# Last recorded job per tab ("txt2img"/"img2img"), kept until the next one so
# its images can be rated after they were looked at.
last_recording = {}

# Word maps of the last finished job per tab (scripts/word_map.py):
# {"first": index of first image in the gallery, "items": [(PIL, [(word, map [H, W] or None, share)])]}
last_wordmaps = {}

# Images of the last finished job per tab, for the area-fix editor:
# {"first": index of first image in the gallery, "items": [(PIL, seed, subseed)], "prompt": str}
last_images = {}
