# Activation Steering

Record what a prompt switches on inside the UNet during a generation and
apply it again later, as a strength-controlled "extension" of the model.

A **profile** (`models/ActSteer/<name>.safetensors`, dir `SD_ACT_STEER_DIR`
from `start_amd.sh`) is an adjustment of the UNet's internal activations,
built from the influence of images you liked (positive) or didn't (negative).

* **Keep influence of the last image** (on by default): for every image of a
  job, each UNet block output is averaged per channel, for the prompt pass
  and the negative pass, in N time buckets, and kept until the next job.
* After a generation, select an image in the gallery (nothing selected = all
  images of the job), choose *Impact* (positive/negative) and *Weight*, then:
  * **➕ Merge into profile**: folds it into the selected profile (weighted
    running mean; weight 2 counts like two images).
  * **✨ Create new profile**: new profile from it (name field, empty = from
    prompt); refuses an existing name.
  * **⏭ Skip**: discards the kept influence.
  * **↩ Undo last merge**: restores the profile from before its last merge
    (`.undo` file, one level; undoing a create deletes the profile).
  * **🗑 Delete profile**.
  * *Merge automatically into profile* does it without clicking (API use).
  An image's push is prompt pass minus negative pass on the same latent; a
  profile is mean push of positives minus mean push of negatives.
* **Apply**: pick profiles under *Apply profiles* and set Strength, or put
  `<steer:name:strength>` in the prompt. Negative strength reverses it.
  * `add` adds strength x direction to the block outputs.
  * `gain` scales each channel by the RMS ratio ** strength.
  * *Prompt rows only* leaves the negative pass alone.
  * Start/End limit it to a part of the steps; Blocks e.g. `IN4-8,M,OUT0-5`.
* API:
  * `GET /actsteer/list`: profiles, metadata, positive/negative counts.
  * `POST /actsteer/merge` `{"profile", "impact": "positive"|"negative", "image": -1, "weight": 1.0, "create": false, "tab": "txt2img"}`
  * `POST /actsteer/skip` `{"tab"}`, `/actsteer/undo` `{"profile"}`, `/actsteer/delete` `{"profile"}`
  * `alwayson_scripts["activation steering"]["args"]`:
    `[keep, profile, buckets, auto ("ask"|"positive"|"negative"), [apply profiles], strength, mode, cond_only, start, end, blocks, record focus, focus override]`.

**Attention-guided (focus) profiles**: put words in *Focus words* before
generating, e.g. `eyes` (they must be in the prompt). The recorded influence is
then weighted by where those words look in the image (cross-attention of every
SpatialTransformer, at half latent resolution, accumulated over the steps), so
the profile describes only that region; the profile remembers its focus and
only takes images recorded with the same focus. When applied, the change is
weighted by the same words' map in the new image, so it follows the eyes
wherever they are. The words must be in the new prompt too, otherwise the
profile does nothing (console says so).
* Use **Start 0.4-0.5** for focus profiles: applied from the first step the
  change spreads over the whole face; from the middle on it stays in the map.
* The map is ~1/16 of the image size and blurred, so it hits "eye area"
  (lids, liner), not just the iris.
* *Focus override* (UI) or `<steer:name:strength:focus words>` applies any
  profile, also a whole-image one, at those words.
* Debug: `SD_ACT_STEER_DUMP_MAPS=<dir>` saves the maps of each job as PNG.
* Focus profiles can't be baked into a checkpoint (skipped with a note).

**Word map and word edits** (top of the *Activation Steering* section; the
separate Word Map accordion is gone). After every generation the word map of
the first image appears by itself (clicking another gallery image or
*Word map of the selected image* switches): the prompt words in colours and
every area tinted like the word most concentrated there (attention divided by
its own image average). Hover a word = spotlight preview; click a word = only
its colour, and it becomes the selected word.
* **Influence** slider (release to apply): 1 normal, 2 double, 0 the word is
  ignored, below 0 pushed away (limited to -1: hat:-2 broke the image). The
  edit lands in *Word edits* (`hat:0, dress:2`, editable by hand) and as a
  badge on the chip.
* **Regenerate with edits** clicks Generate; while edits exist and the seed is
  -1, the seed of the shown image is reused. Each edited word's cross-attention
  is scaled by its factor (renormalised per head) in every attention layer,
  prompt pass only. *Advanced > Word edits start* > 0 keeps the layout more.
* Tested: flowers:0 removes the flowers, clouds:0 + sky:2 gives a clear sky,
  hat:0 keeps the hat (also carried by "straw"), hat:-1 removes it but turns
  the girl around. Note: prompt-changing extensions (dynamic prompts / magic
  prompt) give a different prompt each run, so the same seed is not the same
  image then.
* **Word relations / prompt check**: clicking a word also shows which words act
  on the same area (number on the other chips, list in the panel); below the
  chips a *Prompt check* (one line per kind: words with little visible effect,
  nouns acting on the same area like ocean & sea) and a collapsible relation
  table. Overlap = soft Jaccard of the areas where each word is concentrated.
  No adjective-noun binding check: colour words concentrate on the same places
  as each other even when every colour lands on the right object.
* **✨ Optimize & regenerate** turns the prompt check into changes and
  regenerates with the same seed, most impactful changes first (impact =
  attention share x log(1 + peak concentration)): of two nouns on the same area
  the weaker one is multiplied by 0.6 (biggest overlap first, down to 0.2); weak
  words are boosted x1.5 (up to 3), words with < 0.5 % attention switched off
  (0). At most 6 changes. The list of changes is shown below the buttons.
  * **Locked words** never change. When a locked word shares its area with
    another word, the other one is weakened, even if it has more impact. 🔒
    next to the influence slider locks/unlocks the selected word; a word you set
    with the slider is locked automatically.
  * **Auto-lock** (slider, default 3, 0 = off): the N words of the prompt that
    shape the image most (attention share x concentration, fillers and
    attention sinks left out) count as locked too, listed under the changes
    ("🔒 Auto-locked"); the Locked field itself is not changed. Auto optimize
    picks them once from the first image of the run.
  * Changes build on the factors the image was made with (word edits and
    `(word:x)` prompt weights), so pressing it again after each regenerate keeps
    separating words that still overlap.
  * **Optimize writes into the prompt** (on by default): the changes go into the
    prompt as `(word:0.6)` emphasis, a switched-off word is removed, and those
    words leave the word edits box (no double effect). `<lora:...>` tags are not
    touched. Off: the changes stay word edits (UNet only, prompt unchanged).
    Prompt emphasis scales the text embedding, word edits scale the word inside
    the UNet, so the same number does not give exactly the same image.
  * Only words written in the prompt box change (not words that prompt
    extensions such as dynamic prompts or styles append, they differ every run;
    weight numbers like the 0.6 in `(sea:0.6)` and `<...>` tags are never
    touched). When such a word shares an area with a prompt word, the prompt
    word is weakened.
  API: `/actsteer/wordmap` returns `optimize.plan` and `optimize.edits`.
* **🔁 Auto optimize** repeats Optimize until the worst word conflict (the
  largest shared-area overlap of two nouns, pairs that cannot change left out)
  is below the threshold slider (default 0.30), or after *max rounds* (default
  5). Each round only separates words (overlap >= the threshold, no boosts),
  regenerates with the same seed, and measures the new image when the job
  finishes (the page clicks a hidden step button after each job). It stops
  early when nothing can change any more, when a generation is interrupted, or
  when you press the button again (⏹ Stop auto). A table of the rounds is shown;
  the prompt, negative prompt, word edits and layers stay as the last round left
  them (nothing is changed back). If an earlier round measured better, it is
  named and its prompt is shown to copy. Prompt extensions that append random words
  make rounds less comparable. With *Fix prompt adherence* (and evaluation on)
  a round below the threshold still goes on while the image misses prompt
  tags and there is something left to try for them. The best round: one you
  rated 👍 (rounds rated 👎 never), else quality + 0.3 x adherence.
* **Prompt adherence** (checkbox *Fix prompt adherence*, on;
  `lib_act_steer/adherence.py`): ✨ Optimize uses the last 📊 Evaluate of the
  shown image, or makes a quick one (WD14 tagger only, no quality models); auto
  uses its round evaluation. For each prompt phrase the tagger knows but did
  not find (missed tags):
  * *bind*: the attribute landed on another object (`red hat` missed, the image
    shows the hat and `red dress`): Layers gets `red -> on hat` (the colour's
    attention only where the hat is; with *As Regional Prompter regions* this
    becomes a region). A pair the prompt asks for itself (`red eyes`) is not a
    leak.
  * *boost*: the phrase's noun x1.2 per round, up to 1.4 (limit rules apply,
    locked / kept words not).
  * *move up*: already at 1.4 (or boosting did not help before): the phrase
    moves up, right after the subject.
  Missed words are never weakened by the conflict check in the same round.
* **Timing** (checkbox *Timing*, on; `lib_act_steer/timing.py`): core keeps a
  copy of the word maps at 30 % of the steps (the layout has formed; once per
  job, not again in the hires pass). For a conflict pair whose weaker word
  already pulls at the layout early (early share >= 0.5 x its final share), the
  weaker word is delayed instead of weakened: its phrase leaves its place and
  is added at the end as `[phrase:0.3]` (webui prompt editing: it joins at 30 %
  of the steps; at the end its missing tokens in the first steps shift no
  other word's position). Words that act mostly late anyway are weakened as
  before. A delayed word the image misses is put back in place. Needs *Optimize
  writes into the prompt*. The SD MCP reads `[word:0.3]` as `word` for
  adherence (`_unschedule` in sd_mcp.py).
* **Order** (checkbox *Order the prompt*, on; `lib_act_steer/order.py`; needs
  *Optimize writes into the prompt*): the leading quality / meta tags
  (masterpiece, best quality, score_9, absurdres ...) and LoRA tags keep their
  place; the subject (person word) comes right after them; phrases whose words
  all get almost no attention (< 1.2 %) and that no other change touches move
  to the end. A phrase cut by a 75-token chunk boundary (simulated like the
  webui's tokenizer, with its *Prompt word wrap length limit*) moves to the
  first later place where nothing is cut; when there is none, it gets a BREAK
  before it (not when Layers are set: BREAK would switch the Regional Prompter
  regions off). Prompts with BREAK / AND keep their order.
* **👍 / 👎** (next to 📊 Evaluate): your rating of the shown image. It goes
  into the learning experiments the image belongs to (the one that made it, and
  the one started from it, also when you rate before pressing ✨ / 🔁) and to
  the SD MCP (`rate_image`, 8 / 3 of 10, for its tag statistics). Your ratings
  outweigh the measurements (see Learning).
* **Remove or rephrase** (slider *Remove or rephrase a word when its overlap is
  at least*, default 0.6, 1 = never; used by Optimize and Auto): the weaker
  word of a pair above it is not weakened but taken out of the prompt, with its
  phrase when it is the phrase's noun ("blue sky"). With *Rephrase with the
  rhyme MCP* (on) the rhyme MCP (`~/mcp/rhyme_mcp`, tool `prompt_rephrase_tool`,
  :5007/rhyme, WordNet, no LLM) is asked first; its first suggestion not yet
  tried for that word is used: merge (near-synonyms like ocean/sea: drop one),
  place (scenery -> "... in background" / "distant ...", clothing -> "wearing
  ...", objects -> "holding ..."), synonym, a more specific kind, remove.
  Pressing Optimize again (or the next auto round) tries the next suggestion;
  ✖ Clear edits resets that. Only when Optimize writes into the prompt; with
  word edits the word gets factor 0. `SD_ACT_STEER_REPHRASE_URL` overrides.
  Auto mode allows up to 500 rounds.
* **Only words the model understands** (dropdown *Model vocabulary*, auto):
  rephrasing suggests only words of the checkpoint's vocabulary. `danbooru`
  for tag-trained SDXL (Illustrious, NoobAI, Pony, Animagine): every phrase a
  suggestion creates must be a Danbooru tag (14.6k tags, ranked by use; "sea"
  is none, so it is merged into "ocean"; "cup" -> "holding cup"; more specific
  kinds from tags like "sun hat"). `natural` for natural-language models: CLIP
  tokenizer words, ranked by English frequency. `auto`: from the checkpoint
  name and its base-model metadata, else from the prompt (mostly tags ->
  danbooru). Suggestions never add content words the prompt does not have
  (nude, blood, ...).
* **Filter the word map by influence** (row under the map): *Show words by
  influence* = all words / *top N* (the N words with the largest share of the
  attention to prompt words, strongest first) / *below threshold* (words under
  *Threshold*, % of the attention, default 0.5 %); filtered words show their
  share, filler words (a, the ...) are hidden. *🗑 Remove words below the
  threshold from the prompt* writes the prompt box (no generation, Ctrl+Z undoes
  it): a comma phrase goes only when all its words are below the threshold
  ("best quality" never becomes "quality"); locked words, words kept by rules
  and `<lora:...>` tags stay. Uses the word map of the shown image.
* **Image quality per word** (`lib_act_steer/quality.py`; checkbox *Consider
  image quality*, on): the view gets a collapsed *Image quality by word* table:
  sharpness (mean |Laplacian|), detail (gradient magnitude) and noise (median
  residual of a 3x3 median filter) of the image in each word's attention area,
  relative to a typical part of the image (1 = typical; flagged: blurry / flat
  < 0.7x, noisy > 1.4x). With the checkbox, Optimize and Auto also:
  * strengthen a word whose area is blurry / flat (x1.15, up to 1.6: more
    attention, more definition), weaken one whose area is noisy (x0.85, down to
    0.5);
  * weaken a prompt tag that across the SD MCP's recorded images made them
    blurrier / flatter (or noisier without also making them sharper - the noise
    estimate rises with fine detail). Source: the SD MCP tool `tag_effects`
    (all tags fitted together, each image compared only with images of the same
    checkpoint, size and hires factor; effect >= 0.15 standard deviations,
    bootstrap confidence >= 0.95, only when the wording explains >= 5 % on
    held-out prompts). Older SD MCPs: `token_quality_report` (one tag at a time,
    |r| >= 0.2, p <= 0.01, staple tags skipped);
  * when the image itself came out soft (SD MCP sharpness percentile < 40 among
    the checkpoint's images, from the last evaluation): also weaken tags that
    are smooth by nature but softened earlier images (soft lighting, shallow
    depth of field ...; joint analysis only) and strengthen tags that sharpened
    them (best quality, masterpiece, vibrant colors ...: x1.1, up to 1.3).
  The view gets a collapsed *Prompt tags vs sharpness* table (the prompt's tags
  with their measured effect, red = softer / noisier / flatter, green = sharper,
  plus sharpening tags not in the prompt as a hint; fetched in the background,
  never blocks the UI). Measured on 530 images (2026-10-24): soft lighting,
  shallow depth of field, gentle / golden sunlight scenes, fluffy tail and
  detailed face soften; best quality, masterpiece, vibrant colors, score_9
  sharpen; wording explains ~30 % of the sharpness differences.
  Smooth-by-nature words (sky, skin, bokeh, soft lighting ...) are never
  flagged as blurry in the per-word table, grainy ones (film grain, fur,
  glitter ...) never as noisy.
  At most 4 such changes, after the overlap fixes.
* **Every generation goes into the history** (checkbox *Add every generation to
  the SD MCP history*, on; script arg 18 `record_hist`): after each normal job
  the saved images (`on_image_saved`, no grids; with "Save images" off, copies
  with their generation info in `outputs/act-steer-eval/`) are sent to the SD
  MCP tool `record_images` in a background thread (metrics + WD14 tags, a few
  seconds, generation is not delayed). API jobs are skipped (the SD MCP records
  its own). 📊 Evaluate and auto evaluation add images too.
* **Layers: move prompt words to a part of the image** (field *Layers*, script
  args 20-21; global ones: `place` rules below; right click a word: *📍 Move to*): `sea -> background, lantern ->
  left, hat -> on girl, cloud -> away from girl, cat -> right:0.6`. Targets:
  background, foreground, left, right, top, bottom, center, `on <word>`,
  `away from <word>`; `:0.6` = strength (default 0.85). In every
  cross-attention layer the word's attention outside its target region is
  scaled by 1 - strength (per head renormalised, like word edits), from
  *Layers start* (Advanced, default 0.15 of the steps). background /
  foreground use the subject's area measured live in the same generation
  (person words like girl / 1girl, else the word with the most attention),
  `on` / `away from` the named word's area; left / right / ... are soft halves.
  The next generation keeps the seed, so only the moved words change place.
  Saved as "ActSteer layers" in the generation info. (Regional Prompter, also
  installed, splits the prompt into regions by fixed ratios, painted masks or a
  trigger word; this uses the measured layout of the image instead.)
  * **As Regional Prompter regions** (checkbox next to the Layers field, script
    arg 23, on by default; needs sd-webui-regional-prompter): a word moved to
    `background`, `foreground` or `on <word>` leaves the main prompt and gets
    its own BREAK region: `sea -> background` turns `…, blue sea, …` into
    `… , background BREAK blue sea, background` (a short phrase ending in the
    word moves whole, with its weight; inside a longer phrase only the word).
    RP is switched on in Prompt mode (Attention, threshold 0.4 unless set):
    the region acts where its last word (the target: `background`, the person
    word for foreground, the named word for `on`) gets attention in the main
    prompt; `background` is added to the main prompt when missing. Done in
    before_process, so RP's own process() sees the new prompt. The seed
    keeping / new-prompt checks use the prompt as you wrote it. Saved as
    "ActSteer RP regions". left / right / top / bottom / center and `away from`
    stay on act-steer's attention masking (RP runs
    one mode per generation); prompts that already contain BREAK / ADDCOL /
    ADDROW / ADDCOMM / ADDBASE are left alone. Layer strength isn't used by
    the RP regions.
    Several words on one target share one region (`sea -> background, sky ->
    background` gives `… BREAK blue sea, sky, background`; RP keys a region's
    mask by its target word, so two regions on one target lost the second),
    and a word whose target left the main prompt for a region of its own
    (`cat -> background, collar -> on cat`) stays on act-steer's masking (RP
    would shift the masks of all later regions by one).
  * **Parts of the figure** (`lib_act_steer/seglayers.py`): targets `skin`,
    `clothes` (also cloth / clothing / outfit), `hair`, `face`, `body` (person),
    e.g. `lace -> clothes, glossy -> skin, golden -> hair:0.7`. Made like
    Regional Prompter's Prompt mode, no segmentation model: a part's area is
    where its own prompt words get attention in this generation (dress / shirt
    ... for clothes, hair / ponytail ... for hair, eyes / mouth ... for face,
    the person word for body). A part without a word in the prompt gets its
    name appended (", clothes"), as RP appends "background". With *As Regional
    Prompter regions* on the word moves into a BREAK region whose target is
    that part word ("... BREAK lace, dress"); otherwise act-steer masks the
    word's attention by the part word's attention (skin: minus clothes and
    hair). Also in `place` rules and the *📍 Move to* menu.
* **Forge (neo) and Anima**: on Forge the CLIP tokenizer is Forge's text engine
  (`text_processing_engine_l`), and Forge's row order is read from
  `transformer_options["cond_or_uncond"]` (it batches the negative prompt
  first, or in separate calls), so word maps belong to the prompt, not the
  negative. Anima (Cosmos DiT, Qwen3 + T5 adapter) gets the word map: from the
  `cross_attn` of every other block (q / k with its per-head RMSNorm), word
  positions = T5 token positions (one context row per T5 token). Optimize,
  which writes into the prompt, works from that map. Word edits, layers,
  profiles and recording need a SD 1.x / SDXL UNet and are off on Anima
  (one console note).
* **Refiner** (txt2img *Refiner*: checkpoint + switch at): when the refiner
  takes over during a job, act-steer moves its hooks to the refiner's UNet in
  the same step (core.switch_unet; works whether the webui loads the refiner
  from disk or reuses a cached one). Word maps, word edits, layers and the area
  fix go on with it; profiles apply in blocks with the same channel count (an
  SDXL fine-tune as refiner: all; the official SDXL refiner: none); profile
  recording stops at the switch, so a profile never mixes two models. Models
  without a SD 1.x / SDXL UNet (Flux, Anima ...) switch act-steer off; as a
  refiner they end act-steer's work at the switch. The UNet is held only as a
  weak reference, so act-steer never keeps a model in RAM that the webui has
  dropped (Forge frees the base when the refiner loads).
* **Rules** (sub-section *Rules*, `lib_act_steer/rules.py`, saved in
  `<profile folder>/rules.txt`, apply to every prompt; Save or leaving the box
  saves): one rule per line, `when <word>` at the end limits it to prompts with
  that word, `*` / `?` wildcards (a pattern also matches one word of a phrase),
  `#` comments.
  * `allow hair ~ girl`: the two may share an area, no conflict (prompt check,
    Optimize, auto's worst conflict, LoRA blocks all skip the pair)
  * `prefer ocean over sea`: on overlap sea is the one that changes
  * `keep straw hat`: never changed (also not by replace rules)
  * `never remove sky`: weakened instead of removed / rephrased
  * `limit dress 0.7..1.3`: factor range for Optimize
  * `replace sea -> ocean`: Optimize writes ocean (duplicate phrases merged)
  * `negative keep lowres`, `negative add blurry:1.2`
  * `place sea|ocean -> background` (also `place *hat -> on girl`,
    `place lantern -> left:0.6 when night`): a fixed layer for every generation
    whose prompt has the word or phrase (wildcards expanded against the prompt's
    phrases). Same targets as the Layers field, which wins for a word it names.
    Not seed-keeping (a normal generation). On by default; the checkbox
    *Placement rules* next to the Layers field (script arg 22) turns all place
    rules off. Right click a word: *📌 Always move to* adds one.
  Right click a word: *allow overlap with* / *prefer over* chips for the words
  sharing its area add these rules directly. Patterns take `a|b|c`
  alternatives.
  * **Default rules** (second box, `<profile folder>/default_rules.txt`):
    loaded on every start and always applied together with the rules. The first
    start writes built-in defaults (parts and clothes of a person may share the
    person's area: `allow girl|boy|woman|... ~ hair|eyes|dress|hat|...`;
    anything may overlap background / scenery / outdoors). ⭐ saves them,
    ↺ restores the built-in ones.
  * Saving (button or leaving a box) only parses and writes the file; the word
    map is rendered again in a follow-up event without a progress overlay.
* **Learning: every optimisation improves the next images** (`lib_act_steer/learn.py`,
  `<profile folder>/learned.json`; checkbox *Learn from each optimisation*, on):
  each Optimize / auto round is an experiment. Before: the shown image's worst
  word conflict, number of blurry / flat / noisy word areas and SD MCP quality
  score (from 📊 Evaluate / auto evaluation, else evaluated in a background
  thread). The changes (weaken / strengthen / remove a word, rephrasing,
  replace rule, negative prompt changes). After: the next generation with that
  prompt (measured when the job ends, quality evaluated in the background).
  Per change: score = conflict drop + 4 x quality gain + 0.05 x fewer flagged
  areas + adherence gain + 0.3 x your rating change (👍 +1 / 👎 -1, after minus
  before), averaged; after 2 experiments a change is *helpful* (score >= 0.02,
  quality not down) or *harmful* (score <= -0.02 or quality down). A change
  whose images you rated worse is harmful at once, one you rated better
  helpful (unless measured harmful). Change kinds also include boost_missed,
  bind, move_up / move_down / subject_first, chunk, delay / undelay; helpful
  delays and boosts are applied to new prompts too. Reuse:
  * Optimize / auto skip harmful changes (a harmful weakening becomes removal /
    rephrasing when *Remove or rephrase* is on), and try rephrasings with a good
    record first, harmful ones not at all; harmful negative changes are skipped.
  * *Apply learned improvements to new prompts* (on; script arg 19): the first
    generation of a new prompt (and the first after start) gets the helpful
    changes whose words are in it (up to 6), recorded as "ActSteer learned" in
    the generation info and shown after the job. API jobs are left alone.
  * *Learned* sub-section: experiments, helpful / harmful changes, "allow"
    rule suggestions for overlaps nothing ever fixed (➕ adds them), 🗑 forget.
  Experiments are counted per checkpoint (all checkpoints when it has none).
  Background evaluations run one at a time; the SD MCP's quality models (a ~2 GB
  worker) are only asked for when at least 6 GB RAM is free
  (`SD_ACT_STEER_IQA_MIN_RAM_GB`), otherwise tags / adherence only and that
  score is not used for learning or auto's quality columns.
* **New prompt = clean start** (checkbox *Clear word edits and locked words for
  a new prompt*, on): when a generation's prompt shares less than half of its
  words with the previous job's, it is treated as a new image: the old word
  edits are not applied, the old seed is not reused, and after the job the word
  edits, locked words, tried rephrasings, the last evaluation and a running
  auto optimize are cleared. Optimize / rephrase / hand edits change only a few
  words and keep everything. **🧹 New prompt** clears all of it by hand.
  Script arg 17 (`auto_clear`).
* **Negative prompt** (checkbox *Optimize the negative prompt too*, on; used by
  Optimize and Auto): word maps are also recorded for the negative prompt (its
  words' attention in the uncond rows of the same cross-attention layers) and
  its comma phrases are checked (`lib_act_steer/negative.py`); the view shows a
  collapsed *Negative prompt check*:
  * conflict: the phrase names something the prompt asks for (negative "hat",
    prompt "straw hat") -> removed; when the phrase says more ("black and
    white" against "white dress") it is weakened x0.6 (down to 0.3), only when
    its area overlaps the prompt word's (>= 0.3). Generic words both prompts use
    (quality, detailed, hands, ...) do not count.
  * duplicate phrases -> the later one removed.
  * idle: the model hardly looks at it (< 0.2 % of the negative attention) ->
    removed.
  * defects the last 📊 Evaluate / auto evaluation found in this image
    ("blurry", "bad hands") -> added as `(blurry:1.15)` or strengthened x1.15
    (up to 1.5).
  At most 8 changes per press / round, written into the negative prompt box.
  Textual inversion embeddings, `<...>` tags and locked words are never touched.
  Auto mode records the negative prompt per round and restores the best round's.
* **Evaluation** (SD MCP, `~/mcp/sd_mcp` tool `evaluate_image`, service on
  :2018/sd, runs on the NVIDIA card): **📊 Evaluate** shows quality score (0-1:
  Aesthetic V2.5, LAION aesthetic, MUSIQ, NIMA, prompt adherence minus defect
  tags), prompt adherence (WD14 tagger), prompt tags not found in the image,
  defect tags and flags for the shown image. With *Auto: evaluate each round*
  every auto round is scored too: the table gets quality / adherence / "not in
  image" columns, auto stops when quality falls more than 0.08 below round 0,
  and the kept round is the best quality among rounds below the conflict
  threshold (else the lowest conflict among rounds without a quality drop).
  The image is written with its generation info to `outputs/act-steer-eval/`
  (the MCP reads the prompt from it and records it in its database). The first
  call loads the models (~1 min). Not reachable: auto goes on with the
  conflicts only. `SD_ACT_STEER_EVAL_URL` / `_DIR` / `_TIMEOUT` override.
* **LoRA blocks behind the overlap** (only when a `<lora:...>` is in the prompt
  and word maps are on): below the prompt check, per same-area pair, the
  overlap in every UNet block (LoRA Block Weight names: IN04, M00, OUT02 ...)
  and, per LoRA, how much of it that block loses when the LoRA's change to
  to_q/to_k is taken out (red = the LoRA pulls the words onto the same area,
  green = it separates them). Blocks adding >= 0.05 are named, with a suggested
  `<lora:name:w:lbw=...>` (block weight 1 - the share of the block's overlap the
  LoRA adds; BASE stays 1; SDXL 12 / SD1.5 17 values, relative to the block
  weights the image used). Measured in the first cross-attention of each
  block, from the deltas the built-in Lora extension applies (so LBW block
  weights are included); LoRA changes to other layers (to_v, feed-forward,
  convs) are not separated per block. Cost: one extra q/k product per LoRA and
  block per step. API: `/actsteer/wordmap` returns `lora_blocks` with `lbw`.
* **LoRA tuning** (✨ Optimize / 🔁 Auto optimize, with *Optimize writes into
  the prompt* on; `lib_act_steer/loratune.py`): the `<lora:...>` tags of the
  prompt are rewritten as `<lora:name:te:unet:lbw=...>`.
  - Block weights: every block where the LoRA adds word overlap is multiplied by
    1 - its share of that block's overlap (at most halved per round). The tag's
    lbw list keeps its length (12 / 20 SDXL, 17 / 26 SD1.5; presets like `ALL`
    resolved from lbwpresets.txt, a list of the other model type replaced);
    BASE and blocks without cross-attention (IN00-03 ... of the 20 list) stay.
  - Strength (unet; te is kept): when the LoRA adds overlap in >= 60 % of the
    blocks it acts on, the median block factor goes into the strength instead;
    when it takes the prompt words' areas away (their attention >= 25 % less
    concentrated with the LoRA, for half the words, or for a word the
    evaluation found missing in the image) strength x0.85. Floor 0.2.
  - Word pairs the LoRA change takes care of are not weakened in the same
    round; `limit name lo..hi` / `keep name` rules and Locked words work on
    LoRA names; learning records `lora_weaken:name` / `lora_blocks:name` and
    skips the ones that did not help. Auto's best-round restore restores the
    tags too. API: `/actsteer/wordmap` returns `lora_tune` (old / new tag, why).
* Everything else (profiles, focus, time buckets, blocks ...) sits in the
  collapsed *Profiles* and *Advanced* sub-sections.
* **Click on the image**: the strongest word at that spot (the colour shown
  there) is selected like a click on its label (only its colour, influence
  slider, relations); the words connected to the spot (>= 1.2x their image
  average and >= 75 % of the top word, up to 6) are outlined, the others
  dimmed, and listed under the image with their factor. The spot stays marked
  when the view is rendered again. Uses a 32-cell lift grid per
  word embedded in the view (`data-grid`), no server round trip.
* **Right click** a word, or an area of the image (menu for the top 4 words
  there): influence ×0 / ×0.5 / ×1.5 / ×2 / push away ×-0.5 / reset (a value
  set this way locks the word, like the slider), lock / unlock, show only this
  word, **✂ remove from prompt** (also `(word:x)`, not inside `<lora:...>`
  tags; the word is struck through until the next job) and **↩ undo** of the
  last removal.
* **Refresh**: after every job the server raises a counter
  (`GET /actsteer/version`); the page polls it every 1.5 s and reloads the word
  map, also when another gallery image is clicked. (Watching the gallery `img`
  src did not refresh reliably.)
* Clicks reach Python through `javascript/act_steer.js` (hidden textbox
  `#act_steer_<tab>_word`, `action|word|value|time`); the view is HTML with !important layout rules
  because the webui's `.prose` styles would override plain classes.
* API: `GET /actsteer/wordmap?tab=txt2img&image=0` (per word: colour, share,
  PNG overlay), `&html_view=true` for the page. Script args 14-16:
  word edits text, record word maps, word edits start.
* Examples: `examples/word_map_colours.png`, `examples/word_map_click.png`,
  `examples/word_edit_walkthrough.png` (browser run: map, click, slider, regenerate).

**Fix Wrong Areas** (Script dropdown): generate, press *Load last image*
(select the image in the gallery first if there are several), paint over what
came out wrong, press Generate. Keep prompt and settings unchanged; the
loaded image's seed is used.
1. The image is rendered again (identical) while every cross-attention layer
   measures which prompt words the painted area attends to much more than the
   rest of the image (printed, and in the infotext, e.g. `remove words [boat]`).
2. It is rendered once per strength with those words' attention scaled by
   (1 - strength) inside the area only (renormalised per head). 1 = the area
   no longer sees the words, 2-3 = pushed away. *Words to remove* overrides
   the automatic choice.
* *weaken prompt in area*: CFG guidance scaled by (1 - strength) inside the
  area instead; milder, changes style more than content.
* Start 0.15 (default) keeps the rest of the image; Start 0 can remove big
  objects (a hat) but re-composes the whole picture.
* Small objects get removed and filled with surroundings; large ones fixed in
  the first steps get redesigned instead. If the prompt still asks for the
  object it can reappear elsewhere; take the word out of the prompt for that.
* API: `"script_name": "fix wrong areas"`, `"script_args": [mask (base64 PNG,
  white = wrong), mode, words, "1, 2, 3", start, feather, use_last_seed,
  include_original]` with the seed in the payload and use_last_seed false.
* Example: `examples/fix_areas_example.png`.

**Saving a profile into a checkpoint**: the builtin *Bake LoRA into
checkpoint* plugin (extensions-builtin/bake-lora) also bakes the profiles used
in the last generation, at their strength, into the bias of the last layer of
each UNet block (works without any LoRA too). Exact for `add` mode with
Time buckets = 1; with more buckets the early/mid/late values are averaged
(noticeably different image). Start/End become strength x covered fraction,
*Prompt rows only* is dropped, `gain` is approximated. Don't apply the same
profile again on the baked model. Recipe is in the `sd_merge_recipe` metadata.

Steers are model specific: on a different architecture (SD1 vs SDXL) the
channel counts don't match and the blocks are skipped.

**Speed of the word maps** (`lib_act_steer/tokenmaps.py`): every
cross-attention layer reduces its attention to all words in one call, the
fused Triton kernel `attention_token_maps` from amd_tuned_torch
(`~/optimiert/amd_tools/source/cmp_ext_turing/amd_tuned_torch/attn_token_maps_ops.py`;
taken from the source tree while the venv's read-only copy predates it,
`ACT_STEER_TOKEN_MAPS_SRC` overrides the path), else the same math in plain
torch. Each image's maps are views into one stacked buffer, so adding a layer
and the per-step fade are one op each. Rows are taken as slices: indexing GPU
tensors with Python lists copies the indices from host memory, which on ROCm
waits for the GPU queue (one stall per layer and step). SD 1.5 512x1024 on the
RX 7900 XTX with word maps on: 7.5 -> 9.3 it/s (10.0 with them off).

**Separation LoRA (Forge: SD 1.x / SDXL / Anima).** The accordion "Separation LoRA
(train)" (or `POST /actsteer/train_separation`) generates N images, keeps the
UNet inputs of the prompt rows at a few sampling steps (fp16, on the CPU),
trains a rank-r LoRA on every cross-attention's to_q / to_k (Anima: the
q_proj / k_proj of all 52 blocks' cross_attn, words = T5 tokens) so nouns of
different phrases attend to different areas (KL term keeps the rest of the
attention), and saves it to the LoRA folder (`<lora:name:1>`). Memory: only
32 small thumbnails are kept for the gallery; training stops (and saves what
it has) when less than 4 GB RAM is available, generating stops with nothing
saved; the GPU cache is emptied afterwards so the next generation doesn't
offload the model to RAM.
The UNet inputs are kept at 0-60 % of the sampling and steps up to 30 %, where
the layout forms, are drawn four times as often.
Each training step runs the network once without gradients (the layers' inputs
are kept), takes the loss gradient of the summed word maps, then recomputes
one layer at a time with gradients: same gradient, but only one layer's
attention is in VRAM (Anima at 1024px: 4096 positions x 512 tokens x 16 heads
per layer).
