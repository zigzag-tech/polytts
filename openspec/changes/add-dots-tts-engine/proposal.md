# Add dots.tts as a fourth polytts engine

## Why

polytts' engines were chosen in January 2026. A September 2026 survey of what has
shipped since found one model that beats all three of ours on the metric our
consumers actually depend on — **speaker similarity** — and it was verified on
our own hardware against our own production voices, not on a leaderboard.

**dots.tts** (rednote-hilab, June 2026; MeanFlow and SOAR checkpoints August
2026) is a 2B fully-continuous autoregressive TTS: a Qwen2.5-1.5B text backbone,
an AR flow-matching head over a 48 kHz AudioVAE, and a frozen CAM++ speaker
x-vector as side input. **Apache-2.0 for both code and weights.**

Published, Seed-TTS-Eval, zh WER ↓ / SIM ↑:

| Model | Params | zh WER | zh SIM | en WER | en SIM |
|---|---|---|---|---|---|
| dots.tts (SOAR) | 2B | 0.94 | **81.0** | 1.30 | **77.1** |
| dots.tts (MeanFlow, NFE 4) | 2B | 0.94 | 80.0 | 1.29 | 76.2 |
| VoxCPM 2 — *we ship this* | 2B | 0.97 | 79.5 | 1.84 | 75.3 |
| CosyVoice 3 — *we ship this* | 1.5B | 1.12 | 78.1 | 2.22 | 72.0 |
| Qwen3-TTS — *we ship this* | 1.7B | 1.22 | 77.0 | 1.23 | 71.7 |

A leaderboard is measured on someone else's voices, so the survey measured ours.
On **xc-tower-ubuntu (RTX 3090)**, dots.tts-mf at NFE 4 with `optimize=True` was
run against two real clones from `voices/` and compared to the **live polytts**
serving VoxCPM through the same `/tts/stream` endpoint Benchday calls. Similarity
is Resemblyzer cosine against the reference clip; CER is polyasr (Qwen3-ASR-1.7B)
transcribing the output back.

| Engine · clone mode | Utterance | Sim ↑ | CER ↓ | TTFA ms | RTF ↓ |
|---|---|---|---|---|---|
| dots.tts · continuation | zh narration | **0.965** | 0.000 | 1027 | 0.33 |
| dots.tts · continuation | zh alert | **0.941** | 0.000 | 502 | 0.37 |
| dots.tts · x-vector only | zh narration | **0.908** | 0.000 | 409 | 0.27 |
| VoxCPM 2 (production) | zh narration | 0.753 | 0.000 | 287 | 0.61 |
| dots.tts · continuation | en alert (VoxAlert voice) | **0.838** | 0.032 | 580 | 0.29 |
| dots.tts · x-vector only | en alert (VoxAlert voice) | 0.762 | 0.032 | 86 | 0.23 |
| VoxCPM 2 (production) | en alert (VoxAlert voice) | 0.754 | 0.032 | 175 | 0.58 |

Long-form (a 45–52 s paragraph, similarity per quarter): dots.tts 0.974 overall,
drifting 0.975 → 0.943; VoxCPM 0.761, drifting 0.760 → 0.745. dots.tts drifts
*more* in absolute spread (0.032 vs 0.023) and still never falls near VoxCPM's
best quarter. **The VoxCPM tone-seed lock is not required to beat today's
quality.**

Three facts decide the shape of this change:

1. **The win is real and large on the Chinese narration voice** — +0.19 cosine
   over the engine we ship, at roughly half the RTF, in the same VRAM class.
2. **The win on VoxAlert's voices depends on the clone mode.** All 76 packs are
   registered `x_vector_only_mode=true`, because polytts' own README warns that
   in-context cloning a *synthetic* clip copies its machine cadence. That is
   dots.tts' weaker path: 0.762 against today's 0.754 — a wash. Its advantage
   only appears in continuation mode, which needs the reference transcript. We
   already store one: every pack has `ref_text` in `meta.json`.
3. **The best checkpoint does not fit our shared card.** `dots.tts-soar` (the
   81.0 SIM one) OOM'd twice on a 24 GB RTX 3090 beside polyasr and polytts. The
   MeanFlow checkpoint at NFE 4 fits at 5.4–6.4 GB peak — the same class as
   VoxCPM's 5.5 GB resident — and gives up 1.0 SIM to do so.

polytts already has the abstraction for this. `VoxcpmEngine` is ~60 lines against
the `Engine` base, and dots.tts' runtime arguments line up with our voice
metadata almost exactly: `prompt_audio_path` + `prompt_text` is `voice.wav` +
`ref_text`; omitting `prompt_text` is `x_vector_only_mode`.

## What changes

- **A `DotsEngine` in `engines.py`**, subclassing `Engine` like `VoxcpmEngine`:
  lazy `load` / `unload`, `prepare_voice`, `generate`, and a `stream` that yields
  PCM chunks from `generate_stream()`.
- **Registration in `server.py`** in the three places an engine appears:
  `_FOOTPRINTS`, `_ENGINES`, and the `/voices` engine allowlist.
- **Clone mode becomes an explicit, per-engine decision** rather than an implicit
  consequence of `x_vector_only_mode`. The flag keeps its meaning; what changes
  is that an engine declares which mode it is able to serve and the server stops
  silently degrading when it cannot.
- **New `POLYTTS_DOTS_*` env vars** for checkpoint id, sampling steps and the
  `optimize` switch, defaulting to the configuration that was actually measured.
- **README + `requirements-pytorch.txt`** gain the engine and its dependency.

Existing voices, existing engines and every client are untouched. dots.tts is an
opt-in fourth engine selected per voice, exactly like `cosyvoice` was.

## Non-goals

- **Not replacing VoxCPM, Qwen3-TTS or CosyVoice.** All three stay registered and
  every already-registered voice keeps serving from the engine it was registered
  under. This change adds a choice; it does not migrate anything.
- **Not the MLX path.** `server.py`'s Apple-Silicon path does not use
  `engines.py`, and the only dots.tts MLX implementation is a community port
  (`sb1992/dots-tts-mlx`) whose x-vector-only mode is undocumented. xc-mac-studio
  stays on VoxCPM until that is evaluated separately.
- **Not `dots.tts-soar`.** It is the higher-quality checkpoint and it does not fit
  a card shared with polyasr. Revisit only with a dedicated GPU.
- **No client changes.** Benchday reads the sample rate from `X-Sample-Rate` and
  already serves 48 kHz from VoxCPM; VoxAlert selects a pack, not an engine.
  Unchain's `TtsProvider` union gains `'dots'` in a follow-up in that repo — it
  is not in scope here.
- **Not migrating the VoxAlert packs.** Whether their `x_vector_only_mode=true`
  should be flipped to continuation is a listening judgement (see `design.md`,
  "Open questions"), not something this change decides.
- **Not double-streaming.** `dots.tts-mf-2steps-stts` accepts incremental text
  tokens for duplex dialogue. polytts synthesizes complete sentences; there is no
  consumer for it today.

## Impact

- `engines.py` — one new class, no change to existing ones.
- `server.py` — three registration lines on the manager path only.
- `requirements-pytorch.txt` — adds `dots.tts`, which pins `torch>=2.8.0` and
  **rejects a torch/torchaudio minor-version mismatch at import**. The survey's
  first install resolved torch 2.14 against torchaudio 2.11 and failed to import;
  the deployed pair (2.11.0+cu129) is compatible and must be installed together.
- VRAM: one more evictable unit at a measured 5.4–6.4 GB peak. It does not
  co-reside — the manager keeps one engine resident.
- Storage: no new unbounded store. dots.tts adds ~2.7 GB of weights to the HF
  cache; voice artifacts stay in the existing `voices/` tree under the existing
  `POLYTTS_MAX_VOICES_MEM` cap.
