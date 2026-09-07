# Design — dots.tts engine

Target runtime: **manager path only** (`_MANAGER_PATH`, i.e. any non-MLX runtime).
The MLX path is out of scope; see `proposal.md` → Non-goals.

## Where it plugs in

`engines.py` already states the contract an engine must satisfy:

```
class Engine:
    name: str
    sample_rate: int
    load()                  # lazy; called by the ModelManager unit loader
    unload()                # drop GPU state, free the allocator, trim RAM
    loaded -> bool
    prepare_voice(voice_id, voice_dir, meta)
    generate(text, voice_id, voice_dir, meta, language, gen_kwargs) -> (np.float32[], sr)
    stream(text, voice_id, voice_dir, meta, language, gen_kwargs)   -> yields np.float32[]
```

`server.py` then references the engine in exactly three places, all inside the
`if _MANAGER_PATH:` block or the `/voices` handler:

| Location | Today | Why it matters |
|---|---|---|
| `_FOOTPRINTS` (server.py:129) | `{"qwen": 8.7e9, "voxcpm": 5e9, "cosyvoice": 3e9}` | The livestack planner sizes eviction from this. A wrong number makes the host broker mis-plan residency. |
| `_ENGINES` (server.py:146) | `{"qwen": …, "voxcpm": …, "cosyvoice": …}` | Builds the `ManagedUnit` per engine. `DEFAULT_ENGINE` gets `SOFT_PIN`; the rest are `UNPINNED`. |
| `/voices` allowlist (server.py:912) | `if engine not in ("qwen","voxcpm","cosyvoice")` | The only validation of the `engine` form field. |

`_resolve_engine()` and the `/tts` + `/tts/stream` handlers are engine-agnostic
already — they read `entry["engine"]` from the voice registry and call
`manager.ensure(name)`. Nothing there changes.

## Decision: which checkpoint

`dots.tts-mf` (MeanFlow-distilled) at **NFE 4** with **`optimize=True`**.

Rejected alternatives, with the reason each was rejected:

- **`dots.tts-soar`** — the quality ceiling (81.0 vs 80.0 zh SIM). It OOM'd twice
  on the bench host beside polyasr + polytts: *"Tried to allocate 900.00 MiB.
  GPU 0 has a total capacity of 23.56 GiB of which 148.50 MiB is free."* A
  checkpoint that only runs when its neighbours are evicted is not a co-tenant.
  Revisit with a dedicated card.
- **`dots.tts-base`** — the pretraining baseline; SOAR strictly dominates it.
- **`dots.tts-mf-1step` / `-2steps`** — lower latency still, at 1.02 / 1.00 zh
  WER and 80.2 / 80.4 SIM. Worth measuring later; NFE 4 is the variant this
  change measured, so it is the variant this change ships.
- **`dots.tts-mf-2steps-stts`** — double-streaming, no consumer. See Non-goals.

**The sampling settings are load-bearing, not tuning.** Run at the library
default (10 steps, no `torch.compile`) the same checkpoint measured **RTF 1.81 –
2.02 on the bench host — slower than real time**, against production VoxCPM's
0.60. At NFE 4 with `optimize=True` it measured **RTF 0.23 – 0.37**. The defaults
would have read as a regression and buried the result. The engine therefore sets
both explicitly rather than inheriting them, and `design`-level intent is
recorded in a comment next to the constant.

`optimize=True` compiles at load, so first-load is slower. This is acceptable
because the manager loads lazily and idle-evicts on a long timer; it is the same
trade the CUDA graph path already makes elsewhere.

## Decision: clone mode is engine-declared, not inferred

This is the part of the change that is not a straight port.

`meta.json` carries `x_vector_only_mode`. For VoxCPM that flag is close to free —
it swaps how the reference conditions generation. For dots.tts the two modes are
materially different paths with **measured, different quality**:

| Mode | dots.tts call | Measured sim (VoxAlert pack) | Measured sim (zh voice) |
|---|---|---|---|
| continuation | `prompt_audio_path` + `prompt_text` | 0.838 | 0.941 – 0.965 |
| x-vector only | `prompt_audio_path` alone | 0.762 | 0.908 |

Against a production VoxCPM baseline of 0.754 / 0.753, continuation wins clearly
and x-vector-only is a wash on the VoxAlert voice. So the engine must not treat
the mode as an implementation detail:

- `DotsEngine` maps `x_vector_only_mode=true` → omit `prompt_text`, and
  `false` → pass `meta["ref_text"]` as `prompt_text`. Direct, no cleverness.
- A voice registered `x_vector_only_mode=false` whose `ref_text` is empty cannot
  serve continuation. Today an engine would quietly do something else. This
  change requires the engine to **fail the registration**, not degrade silently —
  a silent downgrade is exactly the kind of defect that shows up months later as
  "the voice sounds different now".

`ref_text` accuracy matters more here than for VoxCPM: upstream states that a
`prompt_text` which does not match the audio "degrades stability and may cause
word-level errors". Our stored `ref_text` values came from ASR over the reference
clip, so they are close but not guaranteed exact. This is a known risk, recorded
below rather than solved here.

## Decision: no tone-seed equivalent

VoxCPM voices may carry `seed.wav` + `seed_text` — a *second* clip that locks
prosody, separate from the timbre reference. dots.tts has no such input: one
prompt carries both, and `seed=` is an RNG seed that *varies* prosody rather than
locking it.

Rather than fake it, `DotsEngine` ignores `seed.wav` and pins a **deterministic
RNG seed derived from `voice_id`**, so repeated synthesis of the same text in the
same voice is byte-identical — which the `/tts/stream` disk PCM cache already
assumes. The measured long-form drift (spread 0.032 over 45 s, floor 0.943) is
the evidence that the missing seed-lock does not cost us what it costs VoxCPM.

`meta.json` keeps `seed_text` for voices that have it; the field simply has no
effect under this engine. `GET /health` and voice metadata must not claim a
capability the engine does not have.

## Decision: footprint and residency

`_FOOTPRINTS["dots"] = 6_500_000_000`, from the measured peak of 5.4 – 6.4 GB
(6.4 GB on the 45 s long-form request, the worst case measured). Rounded up, not
down: the planner uses this to decide whether a unit fits, and under-reporting
produces exactly the OOM that killed the SOAR run.

Residency policy follows the existing rule — `SOFT_PIN` only if `dots` is
`DEFAULT_ENGINE`, otherwise `UNPINNED`. This change does **not** move
`POLYTTS_DEFAULT_ENGINE`; that is a deployment decision per host.

## Sample rate

dots.tts emits **48 kHz**, the same as VoxCPM. `/tts/stream` already advertises
the resident engine's rate in `X-Sample-Rate` and Benchday reads it
(`tts_client.dart`, default 24000 only as a fallback). No client change, and no
resampling in the server.

## Open questions — decisions for a human, not for the implementer

1. **Should the VoxAlert packs move to continuation mode?** The data says
   continuation is where dots.tts wins (0.838 vs 0.762 vs today's 0.754). The
   reason they are x-vector-only is a *VoxCPM* artifact — in-context cloning a
   synthetic clip copied its machine cadence. Whether dots.tts reproduces that
   artifact is an audible judgement, not a cosine one. A/B clips exist from the
   survey run; this must be listened to before any pack is re-registered.
2. **Is the community MLX port good enough for xc-mac-studio?**
   `sb1992/dots-tts-mlx` v0.5.1 has streaming, int4/int8 weights and a reference
   "enrolment" cache, but is unofficial and does not document x-vector-only mode
   — the mode every VoxAlert pack uses. Until answered, mac-studio stays VoxCPM.
3. **Is SOAR worth a dedicated card?** +1.0 zh SIM over MF NFE 4, at a footprint
   that cannot share a 24 GB GPU with polyasr.

## Risks

- **`ref_text` drift.** Stored transcripts are ASR output, not verified
  transcripts. A mismatched `prompt_text` degrades dots.tts specifically. The
  registration-time check above catches *empty*, not *wrong*.
- **torch pairing.** `dots.tts` fails at import when torch and torchaudio minor
  versions differ. Installing it into an existing venv can silently upgrade torch
  and break the pair; it must be installed against the venv's existing torch.
- **First-load latency.** `optimize=True` compiles at load. On a cold engine the
  first request pays for it; the manager's idle timer decides how often that
  happens.
- **Not a regression risk:** no existing voice changes engine, and no existing
  engine is removed, so the blast radius of a bad dots.tts is voices explicitly
  registered under it.
