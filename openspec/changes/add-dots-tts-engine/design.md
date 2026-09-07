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

`optimize=True` compiles at load, so first-load is slower. **Measured, this is
much larger than "slower":**

| Load | Elapsed |
|---|---|
| First ever on the host (cold inductor cache) | **362 s** |
| With a populated inductor cache | **38 – 65 s** |
| VoxCPM, for comparison | ~54 s (~110 s under compile) |

Nearly all of the 362 s is `torch.compile` codegen, and inductor caches it on
disk — so the cost is paid **once per host**, not once per restart, *provided the
cache outlives a reboot*. Its default location is under `/tmp`, which is exactly
where it does not outlive one. `TORCHINDUCTOR_CACHE_DIR` is therefore pinned to
`~/.cache/torchinductor-polytts` in the polytts systemd drop-in; without that, the
6-minute compile returns on every boot, and the server preloads
`POLYTTS_DEFAULT_ENGINE` **synchronously at startup**, so it is paid before the
port answers.

The cache is portable between processes on the same host: seeding the persistent
directory from a previous run's `/tmp` cache cut the next load from 362 s to 94 s
without any other change.

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

`_FOOTPRINTS["dots"] = 8_000_000_000`.

**Corrected during implementation, from 6.5 GB.** The survey's 5.4 – 6.4 GB was
measured in x-vector mode. Serving *continuation* against a 60 s reference clip
(687-char `ref_text`) the same checkpoint reserved **7.6 GB** on an RTX 3090, and
an intermediate run OOM'd at 7.1 GB on a card with ~7.7 GB free. The peak is set
by the **longest registered reference clip**, because continuation encodes the
prompt audio in full while the x-vector path truncates it at
`xvec_max_audio_seconds = 10`. benchday's own narration packs hold references
from 8 s to 60 s, so the long tail is not hypothetical.

8 GB is 7.6 GB rounded up. Do not tune it down to the tidier x-vector number:
this change's own rule — that under-reporting a footprint is what disqualified
SOAR — was violated by the first estimate, and the OOM followed.

Residency policy follows the existing rule — `SOFT_PIN` only if `dots` is
`DEFAULT_ENGINE`, otherwise `UNPINNED`. The change itself does not move
`POLYTTS_DEFAULT_ENGINE`; that is a deployment decision per host.

**But that deployment decision dominates the latency, and it is not a tuning
knob.** With `POLYTTS_DEFAULT_ENGINE=voxcpm`, the residency planner restores the
soft-pinned voxcpm the instant a dots synthesis completes, so the *next* dots
request reloads the model: measured **115 s wall for an 11.2 s phrase whose
generation took 2.46 s**. With `POLYTTS_DEFAULT_ENGINE=dots` the same phrase is
2.3 s. A host that serves dots voices and does not pin dots is not running a slow
engine — it is paying a model load per request, and the fix is the pin, not the
sampling config.

The converse cost is real and belongs in the deployment note: with dots pinned,
the first `voxcpm` or `qwen` request on that host pays the eviction and a cold
load (measured **89 s** for voxcpm), after which the planner restores dots (38 s).
On a host shared with Voxlert, which uses voxcpm, that is a per-switch tax.

## Sample rate

dots.tts emits **48 kHz**, the same as VoxCPM. `/tts/stream` already advertises
the resident engine's rate in `X-Sample-Rate` and Benchday reads it
(`tts_client.dart`, default 24000 only as a fallback). No client change, and no
resampling in the server.

## Open questions — decisions for a human, not for the implementer

1. **Should the packs move to continuation mode?** — **MEASURED, still a human
   call.** All 10 benchday narration packs were rendered three ways through the
   live server and scored against their own reference clips (Resemblyzer cosine):

   | | mean similarity | beats today | RTF |
   |---|---|---|---|
   | VoxCPM (today) | 0.860 | — | 0.53 – 0.59 |
   | dots, x-vector (today's clone mode) | **0.762** | **0 / 10** | 0.22 – 0.23 |
   | dots, continuation | **0.871** | **8 / 10** | 0.24 – 0.34 |

   So a straight engine swap in the *existing* clone mode is a regression on
   every pack, and the survey's "wash" reading (0.762 vs 0.754, one VoxAlert
   voice) does not hold against a proper warm VoxCPM baseline. Continuation is
   the only mode where the switch pays.

   Two packs do not come along:
   - **hl-hev-suit** — continuation returns **0.16 s of audio**. A failed
     generation, not a worse voice. 30 s reference, 289-char transcript.
   - **sc2-protoss-advisor** — 0.911 against 0.945. Small but real.

   The remaining judgement is audible, not numeric: cosine says "close to the
   reference speaker", not "right for the character". Clips:
   <https://claude.ai/code/artifact/7f7d0798-3acc-4b17-87e5-6839ae14f87a>
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
  and break the pair.

  **Implementation found this stronger than written.** The deployed venv on
  xc-tower-ubuntu held `torch 2.14.0+cu130` against `torchaudio 2.11.0+cu130`, and
  torchaudio has **no** cu130 build past 2.11 — so "install against the venv's
  existing torch" was not available: the pair could only be matched by pinning
  torch *down* to 2.11.0. `requirements-pytorch.txt` therefore pins **both**
  (`torch==2.11.0`, `torchaudio==2.11.0`) instead of floating them, and `setup.sh`
  asserts the pair at install time — because the drift is silent everywhere else
  in this venv and surfaces only as "the dots voices 503".

  Verified on a copy of the deployed venv before the deployed one was touched:
  torch 2.11.0 still imports and runs `voxcpm`, `qwen_tts` and `torchcodec`.
  dots.tts additionally pulls `numpy>=2`, which trips funasr's declared `numpy<2`;
  funasr imports and runs on numpy 2.5.3, so that `pip check` conflict is expected
  rather than a break.

  (The related suspicion that this venv's `torchaudio.lib._torchaudio` C extension
  was broken by the mismatch is **wrong** and should not be chased: torchaudio
  2.9+ ships no such extension at all — it delegates to torchcodec. The module is
  absent on a matched pair too.)
- **First-load latency.** `optimize=True` compiles at load. On a cold engine the
  first request pays for it; the manager's idle timer decides how often that
  happens.
- **Not a regression risk:** no existing voice changes engine, and no existing
  engine is removed, so the blast radius of a bad dots.tts is voices explicitly
  registered under it.
