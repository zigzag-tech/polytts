# add-dots-tts-engine

Adds **dots.tts** (rednote-hilab, Apache-2.0) as a fourth polytts engine on the
manager path, beside `qwen`, `voxcpm` and `cosyvoice`.

Origin: a September 2026 survey of open speech models against what polyasr and
polytts have run since January. On the TTS side one model swung out clearly; on
the ASR side nothing did, and Qwen3-ASR-1.7B stays. Full survey, including the
A/B audio and the ASR reasoning:
<https://claude.ai/code/artifact/a3088209-3371-4226-ae32-7a1cc49e2d5a>

- `proposal.md` — why, with the measured numbers and what is out of scope
- `design.md` — checkpoint choice, clone-mode decision, footprint, open questions
- `tasks.md` — implementation checklist
- `specs/tts-engines/` — engine registry, sampling config, eviction, streaming
- `specs/voice-cloning/` — clone mode as an engine capability

**Status: implemented on the manager path.** Deployed and measured on
**xc-tower-ubuntu** (RTX 3090, `torch 2.11.0+cu130`), where polytts now registers
four engines and serves `dots` as the pinned resident engine.

## Measured on the deployment host

All numbers below are from the LIVE server (`POST /tts/stream`), not in-process,
with `dots-studio/dots.tts-mf` at NFE 4 and `optimize=True`.

Both engines measured **warm and pinned** — the only honest comparison, since an
unpinned engine pays a model load per request and that measures the residency
planner, not the model. Ten benchday narration packs, same line, same references.

| | dots.tts-mf | production VoxCPM |
|---|---|---|
| Time to first audio byte | 115 – 186 ms (x-vector), 267 ms (continuation) | 94 – 260 ms |
| RTF | **0.22 – 0.23** (x-vector), 0.24 – 0.30 (continuation) | 0.53 – 0.59 |
| Sample rate | 48 000 (`X-Sample-Rate`) | 48 000 |
| HTTP chunks, ~15 s phrase | 50 – 54 | — |
| VRAM reserved | 6.3 GB x-vector / **7.6 GB continuation, 60 s reference** | ~6.1 GB |
| Model load | 362 s cold / 38 – 65 s with a warm inductor cache | 54 – 110 s |

**So the win is throughput, not first-audio.** dots is ~2.4× faster overall
(RTF 0.23 against 0.55); time-to-first-audio is a wash, and on some packs VoxCPM
is marginally quicker off the mark. An earlier draft of this file claimed a 20×
TTFA win by comparing dots' warm server-side latency against VoxCPM's ~2.7 s
*field* figure — which includes the phone's network and cold starts. That was an
apples-to-oranges comparison and is retracted.

Determinism holds: the same `(text, voice_id)` produced **byte-identical** PCM
across repeat requests, across a cold-load request, and across a process restart.

### Three things measurement changed

1. **The footprint was wrong.** 6.5 GB was an x-vector-mode number. Continuation
   against a 60 s reference reserved 7.6 GB and an intermediate run OOM'd at
   7.1 GB. Now `8_000_000_000`. See `design.md` → footprint.
2. **The pin is the latency.** With `POLYTTS_DEFAULT_ENGINE=voxcpm`, the planner
   restores voxcpm after each dots synthesis, so every dots request reloads the
   model — 115 s wall for a phrase that generates in 2.46 s. Pinning dots makes
   it 2.3 s. See `design.md` → residency.
3. **The venv could not host it as written.** `torch 2.14` has no matching
   `torchaudio` on cu130, so the pair is now pinned at 2.11.0. See `design.md`
   → Risks.

### Not verified here

- **qwen voices 500 on this host** — `models/` does not exist on xc-tower-ubuntu,
  a pre-existing condition unrelated to this change. The qwen regression leg of
  task 5.5 is therefore untested; voxcpm was verified and is unchanged.
- **benchday narration is NOT served by this host.** The hub's TTS targets are
  `xc-mac-studio-tts` and `zz-tower0-tts`; xc-tower-ubuntu is an ASR target only.
  Making dots benchday's narration engine is a separate deployment step, and
  xc-mac-studio runs the MLX path, which this change does not touch — so moving
  the packs to dots takes the mac out of narration service.

## Whether benchday should switch — measured, not decided

All 10 narration packs, rendered three ways through the live server and scored
against their own reference clips. Clips and full table:
<https://claude.ai/code/artifact/7f7d0798-3acc-4b17-87e5-6839ae14f87a>

| | mean similarity | beats today | RTF |
|---|---|---|---|
| VoxCPM (today) | 0.860 | — | 0.53 – 0.59 |
| dots, x-vector (today's clone mode) | 0.762 | **0 / 10** | 0.22 – 0.23 |
| dots, continuation | **0.871** | **8 / 10** | 0.24 – 0.34 |

A straight engine swap in the packs' existing clone mode is a regression on every
one of them. Continuation mode is the only version of this switch worth making,
and it takes two exceptions with it: `hl-hev-suit` collapses to 0.16 s of audio,
and `sc2-protoss-advisor` slips 0.945 → 0.911.

**No pack has been switched.** The dots-registered voices exist on
xc-tower-ubuntu (both modes, distinct ids) but `BENCHDAY_NARRATION_PACKS` is
untouched, so narration still serves VoxCPM everywhere.

## The one thing to read before implementing

The win is not uniform. Against the production VoxCPM baseline (0.753 / 0.754
Resemblyzer cosine), dots.tts scored **0.941–0.965** on the Chinese narration
voice but only **0.762** on a VoxAlert pack in `x_vector_only_mode` — the mode all
76 packs use. Its advantage there needs continuation mode, which needs the
reference transcript. Whether that re-imports the synthetic cadence VoxAlert
switched to x-vector mode to avoid is an audible judgement and is explicitly
deferred to a human (`design.md` → Open questions).
