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

**Status: proposed, not implemented.** No code in this repo has changed.

## The one thing to read before implementing

The win is not uniform. Against the production VoxCPM baseline (0.753 / 0.754
Resemblyzer cosine), dots.tts scored **0.941–0.965** on the Chinese narration
voice but only **0.762** on a VoxAlert pack in `x_vector_only_mode` — the mode all
76 packs use. Its advantage there needs continuation mode, which needs the
reference transcript. Whether that re-imports the synthetic cadence VoxAlert
switched to x-vector mode to avoid is an audible judgement and is explicitly
deferred to a human (`design.md` → Open questions).
