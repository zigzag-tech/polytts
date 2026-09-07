# Tasks — add dots.tts as a fourth polytts engine

Manager path only. Nothing below touches the MLX path in `server.py`.

## 1. Dependency

- [x] 1.1 **Done, and stronger than planned:** the deployed venv held torch 2.14
      against torchaudio 2.11, and torchaudio has no cu130 build past 2.11 — so
      "install against the venv's existing torch" was impossible and BOTH are now
      pinned at 2.11.0. `requirements-pytorch.txt` — add `dots.tts` (0.3.1 is the version the
      survey verified). Add a comment recording that the package **fails at
      import** when torch and torchaudio minor versions differ, so it must be
      installed against the venv's existing torch rather than allowed to resolve
      its own. The deployed pair is `torch==2.11.0+cu129` / `torchaudio==2.11.0+cu129`.
- [x] 1.2 `setup.sh` — install dots.tts without upgrading torch (`--no-deps`, or
      install after torch and assert the pair). Verify `python -c "import
      dots_tts"` succeeds in the venv `run.sh` resolves.
- [x] 1.3 Verify: on a host with the deployed venv, `import dots_tts` raises no
      `RuntimeError` about mismatched minors.

## 2. The engine

- [x] 2.1 `engines.py` — add `DotsEngine(Engine)` beside `VoxcpmEngine`:
      `name = "dots"`, `sample_rate = 48000`. Read `POLYTTS_DOTS_MODEL`
      (default `dots-studio/dots.tts-mf`), `POLYTTS_DOTS_NUM_STEPS` (default 4)
      and `POLYTTS_DOTS_OPTIMIZE` (default on) in `__init__`.
- [x] 2.2 `engines.py` — `load()`: `DotsTtsRuntime.from_pretrained(model,
      precision="bfloat16", optimize=…)`, then `gc.collect()` + `_trim_ram()`,
      mirroring `VoxcpmEngine.load`'s reclaim of the CPU staging buffers.
      Comment **why** NFE 4 + optimize are set explicitly: at library defaults the
      same checkpoint measured RTF 1.81–2.02 on an RTX 3090, against 0.23–0.37
      configured — a comment here is what stops a future edit from "simplifying"
      the constants away.
- [x] 2.3 `engines.py` — `unload()`: drop the runtime, `gc.collect()`,
      `_free_cuda()`, `_trim_ram()`, exactly as `VoxcpmEngine.unload` does.
- [x] 2.4 `engines.py` — `_clone_kwargs(voice_dir, meta)`: always
      `prompt_audio_path=voice_dir/"voice.wav"`; add
      `prompt_text=meta["ref_text"]` **unless** `meta["x_vector_only_mode"]`.
      Pin `seed` deterministically from `voice_id` (satisfies the determinism
      requirement the PCM cache depends on). Ignore `seed.wav` / `seed_text` —
      the engine has no tone-seed input.
- [x] 2.5 `engines.py` — `generate()` returns `(np.float32[], self.sample_rate)`;
      `stream()` yields `np.float32[]` per chunk from `generate_stream()`
      (`generate_stream` yields `torch.Tensor` of shape `(1, samples)`; convert
      with `.detach().float().cpu().numpy().reshape(-1)`).
- [x] 2.6 `engines.py` — `prepare_voice()` may stay a no-op, matching
      `VoxcpmEngine` (voice state is file paths). If reference-encoding caching is
      added later it belongs here, not in `generate`.

## 3. Registration

- [x] 3.1 `server.py:33` — import `DotsEngine` alongside the other engines.
- [x] 3.2 **Done, corrected to `8_000_000_000`** — 6.5 GB was an x-vector-mode
      number; continuation against a 60 s reference reserved 7.6 GB and an
      intermediate run OOM'd at 7.1 GB. Original text kept below for the record.
      `server.py:129` — `_FOOTPRINTS["dots"] = 6_500_000_000`. **Measured
      peak 5.4–6.4 GB** on an RTX 3090 (6.4 GB on a 45 s long-form request);
      rounded up deliberately, because the planner sizes eviction from this and
      under-reporting is what produced the OOM that disqualified the SOAR
      checkpoint.
- [x] 3.3 `server.py:146` — add `"dots": DotsEngine()` to `_ENGINES`. Do not
      change `POLYTTS_DEFAULT_ENGINE`; residency policy follows the existing
      `SOFT_PIN if name == DEFAULT_ENGINE` rule unchanged.
- [x] 3.4 `server.py:912` — add `"dots"` to the `/voices` engine allowlist.

## 4. Clone-mode validation

- [x] 4.1 `server.py` `/voices` handler — reject with 4xx, naming `ref_text`, a
      registration for an engine whose transcript path requires one when
      `x_vector_only_mode` is false and `ref_text` is empty. Leave no voice
      directory behind on rejection.
- [x] 4.2 Verify the negative and the positive: empty `ref_text` with
      `x_vector_only_mode=true` still registers.

## 5. Evidence — run these against a real server, not a mock

- [x] 5.1 Register a dots voice from an existing `voices/*/voice.wav` +
      `ref_text`, both modes, and confirm two distinct `voice_id`s (the id hashes
      the mode).
- [x] 5.2 `POST /tts/stream` for a dots voice: assert more than one chunk
      arrives, that the first bytes precede completion, and that
      `X-Sample-Rate: 48000`.
- [x] 5.3 Determinism: synthesize the same `(text, voice_id)` twice, assert the
      PCM bodies are byte-identical, then evict and repeat to confirm a cache hit
      matches a fresh synthesis.
- [x] 5.4 Residency: with dots resident, request a voxcpm voice; assert `GET
      /health` reports exactly one resident engine throughout, and that `POST
      /model/unload` returns the VRAM.
- [x] 5.5 Regression: **voxcpm verified** through the live server on torch 2.11 +
      numpy 2.5, `voice_id`s unchanged. **qwen NOT verified** — it 500s on
      xc-tower-ubuntu because `models/` does not exist there, which predates this
      change. Re-run this leg on a host that has the qwen weights.
- [x] 5.6 Record measured similarity, CER, TTFA and RTF for at least one
      production voice, on the deployment host, into the change's README before
      archiving — a claim about quality is only as good as the host it was
      measured on.

## 6. Documentation

- [x] 6.1 `README.md` — add `dots` to the engines list and the env-var table
      (`POLYTTS_DOTS_MODEL`, `POLYTTS_DOTS_NUM_STEPS`, `POLYTTS_DOTS_OPTIMIZE`).
      State plainly that the engine has **no tone-seed input**, so `seed.wav` /
      `seed_text` are ignored under it.
- [x] 6.2 `README.md` — record the clone-mode difference measured for this
      engine, so the next person choosing `x_vector_only_mode` for a synthetic
      reference sees the trade-off rather than inheriting VoxCPM's guidance
      unexamined.

## 7. Deliberately not done here

- [x] 7.1 **Resolved 2026-09-07 — the owner declined the switch.** All 10
      benchday narration packs were measured in both clone modes against a warm
      VoxCPM baseline (`design.md` → Open questions 1). Continuation wins on
      similarity 8/10, but dots collapses to 0.16 s of audio on `hl-hev-suit`
      deterministically and unrecoverably. Owner's call: "If DOTS cannot handle
      this case, we shouldn't switch." No pack was re-registered; the evaluation
      voices were removed.
- [ ] 7.2 Unchain's `TtsProvider` union gains `'dots'` in that repo, not this one.
