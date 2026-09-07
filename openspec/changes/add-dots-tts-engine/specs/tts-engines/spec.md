# tts-engines delta — add dots.tts as a fourth polytts engine

Scope: the **manager path** (`_MANAGER_PATH`) only. The MLX path in `server.py`
does not use `engines.py` and is unchanged by this delta.

## ADDED Requirements

### Requirement: The engine registry carries a dots.tts engine

polytts SHALL expose `dots` as a selectable engine on the manager path,
implemented as an `Engine` subclass in `engines.py` and registered in every place
the server enumerates engines: the footprint table, the engine map, and the
`/voices` engine allowlist. Registering it SHALL NOT remove or alter `qwen`,
`voxcpm` or `cosyvoice`, and SHALL NOT change the engine any already-registered
voice serves from.

#### Scenario: The engine is selectable at registration
- **WHEN** a voice is registered with `engine=dots`
- **THEN** the registration succeeds and `meta.json` records `"engine": "dots"`
- **AND** subsequent `/tts` and `/tts/stream` requests for that `voice_id` route
  to the dots engine without the caller naming it again

#### Scenario: An unknown engine is still rejected
- **WHEN** a voice is registered with an engine name that is not one of the four
- **THEN** the server responds 400 naming the unknown engine

#### Scenario: Existing voices are unaffected
- **WHEN** the dots engine is added to a server holding voices registered under
  `qwen`, `voxcpm` or `cosyvoice`
- **THEN** each of those voices continues to serve from the engine it was
  registered under
- **AND** no voice changes `voice_id`

### Requirement: The dots engine declares the sampling configuration it was measured at

The engine SHALL set its checkpoint, sampling-step count and compile switch
explicitly rather than inheriting library defaults, and each SHALL be overridable
by a `POLYTTS_DOTS_*` environment variable. The shipped defaults SHALL be the
configuration under which the engine's performance was measured.

This requirement exists because the library defaults are not a safe fallback: the
same checkpoint measured RTF 1.81–2.02 at the default 10 steps without
compilation, and RTF 0.23–0.37 at NFE 4 with compilation, on the same host.

#### Scenario: Defaults are the measured configuration
- **WHEN** the dots engine loads with no `POLYTTS_DOTS_*` variables set
- **THEN** it loads the MeanFlow checkpoint with 4 sampling steps and
  compilation enabled

#### Scenario: An operator overrides the checkpoint
- **WHEN** `POLYTTS_DOTS_MODEL` names a different dots.tts checkpoint
- **THEN** the engine loads that checkpoint
- **AND** `GET /health` reports the checkpoint actually resident, not the default

### Requirement: The dots engine participates in single-resident eviction

The engine SHALL load lazily on first use, SHALL free its GPU state and return
host memory on `unload`, and SHALL declare a footprint at or above its measured
peak allocation. It SHALL NOT require co-residency with another engine.

Declaring a footprint below the measured peak is a defect: the residency planner
sizes eviction from that number, and the checkpoint rejected by this change was
rejected precisely because it exceeded the free VRAM on a shared card.

#### Scenario: Lazy load
- **WHEN** the server starts and no request has named a dots voice
- **THEN** no dots checkpoint is resident in VRAM

#### Scenario: Eviction returns the VRAM
- **WHEN** the dots engine is resident and `POST /model/unload` is called
- **THEN** the response reports `dots` unloaded
- **AND** the VRAM it held is returned to the driver

#### Scenario: Switching engines evicts
- **WHEN** a request routes to a different engine while dots is resident
- **THEN** dots is evicted before the other engine loads
- **AND** at most one engine is resident throughout

### Requirement: Streaming synthesis yields chunks as they are produced

The dots engine's `stream` SHALL yield audio chunks as the model produces them,
so `/tts/stream` emits its first PCM bytes before synthesis completes. The
response SHALL advertise the engine's own sample rate in `X-Sample-Rate` rather
than assuming a server-wide rate.

#### Scenario: First audio precedes completion
- **WHEN** a client requests `/tts/stream` for a dots voice
- **THEN** the first PCM bytes arrive before the full utterance has been
  synthesized
- **AND** more than one chunk is emitted for a multi-sentence utterance

#### Scenario: The advertised rate is the engine's rate
- **WHEN** `/tts/stream` serves a dots voice
- **THEN** `X-Sample-Rate` reports 48000
- **AND** a client that reads the header plays the audio at the correct pitch

### Requirement: Repeated synthesis of the same text and voice is deterministic

Because the engine has no tone-seed input, it SHALL pin a generation seed derived
from the `voice_id` so that the same `(text, voice_id)` produces identical audio
across requests and across process restarts.

The `/tts/stream` disk PCM cache serves a stored body for an identical
`(text, voice_id)` without touching the GPU; an engine whose output varied per
request would make cached and uncached playback of the same phrase differ
audibly.

#### Scenario: Same request twice
- **WHEN** the same text is synthesized twice for the same dots voice
- **THEN** the two PCM bodies are identical

#### Scenario: Cache hit matches a fresh synthesis
- **WHEN** a phrase is served from the PCM cache and the same phrase is later
  synthesized after an eviction
- **THEN** the audio is the same
