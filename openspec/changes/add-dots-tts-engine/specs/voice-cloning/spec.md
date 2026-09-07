# voice-cloning delta — clone mode is an engine capability, not an implicit flag

Scope: the **manager path** (`_MANAGER_PATH`) only.

## ADDED Requirements

### Requirement: An engine maps the clone mode to a named path

`x_vector_only_mode` in a voice's `meta.json` SHALL select the engine's
speaker-embedding-only cloning path, and its absence or `false` SHALL select the
engine's reference-plus-transcript path. An engine SHALL implement both paths
explicitly; it SHALL NOT treat the flag as advisory.

For the dots engine specifically, the two paths are `prompt_audio_path` alone and
`prompt_audio_path` with `prompt_text` set from `ref_text`. They are not
interchangeable: on the reference voices measured for this change, the
transcript-bearing path scored 0.838–0.965 speaker similarity and the
embedding-only path 0.762–0.908.

#### Scenario: Embedding-only voice
- **WHEN** a dots voice registered `x_vector_only_mode=true` is synthesized
- **THEN** the reference transcript is not supplied to the model
- **AND** the timbre still comes from the reference clip

#### Scenario: Transcript-bearing voice
- **WHEN** a dots voice registered `x_vector_only_mode=false` is synthesized
- **THEN** the stored `ref_text` is supplied as the reference transcript

### Requirement: A voice that cannot serve its declared mode is rejected at registration

Registration SHALL fail with a 4xx naming the missing input when a voice declares
a clone mode its engine cannot serve with the artifacts supplied — in particular,
a voice registered `x_vector_only_mode=false` with an empty `ref_text` on an
engine whose reference-plus-transcript path requires one.

The server SHALL NOT accept the registration and quietly synthesize through the
other path. A silent downgrade produces a voice that sounds wrong long after the
registration that caused it, with nothing in the record connecting the two.

#### Scenario: Transcript mode without a transcript
- **WHEN** a voice is registered for the dots engine with
  `x_vector_only_mode=false` and an empty `ref_text`
- **THEN** the server responds 4xx naming `ref_text` as the missing input
- **AND** no voice directory is left behind

#### Scenario: Embedding-only mode needs no transcript
- **WHEN** a voice is registered for the dots engine with
  `x_vector_only_mode=true` and an empty `ref_text`
- **THEN** the registration succeeds

### Requirement: An engine does not claim a cloning input it ignores

An engine that has no use for a stored voice artifact SHALL ignore it without
reporting that it was applied. The dots engine has no tone-seed input: a voice
carrying `seed.wav` and `seed_text` SHALL still register and synthesize, and the
server SHALL NOT represent that voice as tone-locked under this engine.

#### Scenario: A seeded voice on an engine with no seed input
- **WHEN** a voice carrying `seed.wav` and `seed_text` is registered under the
  dots engine
- **THEN** registration succeeds and synthesis works
- **AND** nothing in `GET /health` or the voice's reported state claims a locked
  tone for that voice
