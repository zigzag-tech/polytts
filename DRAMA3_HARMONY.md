# Fish Audio Drama3 native PolyTTS engine

Source addition2026-10-20 (not an activation claim): an explicit managed request
`voiceMode: "unconditioned"` omits both `voiceId` and `referenceSha256`. It is
Drama-only, requires exact owner membership in installed `unconditionedOwnerIds`,
and uses the real native `tts` session without setting reference audio or text.
No silence, placeholder voice ID or invented reference hash is supplied. The
same native TTS task also accepts the existing explicit registered reference.
Installed source identity must pin `nativeSessionTask: "tts"`; receipts bind
voiceMode and history cannot cross conditioning modes. Reference-free output
does not promise stable timbre between independent requests; retained native
history is the existing continuation mechanism, separately qualified.

Failure diagnostics now distinguish native frame-limit exhaustion from malformed
metadata and retain actual generation proof/digest in `receipt.nativeFailure`.
The original bounded proof file is retained owner-local as
`failed-drama3-generation.json`; no failed partial audio is published. This
changes no EOS decision, sampling or completion admission. Native upstream
documents that some emotion/nonverbal prompts prolong reactions or hit the cap;
reference-free support must be measured and does not itself guarantee a remedy.

An isolated candidate can register a distinct `POLYTTS_DRAMA3_UNIT_KIND`, bound
through installed `engineUnitKinds` and source identity, while the public engine
remains `drama3`. Actual acquisition/admission/run all use that exact mapped
unit. This lets the real Harmony planner evict the old idle unit before warming
the candidate, without restarting the old service or pretending its model is
already resident in the new process. Use a fresh no-preload namespace and keep
the old reference route/receipts intact; no global default changes are implied.

Implementation worktree only; runtime/model qualification is separate. This is exact Drama3, not Fish S2. Official source541566932abbaebcb0a40e562a1f846738383f38 identifies35B-total/3B-active parameters. The official Python FP8 runtime requires48GB; this adapter instead targets the community audio.cpp native GGML portccdbe76929e8a77428db6ec3bbbaadeb35acae38 through its actual C ABI0.2.

Selected checkpoint: Jmica/Drama-3-GGUF@17a5b40c5dc2c2be986f5371a7e650108b828e29, drama-3-backbone-8bit-experts-Q3_K_S.gguf,20325196992bytes, SHAb946a8c9806deba559590b98e37ffc9e0957109263e111513734fa93284db16b. Expert banks useQ3_K, other backbone matricesQ8_0, audio tower/codecFP16. Selection is a concrete24GB experiment with8CPU expert layers/2048context; actual memory/latency/quality must be measured. Upstream's24GB guidance was tested on a96GB card, not physical3090 hardware.

The engine loads and frees native model/session handles inside PolyTTS's single GPU executor. No independent application HTTP/model route bypasses Harmony. Exact native library and model bytes are checked before loading. All managed inference retains actual Harmony admission and the durable job receipt. Torch allocator peaks do not include native GGML allocations and are labelled accordingly.

Required environment: POLYTTS_DRAMA3_MODEL, POLYTTS_DRAMA3_MODEL_SHA256, POLYTTS_DRAMA3_MODEL_REVISION, POLYTTS_DRAMA3_LIBRARY, POLYTTS_DRAMA3_LIBRARY_SHA256, POLYTTS_DRAMA3_SOURCE_REVISION. Optional closed session settings: POLYTTS_DRAMA3_CPU_EXPERT_LAYERS (0..40, default8), POLYTTS_DRAMA3_CONTEXT_LENGTH (2048/4096/8192, default2048).

Managed request engine`drama3` uses explicit temperature/topP/topK/maxTokens/seed/textChunkSize controls. maxTokens bounds acoustic frames per native text chunk; it is not seconds or words. Text uses actual native speaker/event tags; there is no invented SSML/durationFactor. For exact continuation/edit history, history contains exact same-owner/source/voice completed job/request/artifact identities. The managed reader reauthenticates the native source receipt and bytes before passing its original codebooks; arbitrary supplied VQ grids and server paths are refused. The resulting native drama3_take JSON is retained byte exact as drama3-take.json. History is resolved only from an exact completed same-owner/source/voice native job and rehashed actual artifacts; arbitrary supplied VQ grids are not accepted. One registered synthetic reference is supported; no private voices are supplied. English is the initial qualification scope, not a multilingual acceptance claim.

Native output is44100Hz mono float32. FLOAT WAV preserves the native samples rather than silently clipping to PCM16. Any later normalization/assembly stays on the permitted media hosts. Cancellation joins native generation and CUDA synchronization; the backend has no advertised bounded interruption guarantee. A socket close is not physical settlement.

Retain the model's Fish Audio Community License1.0 and NOTICE. This engine is optional and must not restart the working Index8224 instance during current documentary production.

The native source overlay adds an output-only custom artifact carrying the actual reached_limit/frames/wall_ms values. It changes no sampling/audio/stop decision. The engine refuses missing/true limit flags and checks take text/grid frame identity. Exact patch: native_patches/drama3-generation-metadata.json.

Managed instances refuse ordinary /tts and /tts/stream before model work: inference uses the owned Harmony job route. Shared instances without managed-job configuration retain the normal PolyTTS API.
