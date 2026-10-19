# Fish Audio Drama3 native PolyTTS engine

Implementation worktree only; runtime/model qualification is separate. This is exact Drama3, not Fish S2. Official source541566932abbaebcb0a40e562a1f846738383f38 identifies35B-total/3B-active parameters. The official Python FP8 runtime requires48GB; this adapter instead targets the community audio.cpp native GGML portccdbe76929e8a77428db6ec3bbbaadeb35acae38 through its actual C ABI0.2.

Selected checkpoint: Jmica/Drama-3-GGUF@17a5b40c5dc2c2be986f5371a7e650108b828e29, drama-3-backbone-8bit-experts-Q3_K_S.gguf,20325196992bytes, SHAb946a8c9806deba559590b98e37ffc9e0957109263e111513734fa93284db16b. Expert banks useQ3_K, other backbone matricesQ8_0, audio tower/codecFP16. Selection is a concrete24GB experiment with8CPU expert layers/2048context; actual memory/latency/quality must be measured. Upstream's24GB guidance was tested on a96GB card, not physical3090 hardware.

The engine loads and frees native model/session handles inside PolyTTS's single GPU executor. No independent application HTTP/model route bypasses Harmony. Exact native library and model bytes are checked before loading. All managed inference retains actual Harmony admission and the durable job receipt. Torch allocator peaks do not include native GGML allocations and are labelled accordingly.

Required environment: POLYTTS_DRAMA3_MODEL, POLYTTS_DRAMA3_MODEL_SHA256, POLYTTS_DRAMA3_MODEL_REVISION, POLYTTS_DRAMA3_LIBRARY, POLYTTS_DRAMA3_LIBRARY_SHA256, POLYTTS_DRAMA3_SOURCE_REVISION. Optional closed session settings: POLYTTS_DRAMA3_CPU_EXPERT_LAYERS (0..40, default8), POLYTTS_DRAMA3_CONTEXT_LENGTH (2048/4096/8192, default2048).

Managed request engine`drama3` uses explicit temperature/topP/topK/maxTokens/seed/textChunkSize controls. maxTokens bounds acoustic frames per native text chunk; it is not seconds or words. Text uses actual native speaker/event tags; there is no invented SSML/durationFactor. For exact continuation/edit history, previousTakes contains reviewed text and ten native VQ codebooks, never server file paths. The resulting native drama3_take JSON is retained byte exact as drama3-take.json. One registered synthetic reference is supported; no private voices are supplied. English is the initial qualification scope, not a multilingual acceptance claim.

Native output is44100Hz mono float32. FLOAT WAV preserves the native samples rather than silently clipping to PCM16. Any later normalization/assembly stays on the permitted media hosts. Cancellation joins native generation and CUDA synchronization; the backend has no advertised bounded interruption guarantee. A socket close is not physical settlement.

Retain the model's Fish Audio Community License1.0 and NOTICE. This engine is optional and must not restart the working Index8224 instance during current documentary production.
