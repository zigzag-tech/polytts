# IndexTTS 2.5 through PolyTTS and Harmony

This engine uses the official `indextts.infer_v2_5.IndexTTS2` implementation. Model ID: IndexTeam/IndexTTS-2.5; source d9e41aac89fd00b3d71497fddb287b7f24613712, checkpoint c39ce5ba981572cb187443877ff559dfb246ce63. Retain upstream Bilibili model license notices.

An isolated CUDA instance enables `POLYTTS_ENABLED_ENGINES=indextts25`, `POLYTTS_DEFAULT_ENGINE=indextts25`, `POLYTTS_DEFAULT_RESIDENCY=UNPINNED`, `POLYTTS_NO_PRELOAD=1`, explicit `POLYTTS_DEVICE_ID`, pinned `POLYTTS_INDEXTTS25_SOURCE`, `POLYTTS_INDEXTTS25_MODELS`, and `POLYTTS_INDEXTTS25_REVISION`. `POLYTTS_MANAGED_JOBS_CONFIG` selects the private installed credential, allowed owner prefixes, model inventory, physical CUDA UUID and durable state directory.

POST /voices registers an actual engine-specific reference. Use its returned voice ID; an existing VoxCPM voice ID is not an IndexTTS registration.

## Managed interface

POST /v1/tts/jobs accepts a closed request with required engine `indextts25`, jobId, ownerId, text, language, voiceId, referenceSha256 and controls. Index controls are durationFactor (0.5..2), emotionAlpha (0..1), optional eight-entry emotionVector (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm), intervalSilenceMs (0..2000), maxTextTokensPerSegment (20..200) and uint32 seed. All controls are required except the vector.

The server calls the SAME actual PolyTTS engine registered in the Harmony residency manager. It obtains an actual local lease and exact device admission before native synthesis. Protocol version2 capabilities expose the provider plus per-engine identity; status/cancel/audio endpoints retain the existing durable receipt contract. No external fleet authorization is claimed from the local broker.

Native output is mono22050Hz PCM16. Duration factor changes the acoustic length regulator, not exact seconds. Internal segment silence is not a timeline pause. Free-text emotion/SSML are unsupported in this runtime. Cancellation waits for native inference and CUDA synchronization; HTTP timeout is unknown. Generation warnings indicating truncation refuse success. Editorial acceptance is separate from physical/source qualification.

The ordinary /tts API exposes the same controls under snake_case names. No sentence chunking default is applied to IndexTTS.

Deployment and actual provider qualification are recorded separately; this document alone claims source implementation only.
