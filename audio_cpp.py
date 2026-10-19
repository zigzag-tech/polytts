"""The upstream audio.cpp C ABI, hosted on PolyTTS's owned GPU executor.

No HTTP sidecar or application inference path is involved. Native synchronous
calls finish before the borrowed output is copied and the result is freed.
"""
import ctypes as C
from pathlib import Path

import numpy as np


class ModelConfig(C.Structure):
    _fields_ = [(name, C.c_char_p) for name in
                ('family_hint', 'config_id', 'weight_id', 'model_spec_override')]


class BackendConfig(C.Structure):
    _fields_ = [('backend', C.c_char_p), ('device', C.c_int), ('threads', C.c_int)]


class NativeAudioSession:
    def __init__(self, library, model, *, family, session_options, threads=4, task="clon"):
        self.lib = C.CDLL(str(Path(library).resolve(strict=True)))
        self.registry = C.c_void_p()
        self.model = C.c_void_p()
        self.session = C.c_void_p()
        pointer = C.c_void_p
        out = C.POINTER(pointer)
        self._functions = {
            'abi_version': (C.c_uint32, []),
            'last_error': (C.c_char_p, []),
            'options_create': (pointer, []),
            'options_set': (C.c_int, [pointer, C.c_char_p, C.c_char_p]),
            'options_free': (None, [pointer]),
            'registry_create': (C.c_int, [C.c_char_p, out]),
            'registry_free': (None, [pointer]),
            'model_load': (C.c_int, [pointer, C.c_char_p, C.POINTER(ModelConfig), pointer, out]),
            'model_free': (None, [pointer]),
            'model_family': (C.c_char_p, [pointer]),
            'model_supports': (C.c_int, [pointer, C.c_char_p, C.c_char_p]),
            'session_create': (C.c_int, [pointer, C.c_char_p, C.c_char_p, C.POINTER(BackendConfig), pointer, out]),
            'session_free': (None, [pointer]),
            'session_run': (C.c_int, [pointer, pointer, out]),
            'request_create': (pointer, []),
            'request_free': (None, [pointer]),
            'request_set_text': (C.c_int, [pointer, C.c_char_p, C.c_char_p]),
            'request_set_text_language': (C.c_int, [pointer, C.c_char_p]),
            'request_set_voice_audio': (C.c_int, [pointer, C.POINTER(C.c_float), C.c_size_t, C.c_int, C.c_int]),
            'request_set_option': (C.c_int, [pointer, C.c_char_p, C.c_char_p]),
            'result_audio': (C.c_int, [pointer, C.POINTER(C.POINTER(C.c_float)), C.POINTER(C.c_size_t), C.POINTER(C.c_int), C.POINTER(C.c_int)]),
            'result_free': (None, [pointer]),
            'result_artifact_count': (C.c_size_t, [pointer]),
            'result_artifact': (C.c_int, [pointer, C.c_size_t, C.POINTER(C.c_int), C.POINTER(C.c_char_p), C.POINTER(pointer), C.POINTER(C.c_size_t)]),
        }
        for name, (result, arguments) in self._functions.items():
            function = getattr(self.lib, 'audiocpp_' + name)
            function.restype, function.argtypes = result, arguments
        if self.lib.audiocpp_abi_version() >> 8 != 2:
            raise RuntimeError('Pinned audio.cpp ABI0.2 required')
        options = self.lib.audiocpp_options_create()
        try:
            self.check(self.lib.audiocpp_registry_create(None, C.byref(self.registry)))
            config = ModelConfig(family.encode(), None, None, b'embedded')
            self.check(self.lib.audiocpp_model_load(self.registry, str(Path(model).resolve(strict=True)).encode(), C.byref(config), None, C.byref(self.model)))
            if self.lib.audiocpp_model_family(self.model).decode() != family:
                raise RuntimeError('The actual native model family differs; substitution refused')
            if task not in {"tts", "clon"} or self.lib.audiocpp_model_supports(self.model, task.encode(), b'offline') != 1:
                raise RuntimeError('Explicit native offline speech task unsupported')
            for key, value in session_options.items():
                self.check(self.lib.audiocpp_options_set(options, key.encode(), str(value).encode()))
            backend = BackendConfig(b'cuda', 0, threads)
            self.check(self.lib.audiocpp_session_create(self.model, task.encode(), b'offline', C.byref(backend), options, C.byref(self.session)))
        except BaseException:
            self.close()
            raise
        finally:
            self.lib.audiocpp_options_free(options)

    def check(self, code):
        if code:
            message = self.lib.audiocpp_last_error()
            raise RuntimeError('Native audio.cpp refused: ' + (message.decode(errors='replace')[:2000] if message else str(code)))

    def run(self, text, language, reference, reference_rate, options):
        request = self.lib.audiocpp_request_create()
        result = C.c_void_p()
        try:
            self.check(self.lib.audiocpp_request_set_text(request, text.encode(), None))
            self.check(self.lib.audiocpp_request_set_text_language(request, language.encode()))
            if reference is not None:
                reference = np.ascontiguousarray(reference, dtype=np.float32)
                if reference.ndim != 1 or not len(reference) or not np.isfinite(reference).all():
                    raise ValueError('Explicit finite mono reference required')
                self.check(self.lib.audiocpp_request_set_voice_audio(request, reference.ctypes.data_as(C.POINTER(C.c_float)), len(reference), reference_rate, 1))
            elif reference_rate is not None or "reference_text" in options:
                raise ValueError('Unconditioned mode must contain no reference data/transcript')
            for key, value in options.items():
                self.check(self.lib.audiocpp_request_set_option(request, key.encode(), str(value).encode()))
            self.check(self.lib.audiocpp_session_run(self.session, request, C.byref(result)))
            samples = C.POINTER(C.c_float)()
            frames, rate, channels = C.c_size_t(), C.c_int(), C.c_int()
            self.check(self.lib.audiocpp_result_audio(result, C.byref(samples), C.byref(frames), C.byref(rate), C.byref(channels)))
            if rate.value != 44100 or channels.value != 1 or not 0 < frames.value <= 44100 * 600:
                raise RuntimeError('Native Drama3 mono44100Hz output bound refused')
            audio = np.ctypeslib.as_array(samples, shape=(frames.value,)).copy()
            if not np.isfinite(audio).all():
                raise RuntimeError('Nonfinite native Drama3 output refused')
            artifacts = []
            count = self.lib.audiocpp_result_artifact_count(result)
            if count > 8:
                raise RuntimeError('Native artifact count bound exceeded')
            for index in range(count):
                kind, name, payload, size = C.c_int(), C.c_char_p(), C.c_void_p(), C.c_size_t()
                self.check(self.lib.audiocpp_result_artifact(result, index, C.byref(kind), C.byref(name), C.byref(payload), C.byref(size)))
                if size.value > 4*1024**2:
                    raise RuntimeError('Native continuation artifact bound exceeded')
                artifacts.append({'kind': kind.value, 'id': name.value.decode(), 'payload': C.string_at(payload, size.value)})
            return audio, rate.value, artifacts
        finally:
            self.lib.audiocpp_result_free(result)
            self.lib.audiocpp_request_free(request)

    def close(self):
        for name in ('session', 'model', 'registry'):
            handle = getattr(self, name, None)
            if handle:
                getattr(self.lib, 'audiocpp_' + name + '_free')(handle)
                setattr(self, name, C.c_void_p())
