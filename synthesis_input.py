"""Canonical owned TTS input digest; standard library only, no model preparation."""
import hashlib
import json
import sys


def synthesis_input_digest(request, voice_mode):
    # Preserve the original owner API's exact Unicode/ordering representation.
    return hashlib.sha256(json.dumps({'request':request,'voice_mode':voice_mode},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    try:
        raw=sys.stdin.buffer.read(65537)
        if len(raw)>65536:raise ValueError('input bound')
        value=json.loads(raw)
        if not isinstance(value,dict) or set(value)!={'request','voice_mode'} or not isinstance(value['request'],dict) or value['voice_mode'] not in ('native','registered'):
            raise ValueError('closed owned TTS input required')
        print(json.dumps({'provider':'polytts','inputDigest':synthesis_input_digest(value['request'],value['voice_mode'])},separators=(',',':')))
    except (ValueError,TypeError,UnicodeError):
        print('canonical TTS input digest refused',file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
