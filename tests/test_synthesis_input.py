"""Real isolated standard-library CLI; original owner representation retained."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from synthesis_input import synthesis_input_digest

class SynthesisInputTests(unittest.TestCase):
    def cli(self,value):
        return subprocess.run([sys.executable,'-I','-B',str(Path(__file__).resolve().parents[1]/'synthesis_input.py')],input=json.dumps(value,ensure_ascii=False).encode(),capture_output=True)
    def test_unicode_and_insertion_order_match_original_hash(self):
        request={'language':'en','text':'Exact — narration β and 中文.','engine':'voxcpm'}
        original=hashlib.sha256(json.dumps({'request':request,'voice_mode':'native'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.assertEqual(synthesis_input_digest(request,'native'),original)
        result=self.cli({'request':dict(reversed(list(request.items()))),'voice_mode':'native'})
        self.assertEqual(result.returncode,0);self.assertEqual(json.loads(result.stdout),{'provider':'polytts','inputDigest':original})
        self.assertNotEqual(original,hashlib.sha256(json.dumps({'engine':'voxcpm','voiceMode':'native','request':request}).encode()).hexdigest())
    def test_whole_control_submission_and_invalid_mode_refuse_without_echo(self):
        for value in ({'request':{},'voice_mode':'unknown'},{'request':{},'voice_mode':'native','control_token':'PRIVATE-FIXTURE-CONTROL'}):
            result=self.cli(value);self.assertEqual(result.returncode,2);self.assertEqual(result.stdout,b'');self.assertNotIn(b'PRIVATE-FIXTURE',result.stderr)

if __name__=='__main__':unittest.main()
