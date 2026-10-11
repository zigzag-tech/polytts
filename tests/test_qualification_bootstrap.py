"""Actual source bootstrap/worker wiring with explicit CPU gate fixtures."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from fastapi import HTTPException

class QualificationBootstrapTests(unittest.TestCase):
    def namespace(self,enabled,gate):
        source=Path(__file__).resolve().parents[1]/'server.py';tree=ast.parse(source.read_text())
        nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('_lang_name','_authorize_owned_qualification') or isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='_LANG_NAMES' for target in node.targets)]
        namespace={'QUALIFICATION_ONLY':enabled,'app':SimpleNamespace(state=SimpleNamespace(qualification_gate=gate)),'HTTPException':HTTPException,'_synchronize_owned_synthesis':lambda:None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),namespace)
        return namespace['_authorize_owned_qualification']
    def test_qualified_cold_barrier_never_initializes_cuda_but_legacy_barrier_is_retained(self):
        source=Path(__file__).resolve().parents[1]/'server.py';tree=ast.parse(source.read_text())
        node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_synchronize_owned_synthesis')
        calls=[];torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False,is_available=lambda:True,synchronize=lambda:calls.append('barrier')))
        namespace={'QUALIFICATION_ONLY':True}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),namespace)
        with patch.dict('sys.modules',{'torch':torch}):
            namespace['_synchronize_owned_synthesis']();self.assertEqual(calls,[])
            namespace['QUALIFICATION_ONLY']=False;namespace['_synchronize_owned_synthesis']()
        self.assertEqual(calls,['barrier'])

    def test_absent_marker_preserves_default_and_unconfigured_qualified_source_refuses(self):
        self.assertIsNone(self.namespace(False,None)(None,None,None))
        with self.assertRaises(HTTPException) as error:self.namespace(True,None)(None,None,None)
        self.assertEqual(error.exception.status_code,503)
    def test_explicit_english_and_voxcpm_are_required_before_gate(self):
        class Gate:
            def admit(self,*args):raise AssertionError('invalid input reached gate')
        authorize=self.namespace(True,Gate())
        for request in ({'engine':'voxcpm'},{'engine':'voxcpm','language':''},{'engine':'voxcpm','language':'Chinese'},{'engine':'qwen','language':'en'}):
            with self.assertRaises(PermissionError):authorize(SimpleNamespace(request=request),None,None)
    def test_original_physical_worker_runs_inside_exact_source_gate_scope(self):
        events=[]
        class Gate:
            def admit(self,job,authorization):events.append(('admit',job,authorization));return {'fence':7}
            @contextmanager
            def work(self,job,authorization,fence,*,synchronize):events.append(('work',job,authorization,fence));yield;events.append('settled')
        submission=SimpleNamespace(request={'engine':'voxcpm','language':'en'},job_id='fixture-id',owner_id='fixture-owner')
        wrap=self.namespace(True,Gate())(submission,'exact-owner-fingerprint','fixture-permission')
        signal=SimpleNamespace(check=lambda:events.append('checked'))
        self.assertEqual(wrap(lambda signal:events.append('physical-worker') or b'fixture')(signal),b'fixture')
        job={'jobId':'fixture-id','ownerId':'fixture-owner','inputDigest':'exact-owner-fingerprint','modelKind':'voxcpm'}
        self.assertEqual(events,[('admit',job,'fixture-permission'),'checked',('work',job,'fixture-permission',7),'physical-worker','settled'])

if __name__=='__main__':unittest.main()
