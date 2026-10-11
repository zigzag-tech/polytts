"""Actual source bootstrap/worker wiring with explicit CPU gate fixtures."""
import ast
import copy
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
            def work(self,job,authorization,fence,*,synchronize):events.append(('work',job,authorization,fence));proof={'fixture':'mock lease','physicalBarrier':'pending'};yield proof;proof['physicalBarrier']='settled';events.append('settled')
        submission=SimpleNamespace(request={'engine':'voxcpm','language':'en'},job_id='fixture-id',owner_id='fixture-owner')
        wrap=self.namespace(True,Gate())(submission,'exact-owner-fingerprint','fixture-permission')
        recorded=[]
        signal=SimpleNamespace(check=lambda:events.append('checked'),record_qualification_evidence=lambda proof:recorded.append(copy.deepcopy(proof)))
        self.assertEqual(wrap(lambda signal:events.append('physical-worker') or b'fixture')(signal),b'fixture')
        job={'jobId':'fixture-id','ownerId':'fixture-owner','inputDigest':'exact-owner-fingerprint','modelKind':'voxcpm'}
        self.assertEqual(events,[('admit',job,'fixture-permission'),'checked',('work',job,'fixture-permission',7),'physical-worker','settled'])
        self.assertEqual(len(recorded),2);self.assertEqual(recorded[0]['residency']['physicalBarrier'],'pending');self.assertEqual(recorded[1]['residency']['physicalBarrier'],'settled')
        self.assertEqual(recorded[0]['reservation'],{'fence':7})


    def test_capability_declares_actual_qualified_binding_and_legacy_is_unchanged(self):
        source=Path(__file__).resolve().parents[1]/'server.py';tree=ast.parse(source.read_text())
        node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_synthesis_capabilities')
        namespace={'QUALIFICATION_ONLY':False,'app':SimpleNamespace(state=SimpleNamespace(qualification_gate=None)),'HTTPException':HTTPException,
            '_MANAGER_PATH':False,'manager':None,'_ENGINES':{},'RUNTIME':'pytorch','DEVICE':'cpu','_ALIGN_MODEL_LABEL':'CPU fixture'}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),namespace)
        self.assertNotIn('qualification_evidence',namespace['_synthesis_capabilities']())
        namespace['QUALIFICATION_ONLY']=True
        with self.assertRaises(HTTPException):namespace['_synthesis_capabilities']()
        binding={'nodeId':'explicit CPU fixture node'}
        namespace['app'].state.qualification_gate=SimpleNamespace(binding=binding)
        marker=namespace['_synthesis_capabilities']()['qualification_evidence']
        self.assertEqual(marker,{'version':1,'required':True,'binding':binding})
        marker['binding']['nodeId']='changed snapshot'
        self.assertEqual(binding['nodeId'],'explicit CPU fixture node')

if __name__=='__main__':unittest.main()
