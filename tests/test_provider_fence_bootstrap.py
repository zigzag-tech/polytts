"""CPU bootstrap boundary test; does not import the model-heavy server."""
import ast
import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import httpx
from fastapi import FastAPI


class BootstrapTests(unittest.TestCase):
    def test_owner_bootstrap_actual_asgi_and_executor_adapters(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'server.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'configure_provider_fence')
        app = FastAPI()
        @app.post('/legacy')
        def legacy():
            return {'ok':True}
        base = ThreadPoolExecutor(1)
        jobs = SimpleNamespace(_executor=base)
        environment = {'app':app, 'ThreadPoolExecutor':ThreadPoolExecutor,
            '_gpu_executor':base,'_synthesis_jobs':jobs,'RUNTIME':'pytorch',
            '_synchronize_owned_synthesis':lambda:None,
            '_legacy_executor':None,'_owned_alignment_executor':base,
            '_owned_alignment_jobs':jobs,'_provider_fence':None,
            '_synchronize_owned_alignment':lambda:None}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'server.py','exec'),environment)
        def authorize(token,action):
            if token != 'operator':
                raise PermissionError(action)
        fence = environment['configure_provider_fence'](authorize,{'source':'CPU fixture'})
        self.assertIsNot(jobs._executor,base)
        self.assertEqual(jobs._executor.submit(lambda:17).result(),17)
        with self.assertRaises(RuntimeError):
            environment['configure_provider_fence'](authorize,{'source':'CPU fixture'})
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://fixture') as client:
                hold = await client.post('/_owner/fence/hold',headers={'Authorization':'operator'},json={'holder':'deploy'})
                self.assertTrue(hold.json()['drained'])
                self.assertEqual((await client.post('/legacy')).status_code,503)
                self.assertEqual((await client.post('/_owner/fence/status',json={})).status_code,403)
        asyncio.run(run())
        base.shutdown()
        executor = environment.get('_legacy_executor')
        if executor is not None:
            executor.shutdown()
