import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from synthesis_api import create_synthesis_router
from synthesis_jobs import SynthesisJobs


class OwnedSynthesisApiTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.jobs = SynthesisJobs(self.executor, lambda: None)
        self.prepared = []
        def prepare(request, mode):
            self.prepared.append((request, mode))
            return 'fake', lambda signal: b'RIFFfake-WAV'
        app = FastAPI()
        app.include_router(create_synthesis_router(self.jobs, lambda: {
            'engines': {'fake': {'physical_cancellation': {'supported': True}}},
            'voice_registration_gpu': False,
        }, prepare))
        self.client = AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver')
        self.token = 'random-client-owned-secret-' * 3
        self.headers = {'Authorization': f'Bearer {self.token}'}
        self.submission = {'job_id': str(uuid.uuid4()), 'owner_id': 'owner-a',
            'control_token': self.token, 'voice_mode': 'native',
            'request': {'text': 'Exact narration', 'engine': 'fake'}}

    async def asyncTearDown(self):
        await self.client.aclose()
        self.executor.shutdown(wait=True)

    async def test_capability_discovery_has_no_inference_and_job_receipts_do_not_expose_control_tokens(self):
        response = await self.client.get('/tts/capabilities')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['job_protocol_version'], 1)
        self.assertEqual(self.prepared, [])
        accepted = await self.client.post('/tts/jobs', json=self.submission)
        self.assertEqual(accepted.status_code, 200)
        self.executor.submit(lambda: None).result(timeout=2)
        path = '/tts/jobs/' + self.submission['job_id']
        status = await self.client.get(path, headers=self.headers)
        self.assertEqual(status.json()['state'], 'completed')
        self.assertTrue(status.json()['physical_settled'])
        self.assertNotIn(self.token, status.text)
        self.assertEqual((await self.client.get(path + '/audio', headers=self.headers)).content, b'RIFFfake-WAV')
        self.assertEqual((await self.client.get(path, headers={'Authorization': 'Bearer ' + 'wrong-token-' * 4})).status_code, 404)
        self.assertEqual((await self.client.get(path)).status_code, 401)

    async def test_lost_acceptance_cancel_tombstone_rejects_late_submit_without_running(self):
        path = '/tts/jobs/' + self.submission['job_id']
        cancelled = await self.client.post(path + '/cancel', json={'owner_id': 'owner-a'}, headers=self.headers)
        self.assertEqual(cancelled.json()['state'], 'cancelled')
        self.assertTrue(cancelled.json()['physical_settled'])
        accepted = await self.client.post('/tts/jobs', json=self.submission)
        self.assertEqual(accepted.json()['state'], 'cancelled')
        self.assertEqual((await self.client.get(path + '/audio', headers=self.headers)).status_code, 409)

    async def test_retry_cannot_rebind_owned_job_to_changed_text(self):
        self.assertEqual((await self.client.post('/tts/jobs', json=self.submission)).status_code, 200)
        changed = {**self.submission, 'request': {'text': 'Changed narration', 'engine': 'fake'}}
        self.assertEqual((await self.client.post('/tts/jobs', json=changed)).status_code, 409)


if __name__ == '__main__':
    unittest.main()
