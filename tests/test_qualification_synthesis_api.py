"""Real owned-job ASGI routing; canonical permission/worker gate are CPU fixtures."""
import hashlib
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from fastapi import FastAPI
from httpx import AsyncClient,ASGITransport
from synthesis_api import create_synthesis_router
from synthesis_jobs import SynthesisJobs

class QualificationSynthesisApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.executor=ThreadPoolExecutor(max_workers=1);self.jobs=SynthesisJobs(self.executor,lambda:None)
        self.events=[];self.revoked=False;self.token='fixture-control-secret-'*3
        self.permission='Bearer zzops_'+'d'*64
        self.submission={'job_id':str(uuid4()),'owner_id':'fixture-owner','control_token':self.token,'voice_mode':'native','request':{'text':'Exact English fixture.','language':'en','engine':'voxcpm'}}
        def authorize(submission,digest,authorization):
            if authorization!=self.permission or self.revoked:raise PermissionError('mock denied')
            expected=hashlib.sha256(json.dumps({'request':self.submission['request'],'voice_mode':'native'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
            self.assertEqual(digest,expected);self.assertEqual(submission.job_id,self.submission['job_id'])
            self.events.append('permission')
            def wrap(worker):
                def physical(signal):
                    self.events.append('physical-scope');return worker(signal)
                return physical
            return wrap
        def prepare(request,mode):
            self.events.append('prepare');return 'cpu-fixture',lambda signal:self.events.append('worker') or b'RIFFfixture'
        app=FastAPI();app.include_router(create_synthesis_router(self.jobs,lambda:{},prepare,authorize=authorize))
        self.client=AsyncClient(transport=ASGITransport(app=app),base_url='http://fixture')
    async def asyncTearDown(self):await self.client.aclose();self.executor.shutdown(wait=True)
    async def test_missing_forged_and_job_control_credentials_refuse_before_prepare(self):
        for token in ('','Bearer forged','Bearer '+self.token):
            response=await self.client.post('/tts/jobs',json=self.submission,headers={'X-Provider-Qualification-Authorization':token})
            self.assertEqual(response.status_code,403)
        self.assertEqual(self.events,[])
    async def test_exact_digest_authorization_precedes_prepare_and_actual_worker_scope(self):
        response=await self.client.post('/tts/jobs',json=self.submission,headers={'X-Provider-Qualification-Authorization':self.permission})
        self.assertEqual(response.status_code,200)
        self.executor.submit(lambda:None).result(timeout=2)
        self.assertEqual(self.events,['permission','prepare','physical-scope','worker'])
        self.assertNotIn(self.permission,response.text)
    async def test_accepted_settlement_controls_remain_usable_after_permission_revocation(self):
        headers={'X-Provider-Qualification-Authorization':self.permission}
        self.assertEqual((await self.client.post('/tts/jobs',json=self.submission,headers=headers)).status_code,200)
        self.executor.submit(lambda:None).result(timeout=2);self.revoked=True
        self.assertEqual((await self.client.post('/tts/jobs',json=self.submission,headers=headers)).status_code,403)
        path='/tts/jobs/'+self.submission['job_id'];control={'Authorization':'Bearer '+self.token}
        self.assertEqual((await self.client.get(path,headers=control)).json()['state'],'completed')
        self.assertEqual((await self.client.get(path+'/audio',headers=control)).content,b'RIFFfixture')

if __name__=='__main__':unittest.main()
