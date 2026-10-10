"""Optional owned-job API; the historical /tts endpoints are unchanged."""
import hashlib
import json
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from synthesis_jobs import JobCapacity, JobConflict


class SynthesisSubmission(BaseModel):
    job_id: str
    owner_id: str
    control_token: str = Field(min_length=32)
    voice_mode: Literal["registered", "native"] = "registered"
    request: dict


class SynthesisCancellation(BaseModel):
    owner_id: str


def create_synthesis_router(jobs, capabilities, prepare):
    """prepare validates without GPU work, returning (engine, worker)."""
    router = APIRouter()

    def token(authorization):
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "owned synthesis control token required")
        return authorization[7:]

    def call(fn):
        try:
            return fn()
        except KeyError:
            raise HTTPException(404, "unknown owned synthesis job")
        except JobConflict as exc:
            raise HTTPException(409, str(exc))
        except JobCapacity as exc:
            raise HTTPException(503, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @router.get("/tts/capabilities")
    def job_capabilities():
        return {"job_protocol_version": 1, "server_instance_id": jobs.instance_id, **capabilities()}

    @router.post("/tts/jobs")
    def submit_owned_job(submission: SynthesisSubmission):
        # The hash excludes the control secret and includes the exact caller
        # request/profile. No semantic rewriting or repair happens here.
        fingerprint = hashlib.sha256(json.dumps({"request": submission.request,
            "voice_mode": submission.voice_mode}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        engine, worker = call(lambda: prepare(submission.request, submission.voice_mode))
        return call(lambda: jobs.submit(submission.job_id, submission.owner_id,
            submission.control_token, fingerprint, engine, worker))

    @router.get("/tts/jobs/{job_id}")
    def owned_job_status(job_id: str, authorization: str | None = Header(default=None)):
        return call(lambda: jobs.status(job_id, token(authorization)))

    @router.post("/tts/jobs/{job_id}/cancel")
    def cancel_owned_job(job_id: str, request: SynthesisCancellation, authorization: str | None = Header(default=None)):
        return call(lambda: jobs.cancel(job_id, token(authorization), request.owner_id))

    @router.get("/tts/jobs/{job_id}/audio")
    def owned_job_audio(job_id: str, authorization: str | None = Header(default=None)):
        audio = call(lambda: jobs.audio(job_id, token(authorization)))
        return Response(audio, media_type="audio/wav")

    return router
