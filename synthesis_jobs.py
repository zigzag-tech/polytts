"""PolyTTS wire compatibility over Livestack's owned executor-job runtime."""
from livestack_node.owned_executor_jobs import (
    CancellationSignal, JobCapacity, JobConflict, OwnedExecutorJobs,
    OwnedJobCancelled as SynthesisCancelled,
)


class SynthesisJobs(OwnedExecutorJobs):
    def submit(self, job_id, owner_id, token, request_sha256, engine, worker):
        return super().submit(job_id, owner_id, token, request_sha256, engine, worker)

    def audio(self, job_id, token):
        return self.result(job_id, token)

    def _receipt(self, job):
        receipt = super()._receipt(job)
        receipt['engine'] = receipt.pop('kind')
        return receipt
