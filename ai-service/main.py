from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job
from fastapi import FastAPI, HTTPException

from capabilities.registry import CAPABILITIES
from config import settings
from db import close_pool
from logging_service import log_invocation

_redis_pool = None


async def get_redis():
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    return _redis_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_pool()


app = FastAPI(title="GraySync AI Service", lifespan=lifespan)


@app.post("/ai/{capability_name}")
async def run_capability(capability_name: str, payload: dict):
    """One generic route for every capability. New features never need a new route."""
    capability = CAPABILITIES.get(capability_name)
    if not capability:
        raise HTTPException(status_code=404, detail=f"Unknown capability: {capability_name}")

    input_data = capability.input_schema().model_validate(payload)
    organisation_slug = input_data.organisationSlug

    if capability.is_async:
        redis = await get_redis()
        job = await redis.enqueue_job("run_capability_job", capability_name, payload)
        return {"status": "PROCESSING", "jobId": job.job_id}

    try:
        run_result = await capability.run(input_data, organisation_slug)
    except Exception as e:
        await log_invocation(
            organisation_slug, capability_name, payload, None, None, "FAILED", error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

    await log_invocation(
        organisation_slug,
        capability_name,
        payload,
        run_result.result.model_dump(),
        run_result.usage,
        run_result.status,
        context_hash=run_result.context_hash,
        cache_scope=run_result.cache_scope,
        embedding=run_result.embedding,
    )
    return run_result.result.model_dump()


@app.get("/ai/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Frontend polls this while an async capability (e.g. summarize_project) runs."""
    redis = await get_redis()
    job = Job(job_id, redis)
    status = await job.status()

    if str(status) == "JobStatus.complete":
        try:
            result = await job.result(timeout=1)
        except Exception as e:
            # job.result() re-raises whatever exception the worker's job function
            # raised - surface it as detail instead of a bare 500 with no context.
            raise HTTPException(status_code=500, detail=f"Job failed: {e}")
        return {"status": "COMPLETED", "result": result}
    return {"status": str(status).replace("JobStatus.", "").upper()}
