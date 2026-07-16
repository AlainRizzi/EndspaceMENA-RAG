from arq.connections import RedisSettings

from capabilities.registry import CAPABILITIES
from config import settings
from logging_service import log_invocation


async def run_capability_job(ctx, capability_name: str, payload: dict) -> dict:
    """Executed by the arq worker process for any capability with is_async = True."""
    capability = CAPABILITIES[capability_name]
    subdomain_name = payload.get("subdomainName", "unknown")
    organisation_slug = payload.get("organisationSlug")  # may be None
    input_data = capability.input_schema().model_validate(payload)

    try:
        run_result = await capability.run(input_data, subdomain_name, organisation_slug)
    except Exception as e:
        await log_invocation(
            subdomain_name, organisation_slug, capability_name, payload, None, None, "FAILED", error_message=str(e)
        )
        raise

    await log_invocation(
        subdomain_name,
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


class WorkerSettings:
    functions = [run_capability_job]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
