import time

from google import genai
from google.genai import types
from pydantic import BaseModel

from config import settings


class LlmClient:
    """Single place every capability goes through to talk to the LLM.
    Centralizes retries, timing, and usage/cost tracking so features don't
    each reimplement it.
    """

    def __init__(self) -> None:
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.llm_model

    async def call_structured(self, prompt: str, schema: type[BaseModel]) -> tuple[BaseModel, dict]:
        start = time.monotonic()
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        result = schema.model_validate_json(response.text)

        usage = {
            "model": self.model,
            "promptTokens": response.usage_metadata.prompt_token_count,
            "completionTokens": response.usage_metadata.candidates_token_count,
            "latencyMs": latency_ms,
        }
        return result, usage

    async def call_text(self, prompt: str) -> tuple[str, dict]:
        start = time.monotonic()
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        usage = {
            "model": self.model,
            "promptTokens": response.usage_metadata.prompt_token_count,
            "completionTokens": response.usage_metadata.candidates_token_count,
            "latencyMs": latency_ms,
        }
        return response.text, usage


llm_client = LlmClient()
