import time

import anthropic
from pydantic import BaseModel

from config import settings


class LlmClient:
    """Single place every capability goes through to talk to the LLM.
    Centralizes retries, timing, and usage/cost tracking so features don't
    each reimplement it.
    """

    def __init__(self) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model

    async def call_structured(self, prompt: str, schema: type[BaseModel]) -> tuple[BaseModel, dict]:
        start = time.monotonic()
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            tools=[
                {
                    "name": "return_result",
                    "description": "Return the structured result",
                    "input_schema": schema.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": "return_result"},
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        tool_use = next(b for b in response.content if b.type == "tool_use")
        result = schema.model_validate(tool_use.input)

        usage = {
            "model": self.model,
            "promptTokens": response.usage.input_tokens,
            "completionTokens": response.usage.output_tokens,
            "latencyMs": latency_ms,
        }
        return result, usage

    async def call_text(self, prompt: str) -> tuple[str, dict]:
        start = time.monotonic()
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        text = response.content[0].text
        usage = {
            "model": self.model,
            "promptTokens": response.usage.input_tokens,
            "completionTokens": response.usage.output_tokens,
            "latencyMs": latency_ms,
        }
        return text, usage


llm_client = LlmClient()
