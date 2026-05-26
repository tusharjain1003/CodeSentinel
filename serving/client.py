from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from config import settings

_client = AsyncOpenAI(
    base_url=settings.vllm_base_url,
    api_key=settings.openai_api_key or "not-needed",
)
_openai_client = AsyncOpenAI(api_key=settings.openai_api_key or "not-needed")
_groq_client = (
    AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.groq_api_key,
    )
    if settings.groq_api_key
    else None
)


def _select_client(model: str | None = None) -> AsyncOpenAI:
    model_name = model or settings.vllm_model_name
    if model_name == settings.gpt4o_model_name or model_name.startswith("gpt-"):
        return _openai_client
    if model_name == settings.groq_model_name or "groq" in str(model_name).lower():
        if _groq_client is None:
            msg = "GROQ_API_KEY not configured in .env"
            raise ValueError(msg)
        return _groq_client
    return _client


async def complete(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
    response_format: dict[str, Any] | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model or settings.vllm_model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    response = await _select_client(model).chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


async def complete_with_function_call(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    model: str | None = None,
) -> dict[str, Any]:
    response = await _select_client(model).chat.completions.create(
        model=model or settings.vllm_model_name,
        messages=messages,
        tools=tools,
        tool_choice="required",
        temperature=0.1,
    )
    tool_calls = response.choices[0].message.tool_calls or []
    if not tool_calls:
        return json.loads(response.choices[0].message.content or "{}")
    return json.loads(tool_calls[0].function.arguments)
