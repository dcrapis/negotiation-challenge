"""Model API integration for the negotiation challenge.

Supports:
- Direct Gemini API via `google-genai`
- OpenRouter via an OpenAI-compatible HTTP route

The Gemini path mirrors production behavior: function-calling with the
`negotiate` tool.

Note: thinking_config is intentionally omitted. Combining Gemini's thinking
mode with function calling triggers a known model-side bug where the API
returns FinishReason.MALFORMED_FUNCTION_CALL — the model attempts a tool call
but produces invalid JSON. This is not an SDK or network issue; it is tracked
upstream as googleapis/python-genai#1120 (open since Jul 2025, p2/unfixed).
Disabling thinking eliminates these failures entirely and also reduces
per-call latency by ~3-6x.
"""

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Literal, Optional

from google import genai

from .engine import RESOURCE_TYPES

logger = logging.getLogger(__name__)

ProviderName = Literal["gemini", "openrouter"]

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
OPENROUTER_MODEL = f"google/{GEMINI_MODEL}"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _make_negotiate_tool():
    return genai.types.Tool(
        function_declarations=[
            genai.types.FunctionDeclaration(
                name="negotiate",
                description="Submit your negotiation move.",
                parameters=genai.types.Schema(
                    type="OBJECT",
                    properties={
                        "message": genai.types.Schema(
                            type="STRING",
                            description="Your public message to the other player",
                        ),
                        "action": genai.types.Schema(
                            type="STRING",
                            description="Your action this turn",
                            enum=["propose", "accept", "reject"],
                        ),
                        "offer": genai.types.Schema(
                            type="OBJECT",
                            description="Required when action is 'propose'. The proposed resource split.",
                            properties={
                                "my_share": genai.types.Schema(
                                    type="OBJECT",
                                    description="Resources you keep",
                                    properties={
                                        "books": genai.types.Schema(type="INTEGER"),
                                        "hats": genai.types.Schema(type="INTEGER"),
                                        "balls": genai.types.Schema(type="INTEGER"),
                                    },
                                    required=["books", "hats", "balls"],
                                ),
                                "their_share": genai.types.Schema(
                                    type="OBJECT",
                                    description="Resources the other player gets",
                                    properties={
                                        "books": genai.types.Schema(type="INTEGER"),
                                        "hats": genai.types.Schema(type="INTEGER"),
                                        "balls": genai.types.Schema(type="INTEGER"),
                                    },
                                    required=["books", "hats", "balls"],
                                ),
                            },
                            required=["my_share", "their_share"],
                        ),
                    },
                    required=["message", "action"],
                ),
            )
        ]
    )


_negotiate_tool = None
_tool_config = None


def _get_tool_and_config():
    global _negotiate_tool, _tool_config
    if _negotiate_tool is None:
        _negotiate_tool = _make_negotiate_tool()
        _tool_config = genai.types.ToolConfig(
            function_calling_config=genai.types.FunctionCallingConfig(mode="ANY")
        )
    return _negotiate_tool, _tool_config


def create_client(provider: ProviderName = "gemini") -> object:
    """Create a provider-specific client object.

    OpenRouter uses direct HTTP requests, so there is no persistent SDK client.
    """
    if provider == "gemini":
        return genai.Client()
    if provider == "openrouter":
        return None
    raise ValueError(f"Unsupported provider: {provider}")


async def call_gemini(
    client: genai.Client,
    system_prompt: str,
    user_message: str,
    semaphore: asyncio.Semaphore,
) -> Optional[dict]:
    """One Gemini API call -> parsed negotiate tool response.

    Returns dict with keys: action, message, reasoning, offer (or None on failure).
    """
    negotiate_tool, tool_config = _get_tool_and_config()
    async with semaphore:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_message,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[negotiate_tool],
                tool_config=tool_config,
                temperature=0.7,
                max_output_tokens=512,
            ),
        )

        if not response.candidates:
            block = response.prompt_feedback
            logger.warning(
                "No candidates: prompt blocked (%s)",
                block.block_reason if block else "unknown",
            )
            return None

        candidate = response.candidates[0]
        fc_list = response.function_calls
        if not fc_list:
            logger.warning(
                "No function call in response (finish_reason=%s)",
                candidate.finish_reason,
            )
            return None

        for fc in fc_list:
            if fc.name == "negotiate" and fc.args:
                args = fc.args
                action = str(args.get("action", "reject"))
                message = str(args.get("message", ""))[:500]
                offer = None

                if action == "propose" and args.get("offer"):
                    raw = args["offer"]
                    my_share = raw.get("my_share", {})
                    their_share = raw.get("their_share", {})
                    offer = {
                        "my_share": {r: int(my_share.get(r, 0)) for r in RESOURCE_TYPES},
                        "their_share": {r: int(their_share.get(r, 0)) for r in RESOURCE_TYPES},
                    }

                return {
                    "action": action,
                    "message": message,
                    "reasoning": None,
                    "offer": offer,
                }

        logger.warning(
            "negotiate function call not found among %d calls", len(fc_list)
        )
        return None


def _make_openrouter_negotiate_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "negotiate",
            "description": "Submit your negotiation move.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Your public message to the other player",
                    },
                    "action": {
                        "type": "string",
                        "description": "Your action this turn",
                        "enum": ["propose", "accept", "reject"],
                    },
                    "offer": {
                        "type": "object",
                        "description": "Required when action is 'propose'. The proposed resource split.",
                        "properties": {
                            "my_share": {
                                "type": "object",
                                "description": "Resources you keep",
                                "properties": {r: {"type": "integer"} for r in RESOURCE_TYPES},
                                "required": RESOURCE_TYPES,
                            },
                            "their_share": {
                                "type": "object",
                                "description": "Resources the other player gets",
                                "properties": {r: {"type": "integer"} for r in RESOURCE_TYPES},
                                "required": RESOURCE_TYPES,
                            },
                        },
                        "required": ["my_share", "their_share"],
                    },
                },
                "required": ["message", "action"],
            },
        },
    }


def _openrouter_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY (or OPENAI_API_KEY) for OpenRouter.")
    return key


def _sync_call_openrouter(system_prompt: str, user_message: str) -> dict:
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "tools": [_make_openrouter_negotiate_tool()],
        "tool_choice": {"type": "function", "function": {"name": "negotiate"}},
        "temperature": 0.7,
        "max_tokens": 512,
    }

    headers = {
        "Authorization": f"Bearer {_openrouter_api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "negotiation-challenge",
    }

    last_error: Exception | None = None
    for attempt in range(3):
        req = urllib.request.Request(
            OPENROUTER_API_URL,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as err:
            body = err.read().decode(errors="replace")
            if err.code in {429, 502, 503, 504} and attempt < 2:
                time.sleep(1.0 * (2**attempt))
                continue
            raise RuntimeError(f"OpenRouter HTTP {err.code}: {body}") from err
        except Exception as err:  # pragma: no cover - network failure path
            last_error = err
            if attempt < 2:
                time.sleep(1.0 * (2**attempt))
                continue
            raise
    raise RuntimeError(f"OpenRouter request failed: {last_error}")


async def call_openrouter(
    _client: object,
    system_prompt: str,
    user_message: str,
    semaphore: asyncio.Semaphore,
) -> Optional[dict]:
    """One OpenRouter API call -> parsed negotiate tool response."""
    async with semaphore:
        response = await asyncio.to_thread(_sync_call_openrouter, system_prompt, user_message)

    choices = response.get("choices", [])
    if not choices:
        logger.warning("No choices in OpenRouter response")
        return None

    message_obj = choices[0].get("message", {})
    tool_calls = message_obj.get("tool_calls") or []
    if not tool_calls:
        finish_reason = choices[0].get("finish_reason")
        logger.warning("No tool call in OpenRouter response (finish_reason=%s)", finish_reason)
        return None

    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        if function.get("name") != "negotiate":
            continue
        raw_args = function.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except Exception:
            logger.warning("Malformed function arguments from OpenRouter: %r", raw_args)
            return None

        action = str(args.get("action", "reject"))
        message = str(args.get("message", ""))[:500]
        offer = None

        if action == "propose" and args.get("offer"):
            raw = args["offer"]
            my_share = raw.get("my_share", {})
            their_share = raw.get("their_share", {})
            offer = {
                "my_share": {r: int(my_share.get(r, 0)) for r in RESOURCE_TYPES},
                "their_share": {r: int(their_share.get(r, 0)) for r in RESOURCE_TYPES},
            }

        return {
            "action": action,
            "message": message,
            "reasoning": message_obj.get("reasoning"),
            "offer": offer,
        }

    logger.warning("negotiate tool call not found in OpenRouter response")
    return None


async def call_model(
    client: object,
    system_prompt: str,
    user_message: str,
    semaphore: asyncio.Semaphore,
    provider: ProviderName = "gemini",
) -> Optional[dict]:
    """Dispatch one negotiation turn to the selected provider."""
    if provider == "gemini":
        return await call_gemini(client, system_prompt, user_message, semaphore)
    if provider == "openrouter":
        return await call_openrouter(client, system_prompt, user_message, semaphore)
    raise ValueError(f"Unsupported provider: {provider}")
