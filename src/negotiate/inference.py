"""Gemini API integration for the negotiation challenge.

Mirrors the production behavior: function-calling with the `negotiate` tool,
native thinking for reasoning, retry on failure.
"""

import asyncio
import logging
from typing import Optional

from google import genai

from .engine import RESOURCE_TYPES

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"


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
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=user_message,
                        config=genai.types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            tools=[negotiate_tool],
                            tool_config=tool_config,
                            temperature=0.7,
                            max_output_tokens=512,
                            thinking_config=genai.types.ThinkingConfig(
                                thinking_budget=1024,
                                include_thoughts=True,
                            ),
                        ),
                    ),
                    timeout=30,
                )

                # Extract thinking
                thinking = ""
                for part in response.candidates[0].content.parts:
                    if getattr(part, "thought", False) and getattr(part, "text", None):
                        thinking += part.text

                for part in response.candidates[0].content.parts:
                    if part.function_call and part.function_call.name == "negotiate":
                        args = part.function_call.args
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
                            "reasoning": thinking[:1000] if thinking else None,
                            "offer": offer,
                        }

                logger.warning("No function call in response, attempt %d", attempt + 1)
            except Exception as e:
                logger.warning("API error on attempt %d: %s", attempt + 1, e)

    return None
