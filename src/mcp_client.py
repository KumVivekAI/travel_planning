"""MCP client wiring via langchain-mcp-adapters (stdio transport)."""

from __future__ import annotations

import asyncio
import sys
from functools import lru_cache
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config import PROJECT_ROOT

WEATHER_SERVER = PROJECT_ROOT / "mcp_servers" / "weather_server.py"
CURRENCY_SERVER = PROJECT_ROOT / "mcp_servers" / "currency_server.py"

PYTHON = sys.executable


def _connection_config() -> dict:
    return {
        "weather": {
            "transport": "stdio",
            "command": PYTHON,
            "args": [str(WEATHER_SERVER)],
        },
        "currency": {
            "transport": "stdio",
            "command": PYTHON,
            "args": [str(CURRENCY_SERVER)],
        },
    }


@lru_cache(maxsize=1)
def _get_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(_connection_config())


async def _load_tools_async():
    client = _get_client()
    return await client.get_tools()


def _run_async(coro):
    return asyncio.run(coro)


@lru_cache(maxsize=1)
def _get_tool_map() -> dict[str, object]:
    tools = _run_async(_load_tools_async())
    return {tool.name: tool for tool in tools}


def invoke_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Invoke a named MCP tool through the LangChain MCP adapter."""
    tool_map = _get_tool_map()
    tool = tool_map.get(tool_name)
    if tool is None:
        available = ", ".join(sorted(tool_map))
        raise RuntimeError(f"MCP tool '{tool_name}' not found. Available: {available}")

    async def _call():
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(arguments)
        return tool.invoke(arguments)

    result = _run_async(_call())
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "\n".join(parts)
    return str(result)


def invoke_weather_forecast_mcp(days: int = 3) -> str:
    return invoke_mcp_tool("get_singapore_weather_forecast", {"days": days})


def invoke_currency_conversion_mcp(amount: float, from_currency: str, to_currency: str) -> str:
    return invoke_mcp_tool(
        "convert_currency_amount",
        {
            "amount": amount,
            "from_currency": from_currency,
            "to_currency": to_currency,
        },
    )
