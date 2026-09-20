"""LangChain-based travel assistant combining RAG and MCP tools."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import DEFAULT_LLM_MODEL
from src.currency_parser import parse_currency_request
from src.mcp_client import invoke_currency_conversion_mcp, invoke_weather_forecast_mcp
from src.mcp_tools.currency import convert_currency, format_currency_for_llm
from src.mcp_tools.weather import format_weather_for_llm, get_weather_forecast
from src.prompts import COMBINED_PROMPT, SYSTEM_PROMPT
from src.rag.retriever import retrieve_context


@dataclass
class AssistantResponse:
    answer: str
    kb_sources: list[dict] = field(default_factory=list)
    mcp_tools_used: list[str] = field(default_factory=list)
    used_rag: bool = False


def _get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it in a .env file or your environment."
        )
    return ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0.3, api_key=api_key)


def _needs_weather(question: str) -> bool:
    keywords = ["weather", "rain", "forecast", "temperature", "indoor", "outdoor", "umbrella", "storm"]
    return any(k in question.lower() for k in keywords)


def _needs_currency(question: str) -> bool:
    keywords = ["convert", "budget", "currency", "inr", "sgd", "usd", "rupee", "dollar", "exchange"]
    return any(k in question.lower() for k in keywords)


def _extract_forecast_days(question: str) -> int:
    match = re.search(r"(\d+)\s*[- ]?day", question.lower())
    if match:
        return min(max(int(match.group(1)), 1), 7)
    if "week" in question.lower() or "next 7" in question.lower():
        return 7
    if "tomorrow" in question.lower():
        return 1
    return 3


def _format_history(messages: list[dict]) -> str:
    if not messages:
        return "No prior conversation."
    lines = []
    for msg in messages[-6:]:
        role = msg.get("role", "user").capitalize()
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _weather_via_mcp(days: int) -> tuple[str, str]:
    """Call weather through MCP; fall back to direct API on transport failure."""
    try:
        return invoke_weather_forecast_mcp(days=days), "MCP Weather Tool (stdio → Open-Meteo)"
    except Exception as exc:
        result = get_weather_forecast(days=days)
        formatted = format_weather_for_llm(result)
        if result.get("success"):
            formatted += f"\n(MCP transport unavailable: {exc}; used direct service fallback.)"
        return formatted, "MCP Weather Tool (fallback direct API)"


def _currency_via_mcp(amount: float, from_currency: str, to_currency: str) -> tuple[str, str]:
    try:
        return (
            invoke_currency_conversion_mcp(amount, from_currency, to_currency),
            "MCP Currency Tool (stdio → Frankfurter)",
        )
    except Exception as exc:
        result = convert_currency(amount, from_currency, to_currency)
        formatted = format_currency_for_llm(result)
        if result.get("success"):
            formatted += f"\n(MCP transport unavailable: {exc}; used direct service fallback.)"
        return formatted, "MCP Currency Tool (fallback direct API)"


def ask(question: str, history: list[dict] | None = None) -> AssistantResponse:
    """Main entry point: RAG retrieval + MCP tools + LLM synthesis."""
    history = history or []
    mcp_context_parts: list[str] = []
    mcp_tools_used: list[str] = []

    kb_context, kb_sources = retrieve_context(question)
    used_rag = bool(kb_context)

    if _needs_weather(question):
        days = _extract_forecast_days(question)
        weather_text, tool_label = _weather_via_mcp(days)
        mcp_context_parts.append(weather_text)
        mcp_tools_used.append(tool_label)

    if _needs_currency(question):
        conversion = parse_currency_request(question)
        if conversion:
            amount, from_cur, to_cur = conversion
            currency_text, tool_label = _currency_via_mcp(amount, from_cur, to_cur)
            mcp_context_parts.append(currency_text)
            mcp_tools_used.append(tool_label)
        else:
            mcp_context_parts.append(
                "[MCP Currency Tool] Could not parse amount/currencies from the question. "
                "Please specify amount and currency codes (e.g. Convert INR 50,000 to SGD)."
            )

    mcp_context = (
        "\n\n".join(mcp_context_parts)
        if mcp_context_parts
        else "No MCP tools were needed for this query."
    )

    prompt = COMBINED_PROMPT.format(
        kb_context=kb_context or "No relevant knowledge base content found.",
        mcp_context=mcp_context,
        history=_format_history(history),
        question=question,
    )

    llm = _get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    return AssistantResponse(
        answer=response.content,
        kb_sources=kb_sources,
        mcp_tools_used=mcp_tools_used,
        used_rag=used_rag,
    )
