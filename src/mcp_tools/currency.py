"""Currency conversion tool logic (used by MCP server and LangChain agent)."""

from __future__ import annotations

import httpx

FRANKFURTER_API_URL = "https://api.frankfurter.dev/v1/latest"


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert currency using the Frankfurter API (free, no API key required)."""
    from_currency = from_currency.upper().strip()
    to_currency = to_currency.upper().strip()

    if amount <= 0:
        return {
            "success": False,
            "error": "Amount must be greater than zero.",
            "source": "MCP Currency Tool (Frankfurter API)",
        }

    params = {"amount": amount, "from": from_currency, "to": to_currency}

    try:
        response = httpx.get(FRANKFURTER_API_URL, params=params, timeout=15.0)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        return {
            "success": False,
            "error": f"Currency service unavailable: {exc}",
            "source": "MCP Currency Tool (Frankfurter API)",
        }

    rates = data.get("rates", {})
    converted = rates.get(to_currency)

    if converted is None:
        return {
            "success": False,
            "error": f"Could not convert {from_currency} to {to_currency}.",
            "source": "MCP Currency Tool (Frankfurter API)",
        }

    return {
        "success": True,
        "amount": amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "converted_amount": round(float(converted), 2),
        "rate_date": data.get("date"),
        "source": "MCP Currency Tool (Frankfurter API)",
        "note": "Exchange rate retrieved at query time.",
    }


def format_currency_for_llm(result: dict) -> str:
    """Format currency API response as readable text for the LLM."""
    if not result.get("success"):
        return f"[MCP Currency Tool - FAILED] {result.get('error', 'Unknown error')}"

    return (
        f"[MCP Currency Tool] {result['amount']} {result['from_currency']} = "
        f"{result['converted_amount']} {result['to_currency']} "
        f"(rate date: {result.get('rate_date', 'N/A')})\n"
        f"Source: {result['source']}"
    )
