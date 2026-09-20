"""MCP server for currency conversion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from src.mcp_tools.currency import convert_currency, format_currency_for_llm

mcp = FastMCP("currency-converter")


@mcp.tool()
def convert_currency_amount(amount: float, from_currency: str, to_currency: str) -> str:
    result = convert_currency(amount, from_currency, to_currency)
    return format_currency_for_llm(result)


if __name__ == "__main__":
    mcp.run()
