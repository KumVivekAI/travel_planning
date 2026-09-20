"""MCP server for Singapore weather forecasts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from src.mcp_tools.weather import format_weather_for_llm, get_weather_forecast

mcp = FastMCP("singapore-weather")


@mcp.tool()
def get_singapore_weather_forecast(days: int = 3) -> str:
    
    result = get_weather_forecast(days=days)
    return format_weather_for_llm(result)


if __name__ == "__main__":
    mcp.run()
