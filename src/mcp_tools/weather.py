"""Weather tool logic (used by MCP server and LangChain agent)."""

from __future__ import annotations

import httpx

from src.config import DESTINATION, DESTINATION_LATITUDE, DESTINATION_LONGITUDE, DESTINATION_TIMEZONE

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _describe_weather(code: int) -> str:
    return WEATHER_CODES.get(int(code), f"Code {code}")


def get_weather_forecast(days: int = 3) -> dict:
    """Fetch weather forecast from Open-Meteo (free, no API key required)."""
    days = max(1, min(days, 7))
    params = {
        "latitude": DESTINATION_LATITUDE,
        "longitude": DESTINATION_LONGITUDE,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
        "timezone": DESTINATION_TIMEZONE,
        "forecast_days": days,
    }

    try:
        response = httpx.get(OPEN_METEO_FORECAST_URL, params=params, timeout=15.0)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        return {
            "success": False,
            "destination": DESTINATION,
            "error": f"Weather service unavailable: {exc}",
            "source": "MCP Weather Tool (Open-Meteo)",
        }

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    forecast = []

    for i, date in enumerate(dates):
        precip = daily.get("precipitation_sum", [0] * len(dates))[i]
        code = daily.get("weathercode", [0] * len(dates))[i]
        forecast.append(
            {
                "date": date,
                "temp_max_c": daily.get("temperature_2m_max", [None] * len(dates))[i],
                "temp_min_c": daily.get("temperature_2m_min", [None] * len(dates))[i],
                "precipitation_mm": precip,
                "conditions": _describe_weather(code),
                "rain_expected": float(precip or 0) > 1.0,
            }
        )

    return {
        "success": True,
        "destination": DESTINATION,
        "forecast_days": days,
        "forecast": forecast,
        "source": "MCP Weather Tool (Open-Meteo API)",
        "note": "Current forecast data retrieved at query time.",
    }


def format_weather_for_llm(result: dict) -> str:
    """Format weather API response as readable text for the LLM."""
    if not result.get("success"):
        return f"[MCP Weather Tool - FAILED] {result.get('error', 'Unknown error')}"

    lines = [
        f"[MCP Weather Tool] Forecast for {result['destination']} ({result['forecast_days']} days)",
        f"Source: {result['source']}",
        "",
    ]
    for day in result.get("forecast", []):
        rain_note = "Rain expected — prefer indoor activities" if day["rain_expected"] else "Suitable for outdoor activities"
        lines.append(
            f"- {day['date']}: {day['conditions']}, "
            f"{day['temp_min_c']}°C–{day['temp_max_c']}°C, "
            f"precipitation {day['precipitation_mm']} mm. {rain_note}"
        )
    return "\n".join(lines)
