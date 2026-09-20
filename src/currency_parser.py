"""Parse natural-language currency conversion requests."""

from __future__ import annotations

import re

CURRENCY_ALIASES = {
    "SGD": "SGD",
    "S$": "SGD",
    "SINGAPORE DOLLAR": "SGD",
    "SINGAPORE DOLLARS": "SGD",
    "INR": "INR",
    "RS": "INR",
    "RUPEE": "INR",
    "RUPEES": "INR",
    "USD": "USD",
    "US DOLLAR": "USD",
    "US DOLLARS": "USD",
    "DOLLAR": "USD",
    "DOLLARS": "USD",
}


def _normalize_currency(token: str) -> str | None:
    cleaned = token.upper().strip().replace(".", "")
    if cleaned in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[cleaned]
    if re.fullmatch(r"[A-Z]{3}", cleaned):
        return cleaned
    return None


def parse_currency_request(question: str) -> tuple[float, str, str] | None:
    """Extract amount and currency pair from a user question."""
    text = question.strip()

    patterns = [
        r"convert\s+([A-Za-z$]{1,20}|\b[A-Z]{3}\b)\s+([\d,]+(?:\.\d+)?)\s+to\s+([A-Za-z$ ]{1,30})",
        r"([\d,]+(?:\.\d+)?)\s+([A-Za-z$]{1,20}|\b[A-Z]{3}\b)\s+(?:to|in)\s+([A-Za-z$ ]{1,30})",
        r"([A-Za-z$]{1,20}|\b[A-Z]{3}\b)\s+([\d,]+(?:\.\d+)?)\s+to\s+([A-Za-z$ ]{1,30})",
        r"budget\s+of\s+([A-Za-z$ ]{1,20})\s+([\d,]+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 2:
            from_cur = _normalize_currency(groups[0])
            amount = float(groups[1].replace(",", ""))
            to_cur = "SGD"
            if from_cur:
                return amount, from_cur, to_cur
            continue

        a, b, c = groups
        if re.search(r"\d", a):
            amount_str, from_token, to_token = a, b, c
        elif re.search(r"\d", b):
            from_token, amount_str, to_token = a, b, c
        else:
            from_token, amount_str, to_token = a, b, c

        from_cur = _normalize_currency(from_token)
        to_cur = _normalize_currency(to_token.split()[-1] if " " in to_token else to_token)
        if not to_cur and "singapore" in to_token.lower():
            to_cur = "SGD"
        if from_cur and to_cur:
            return float(amount_str.replace(",", "")), from_cur, to_cur

    pair_match = re.search(
        r"from\s+([A-Za-z$ ]{1,20})\s+to\s+([A-Za-z$ ]{1,30})",
        text,
        re.IGNORECASE,
    )
    if pair_match:
        from_cur = _normalize_currency(pair_match.group(1))
        to_cur = _normalize_currency(pair_match.group(2).split()[-1])
        if not to_cur and "singapore" in pair_match.group(2).lower():
            to_cur = "SGD"
        if from_cur and to_cur:
            return 1.0, from_cur, to_cur

    return None
