"""Secret redaction for anything that can reach logs or API responses.

Exchange SDKs happily embed the signed request URL in their exception text, and
several venues sign GET requests by putting the API key and signature into the
query string.  Everything that leaves the adapter therefore goes through
:func:`redact_secrets` first.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

__all__ = ["REDACTED", "redact_secrets"]

REDACTED = "<redacted>"

#: Whole query strings are dropped: they are the usual carrier of signed params.
_QUERY_RE = re.compile(r"\?[^\s\"']*")

#: ``key=value`` / ``key: value`` pairs with a sensitive name.
_SENSITIVE_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?key(?:id)?|secret|signature|sign|token|"
    r"password|passphrase|authorization)\b\s*[:=]\s*[^\s,&;)\"']+"
)

#: Shortest value we are willing to substitute verbatim (avoids mangling text).
_MIN_SECRET_LENGTH = 6


def redact_secrets(text: str, *, extra: Iterable[str] = ()) -> str:
    """Strip query strings, sensitive key/value pairs and known secret values."""
    cleaned = _QUERY_RE.sub(f"?{REDACTED}", text)
    cleaned = _SENSITIVE_RE.sub(lambda match: f"{match.group(1)}={REDACTED}", cleaned)
    for secret in extra:
        if secret and len(secret) >= _MIN_SECRET_LENGTH:
            cleaned = cleaned.replace(secret, REDACTED)
    return cleaned
