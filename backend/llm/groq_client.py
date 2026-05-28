"""
llm/groq_client.py — Groq Cloud LLM client for TechnoBot.

Replaces the local Ollama inference stack.  All call sites that previously
used call_ollama() / call_ollama_stream() route through call_groq() here.

API key must be set in .env:
    GROQ_API_KEY=gsk_...

Model
─────
  llama3-8b-8192  — 8 B parameters, 8 k context, ~1–3 s latency on Groq Cloud.
  Produces structured bullet output without the depth guards needed for local
  small models.

Response contract
─────────────────
  Success → {"content": "<text>"}
  Error   → {"error": "<code>", "message": "<human-readable>"}

  Callers check resp.get("error") before accessing resp["content"].
  Never raises — all exceptions are caught and returned as error dicts.
"""

from __future__ import annotations

import os
import time
import requests

# ── Load .env if python-dotenv is installed ──────────────────────────────────
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass   # python-dotenv optional — env vars may be set by the shell

# ── Configuration ─────────────────────────────────────────────────────────────
GROQ_API_KEY: str | None = os.environ.get("GROQ_API_KEY") or None
GROQ_URL:     str        = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL:   str        = "llama-3.1-8b-instant"

# Hard token cap — prevents runaway generation.
# 5 bullets × ~50 words × 1.4 tokens/word ≈ 350 tokens; add headroom → 800.
_GROQ_MAX_TOKENS: int = 800

# ── Startup diagnostics ───────────────────────────────────────────────────────
if GROQ_API_KEY:
    print(f"[LLM] Using Groq model: {GROQ_MODEL}")
else:
    print("[LLM] WARNING: GROQ_API_KEY not set — all LLM calls will fail. "
          "Add GROQ_API_KEY to backend/.env")


# ── Public API ────────────────────────────────────────────────────────────────

def call_groq(
    messages:    list[dict],
    temperature: float = 0.3,
    timeout:     int   = 30,
    max_tokens:  int   = _GROQ_MAX_TOKENS,
) -> dict:
    """Send a chat-completion request to Groq and return a result dict.

    Parameters
    ----------
    messages    : [{"role": "system"|"user"|"assistant", "content": "..."}]
    temperature : 0.1 for structured format calls, 0.3 for analysis.
    timeout     : HTTP request timeout in seconds.
    max_tokens  : Hard cap on generated tokens.

    Returns
    -------
    {"content": "<text>"}                on success
    {"error": "<code>", "message": "…"} on any failure

    Never raises — all exceptions are caught.
    """
    if not GROQ_API_KEY:
        return {
            "error":   "no_api_key",
            "message": "⚠️ GROQ_API_KEY not configured — set it in backend/.env",
        }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       GROQ_MODEL,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    _t0 = time.monotonic()
    try:
        response = requests.post(
            GROQ_URL,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        elapsed = time.monotonic() - _t0

        if response.status_code == 429:
            print(f"[LLM] Groq rate-limited after {elapsed:.2f}s")
            return {
                "error":   "rate_limit",
                "message": "⚠️ High demand — please retry in a few seconds",
            }

        if response.status_code != 200:
            body = response.text[:200]
            print(f"[LLM] Groq API error {response.status_code}: {body}")
            return {
                "error":   "api_error",
                "message": f"⚠️ LLM service error ({response.status_code}) — please try again",
            }

        content = response.json()["choices"][0]["message"]["content"]
        print(f"[LLM] Groq OK in {elapsed:.2f}s "
              f"({len(content.split())} words, model={GROQ_MODEL})")
        return {"content": content}

    except requests.exceptions.Timeout:
        elapsed = time.monotonic() - _t0
        print(f"[LLM] Groq timeout after {elapsed:.2f}s")
        return {
            "error":   "timeout",
            "message": "⚠️ Response took too long — please try again",
        }
    except requests.exceptions.ConnectionError as exc:
        print(f"[LLM] Groq connection error: {exc}")
        return {
            "error":   "connection_error",
            "message": "⚠️ Cannot reach Groq API — check your internet connection",
        }
    except Exception as exc:
        print(f"[LLM] Groq unexpected error: {type(exc).__name__}: {exc}")
        return {
            "error":   "unexpected",
            "message": "⚠️ LLM service unavailable — please try again",
        }
