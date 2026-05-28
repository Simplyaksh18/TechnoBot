from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ── UTF-8 / emoji-path fix — must happen BEFORE any yfinance/certifi import ──
import sys
import os
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# ── Startup environment validation ───────────────────────────────────────────
# Non-fatal: validation prints warnings/errors but does not prevent the server
# from starting.  This lets VS Code / uvicorn --reload continue to work even
# when internet is temporarily unavailable.

# Force-register python-multipart into the system namespace for FastAPI
try:
    __import__('multipart')
except ImportError:
    # Manually tell Python that python-multipart satisfies 'multipart'
    sys.modules['multipart'] = __import__('python_multipart')
try:
    from startup import validate_environment as _validate_env
    _validate_env(strict=False)
except Exception as _se:
    print(f"[TechnoBot/startup] Startup validation skipped: {_se}")

from technobot_router import (
    answer, build_prompt, get_session, CONVERSATION_WINDOW,
    _build_groq_messages, _format_llm_error,
    _build_evidence_dict, _build_insight_dict,
    _parse_insight_text,
    generate_followup_from_last_bullet,
    build_evidence_summary_for_context,
    # _determine_response_status — Fix 3: no longer called; status is read from
    # context["_final_status"] stamped by answer(). Function kept in router for
    # comparison path and edge cases but NOT imported or used in api_server.py.
)
from llm.groq_client import call_groq, GROQ_MODEL
from fastapi.responses import FileResponse, StreamingResponse
import os
from typing import Optional
from pydantic import BaseModel

from technobot_data.historical_fetcher import fetch_historical_data, export_csv
from technobot_router import detect_instrument, detect_timeframe
from fastapi import UploadFile, File
import shutil
import pathlib
import json
import re


class ExportRequest(BaseModel):
    instrument: Optional[str] = None
    ticker: Optional[str] = None
    timeframe: Optional[str] = None

class TitleRequest(BaseModel):
    message: str

class TitleResponse(BaseModel):
    title: str

class ChatRequest(BaseModel):
    chat_id: str
    message: str

class ChatResponse(BaseModel):
    status:   str       # "ok" | "partial" | "unavailable"
    analysis: str       # bullet-point TA text OR markdown comparison table
    evidence: dict = {} # dynamic {source, ticker, instrument, timeframe, link}
    insight:  dict = {} # deterministic {regime, pattern, implication}
    followup: str  = "" # single contextual follow-up question


def _render_comparison_markdown(data: dict) -> str:
    """Render a comparison payload dict as a frontend-friendly markdown table.

    Input shape (from handle_comparison_request):
        {"inst1": "...", "inst2": "...", "timeframe": "...",
         "rows": [{"aspect": "...", "a": "...", "b": "..."}, ...]}

    Output: markdown heading + pipe table.
    """
    inst1 = data.get("inst1", "Stock A")
    inst2 = data.get("inst2", "Stock B")
    tf    = data.get("timeframe", "")
    rows  = data.get("rows") or []

    heading = f"### {inst1} vs {inst2}"
    if tf:
        heading += f"  —  {tf.title()} timeframe"

    table = [
        f"| Metric | {inst1} | {inst2} |",
        "|--------|---------|---------|",
    ]
    for row in rows:
        table.append(
            f"| {row.get('aspect','—')} | {row.get('a','—')} | {row.get('b','—')} |"
        )

    return heading + "\n\n" + "\n".join(table)

app = FastAPI(title="TechnoBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    """D1: SSE streaming endpoint. Yields tokens as `data: <token>\n\n` frames.

    Non-streaming paths (comparison, historical, file upload, OOT, disambiguation,
    confirmations) are routed through answer() and emitted as a single SSE frame.
    Only plain single-stock LLM analysis streams token-by-token.
    """
    from technobot_router import (
        build_prompt, enforce_bullet_only, sanitize_language,
        _enrich_bullet_semantics, _ensure_naturalness,
        generate_followup_from_last_bullet, format_evidence_summary,
        generate_insight_layer,
        is_greeting, greeting_response, normalize_query,
        detect_comparison_query, is_ta_query,
        _format_data_unavailable,
    )

    sess = get_session(req.chat_id)
    msg  = req.message.strip()
    msg_lower = msg.lower()

    # ── Intent classification: which paths MUST go through answer() ───────────
    #
    # 1. Comparison queries ("X vs Y", "compare A and B") must hit
    #    handle_comparison_request() so they produce __COMPARISON__{json}.
    #    The streaming LLM path has no comparison routing at all.
    #
    # 2. Out-of-topic queries must hit the OOT guard in answer() — otherwise the
    #    streaming path calls the LLM on completely irrelevant prompts.
    #
    # 3. Historical / export / confirmation paths need answer()'s session logic.
    #
    # 4. File-path messages trigger parse_uploaded_file inside answer().
    _is_comparison = detect_comparison_query(msg) is not None
    has_uploaded_context = bool(
        sess.get("last_context")
        and sess["last_context"].get("data_source") == "uploaded_file"
    )
    uploaded_file_intent = has_uploaded_context and any(
        kw in msg_lower for kw in (
            "file", "csv", "upload", "analyze this file", "analyse this file",
            "analyze file", "analyse file", "this file", "uploaded file",
        )
    )
    _is_oot        = not is_ta_query(normalize_query(msg)) and not uploaded_file_intent
    _is_historical = any(k in msg_lower for k in [
        "historical", "history", "price data", "show data", "export",
    ])
    _is_confirm    = any(k in msg_lower for k in [
        "yes", "yeah", "yep", "ok", "okay", "sure", "go ahead",
    ])
    _is_file       = os.path.exists(msg)

    needs_full_answer = (
        _is_comparison
        or _is_oot
        or _is_historical
        or _is_confirm
        or _is_file
    )

    if needs_full_answer:
        reason = (
            "comparison" if _is_comparison else
            "oot"        if _is_oot        else
            "historical" if _is_historical else
            "confirm"    if _is_confirm    else
            "file"
        )
        print(f"[TechnoBot/stream] full-answer route: {reason}")
        full = answer(req.message, chat_id=req.chat_id)
        async def full_response_gen():
            yield f"data: {json.dumps({'token': full, 'done': True})}\n\n"
        return StreamingResponse(full_response_gen(), media_type="text/event-stream")

    if is_greeting(msg):
        greeting = greeting_response(None)
        async def greeting_gen():
            yield f"data: {json.dumps({'token': greeting, 'done': True})}\n\n"
        return StreamingResponse(greeting_gen(), media_type="text/event-stream")

    # Follow-up history trimming: 1 pair for follow-ups keeps token overhead low
    is_followup = bool(sess["conversation_history"])
    if is_followup:
        history_window = sess["conversation_history"][-2:]
    else:
        history_window = sess["conversation_history"][-(CONVERSATION_WINDOW * 2):]

    sys_prompt, usr_msg, context = build_prompt(req.message, fallback_context=sess.get("last_context"))

    # ── Data validation gate (mirrors answer()) ───────────────────────────────
    # Precedence:
    #   1. _lightweight — all providers failed; context is qualitative only
    #      (trend="consolidating", not "unknown" — must be checked explicitly)
    #   2. rate_limited — provider returned 429
    #   3. trend=="unknown" — no OHLCV processed at all
    # None of these should ever reach Ollama.
    _data_error_msg: str | None = None

    if context.get("_lightweight"):
        # Phase 12: lightweight path — _format_data_unavailable handles all 3 cases
        print(f"[TechnoBot/stream] GATE _lightweight=True — no LLM call "
              f"ticker={context.get('ticker')} tf={context.get('timeframe')}")
        _data_error_msg = _format_data_unavailable(context)
    elif context.get("rate_limited"):
        instr_name = context.get("instrument_name") or "this instrument"
        _data_error_msg = (
            f"⚠️ Live market data for **{instr_name}** is temporarily unavailable "
            "due to Yahoo Finance rate limiting (HTTP 429).\n\n"
            "This happens when multiple requests are made in a short time. "
            "**Please wait 3–5 minutes and try again.**\n\n"
            "In the meantime you can:\n"
            "• **Upload a CSV** with price data to analyse offline\n"
            "• **Ask about TA concepts** (RSI, MACD, support/resistance) without live data\n"
            "• **Come back shortly** — the data provider cooldown clears automatically"
        )
    elif (
        context.get("trend") == "unknown"
        and context.get("rsi") is None
        and context.get("last_price") is None
    ):
        instr_name = context.get("instrument_name") or "this instrument"
        tf_label   = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}.get(
            context.get("timeframe", "1d"), context.get("timeframe", "daily")
        )
        _data_error_msg = (
            f"⚠️ Live market data for **{instr_name}** on the {tf_label} timeframe "
            f"could not be retrieved right now.\n\n"
            "What you can do:\n"
            "• **Try again** in 30–60 seconds — transient network delays resolve quickly\n"
            "• **Check the ticker** — e.g., use 'Reliance' or 'RELIANCE.NS' explicitly\n"
            "• **Switch timeframe** — try 'daily' if you asked for weekly/monthly\n"
            "• **Upload a CSV** to analyse offline data without a live feed"
        )

    if _data_error_msg:
        async def data_error_gen():
            yield f"data: {json.dumps({'token': _data_error_msg, 'done': True})}\n\n"
        return StreamingResponse(data_error_gen(), media_type="text/event-stream")

    # Snapshot tracking fields before async generator closes over them
    prev_ticker      = sess.get("last_response_ticker")
    prev_tf          = sess.get("last_response_tf")
    prev_insight_hash = sess.get("last_insight_hash")

    async def token_generator():
        # Single Groq call — yields full response in one chunk (1–3 s latency)
        import time as _time
        _t_stream = _time.monotonic()
        print(f"[LLM START] stream ticker={context.get('ticker','?')} tf={context.get('timeframe','?')}")
        _msgs_stream = _build_groq_messages(sys_prompt, usr_msg, history=history_window, query=req.message)
        _resp_stream  = call_groq(_msgs_stream, temperature=0.1, timeout=30)
        print(f"[LLM END] stream duration={_time.monotonic() - _t_stream:.2f}s")

        if _resp_stream.get("error"):
            err_txt = _format_llm_error(_resp_stream)
            yield f"data: {json.dumps({'token': err_txt, 'done': True})}\n\n"
            return

        raw = _resp_stream["content"]
        # Emit full text as a single SSE token (frontend accumulates it normally)
        yield f"data: {json.dumps({'token': raw, 'done': False})}\n\n"

        # Post-processing: apply same quality pipeline as non-streaming path
        analysis_text = enforce_bullet_only(sanitize_language(raw)) or raw.strip()
        # Semantic enrichment (meaning-driven, not length-driven)
        analysis_text = _enrich_bullet_semantics(analysis_text, context)
        # Naturalness pass — remove robotic artifacts and repetition
        analysis_text = _ensure_naturalness(analysis_text)
        # Always use controlled template follow-up (never the raw LLM question)
        followup = generate_followup_from_last_bullet(analysis_text, req.message, context)

        # ── Follow-up verbosity control ──────────────────────────────────────
        # When the user is asking about the same instrument + timeframe as the
        # previous turn, suppress evidence, multi-TF context, and insight — they
        # were already shown and haven't changed, so re-emitting them is noise.
        curr_ticker = context.get("ticker", "")
        curr_tf     = context.get("timeframe", "")
        context_unchanged = (
            is_followup
            and bool(prev_ticker)
            and prev_ticker == curr_ticker
            and prev_tf == curr_tf
            and context.get("data_source") != "uploaded_file"
        )

        if context_unchanged:
            evidence = ""
            multitf  = ""
            raw_insight = generate_insight_layer(context)
            new_hash = str(hash(raw_insight[:80]))
            insight = "" if new_hash == prev_insight_hash else raw_insight
        else:
            evidence = context.get("evidence") or build_evidence_summary_for_context(context)
            insight  = generate_insight_layer(context)
            # Multi-timeframe context (uses LRU cache — ~0ms if warm)
            from technobot_router import generate_multitf_context
            multitf = ""
            if context.get("data_source") != "uploaded_file":
                try:
                    multitf = generate_multitf_context(
                        curr_ticker,
                        curr_tf,
                        context,
                    )
                except Exception:
                    multitf = ""

        print(f"[TechnoBot/stream] followup_mode={context_unchanged} | evidence={bool(evidence)} | insight={bool(insight)} | multitf={bool(multitf if not context_unchanged else '')}")

        suffix_parts = []
        if evidence:
            suffix_parts.append(evidence)
        if not context_unchanged and multitf:
            suffix_parts.append(multitf)
        if insight:
            suffix_parts.append(insight)
        if followup:
            suffix_parts.append(followup)
        suffix = "\n\n".join(suffix_parts)

        if suffix:
            yield f"data: {json.dumps({'token': '\n\n' + suffix, 'done': False})}\n\n"

        # Update session memory
        sess["last_context"]          = context
        sess["last_followup_intent"]  = followup
        sess["last_response_ticker"]  = curr_ticker
        sess["last_response_tf"]      = curr_tf
        if insight:
            sess["last_insight_hash"] = str(hash(insight[:80]))

        full_response = analysis_text + (("\n\n" + suffix) if suffix else "")

        sess["conversation_history"].append({"role": "user", "content": req.message})
        sess["conversation_history"].append({"role": "assistant", "content": full_response})
        max_entries = CONVERSATION_WINDOW * 2
        if len(sess["conversation_history"]) > max_entries:
            sess["conversation_history"] = sess["conversation_history"][-max_entries:]

        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.post("/generate-title", response_model=TitleResponse)
async def generate_title(req: TitleRequest):
    """A2: Generate a concise 4-7 word chat title from the first user message.

    Uses Groq with a minimal prompt. Falls back gracefully (empty string) if
    the LLM is unavailable so the UI can use its client-side title generator.
    """
    instrument_name, _ticker = detect_instrument(req.message)

    prompt = (
        "Generate a concise 2-5 word chat title for the following market query. "
        "Focus on the instrument name and the analysis intent. "
        "Do not include stock, share, timeframe, or technical-analysis boilerplate unless it is essential. "
        "Use title case. Return ONLY the title — no quotes, no punctuation at the end.\n\n"
        f"Query: {req.message}"
    )

    def _fallback_title() -> str:
        if instrument_name:
            return f"{instrument_name} Analysis"
        return "Market Analysis"

    def _sanitize_title(raw_title: str) -> str:
        title = raw_title.strip().strip('"').strip("'")
        title = re.sub(r"\b(stock|shares?|technical|analysis|chart|candles?)\b", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\b\d+\s*[-/]?\s*(year|month|week|day)s?\b", "", title, flags=re.IGNORECASE)
        title = re.sub(r"\s+", " ", title).strip(" -:|")
        if not title:
            return _fallback_title()
        if len(title) > 32 or re.search(r"\d", title):
            return _fallback_title()
        if instrument_name and instrument_name.lower() not in title.lower():
            return _fallback_title()
        return title

    try:
        resp = call_groq(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=10,
            max_tokens=30,
        )
        if resp.get("error"):
            return {"title": _fallback_title()}
        title = _sanitize_title(resp["content"])
        return {"title": title}
    except Exception:
        # Graceful fallback — client-side generator handles the rest
        return {"title": _fallback_title()}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Sync endpoint so FastAPI runs it in anyio's thread pool.

    Always returns exactly 5 fields — no optional or missing keys:
      status   — "ok" | "partial" | "unavailable"
      analysis — bullet-point TA text OR markdown comparison table
      evidence — {source, ticker, instrument, timeframe, link}  (empty dict on failure)
      insight  — {regime, pattern, implication}                  (empty dict on failure)
      followup — single contextual follow-up question            ("" on failure)
    """
    try:
        resp = answer(req.message, chat_id=req.chat_id)
    except Exception as e:
        print(f"[TechnoBot/chat] exception: {e}")
        return {
            "status":   "unavailable",
            "analysis": f"Service error — please try again.",
            "evidence": {},
            "insight":  {},
            "followup": "",
        }

    sess    = get_session(req.chat_id)
    context = sess.get("last_context") or {}
    rtype   = sess.get("_last_response_type", "text")

    # ── Comparison path ───────────────────────────────────────────────────────
    # Render the structured comparison payload as a markdown table inside `analysis`.
    # No separate `comparison` field is exposed — the frontend reads `analysis`.
    if rtype == "comparison" or (isinstance(resp, str) and resp.startswith("__COMPARISON__")):
        raw_payload = resp.split("\n\n", 1)[0].removeprefix("__COMPARISON__")
        try:
            comparison_data = json.loads(raw_payload)
        except Exception:
            comparison_data = None

        # Render rows to markdown table; fallback to plain text if parse failed
        if comparison_data and comparison_data.get("rows"):
            cmp_analysis = _render_comparison_markdown(comparison_data)
        else:
            # Fallback: strip the __COMPARISON__ sentinel and return raw text
            cmp_analysis = "\n\n".join(resp.split("\n\n")[1:]).strip() or resp

        # Comparison insight → structured insight dict
        _cmp_insight_raw = sess.get("_last_comparison_insight", "")
        _cmp_insight = _parse_insight_text(_cmp_insight_raw) if _cmp_insight_raw else {}

        _cmp_followup = sess.get("_last_followup", "")
        _cmp_evidence = _build_evidence_dict(context) if context else {}

        ticker = context.get("ticker", "?")
        print(f"[DATA STATUS] {ticker} → ok (source=comparison)")

        return {
            "status":   "ok",
            "analysis": cmp_analysis,
            "evidence": _cmp_evidence or {},
            "insight":  _cmp_insight or {},
            "followup": _cmp_followup,
        }

    # ── Status determination ──────────────────────────────────────────────────
    # Fix 2: Read _final_status stamped by answer() — single source of truth.
    # _determine_response_status() is NO LONGER called here; it recomputes from
    # context fields and can diverge from the routing decision that was actually
    # made, producing inconsistent results across requests.
    status = context.get("_final_status", "unavailable") if context else "unavailable"
    # Partial analysis path still overrides (rtype set by comparison path)
    if rtype == "partial_analysis":
        status = "partial"
    # Uploaded-file sessions and pure text always report ok
    if rtype not in ("analysis", "partial_analysis", "unavailable"):
        status = "ok"

    # ── Analysis text ─────────────────────────────────────────────────────────
    if rtype in ("analysis", "partial_analysis"):
        analysis = sess.get("_last_analysis") or resp
    else:
        analysis = resp

    # ── Evidence — always a dict (never None) ────────────────────────────────
    # For unavailable responses caused by an unknown / unresolvable ticker,
    # return a clean empty-ish evidence block so the frontend never shows
    # stale ticker data from a previous session.
    # Rate-limited responses still show the (valid) ticker info.
    _is_unknown_ticker = (
        status == "unavailable"
        and not (context or {}).get("rate_limited")
        and (
            not context
            or (
                context.get("trend") == "unknown"
                and context.get("last_price") is None
            )
        )
    )
    if _is_unknown_ticker:
        evidence = {
            "source":     "No data",
            "ticker":     "",
            "instrument": "",
            "timeframe":  "",
            "link":       "",
        }
    else:
        evidence = (_build_evidence_dict(context) if context else {}) or {}

    # ── Insight — always a dict (never None) ─────────────────────────────────
    # Uses generate_insight_layer() via _build_insight_dict() — 11-case engine.
    # Populated for both "ok" and "partial".
    if rtype in ("analysis", "partial_analysis") and context:
        insight = _build_insight_dict(context) or {}
    else:
        insight = {}

    # ── Followup — always a string (never None) ──────────────────────────────
    followup = sess.get("_last_followup", "")
    if not followup and rtype in ("analysis", "partial_analysis") and context:
        followup = generate_followup_from_last_bullet(
            sess.get("_last_analysis", ""), req.message, context
        )

    # ── [DATA STATUS] logging ─────────────────────────────────────────────────
    ticker = context.get("ticker", "?") if context else "?"
    source = context.get("data_source", "unknown") if context else "unknown"
    print(f"[DATA STATUS] {ticker} → {status} (source={source})")
    # Fix 4: Final status log — confirms what the frontend receives.
    print(f"[TechnoBot/api] FINAL STATUS → {status} ticker={context.get('ticker') if context else '?'}")

    return {
        "status":   status,
        "analysis": analysis or "",
        "evidence": evidence,
        "insight":  insight,
        "followup": followup or "",
    }


@app.get("/chart-data")
async def chart_data(ticker: str, timeframe: str = "1d"):
    """Return sampled OHLCV close prices for a ticker as JSON.

    Used by the frontend PriceLineChart component. Returns up to 200 data
    points (sampled evenly from the full history) so charts stay responsive.
    """
    try:
        df, meta = fetch_historical_data(ticker, timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {e}")

    # Reset index first so Date becomes a regular column, THEN normalise all names
    df = df.reset_index()
    df.columns = [c.strip().lower() for c in df.columns]

    # Find date column
    date_col = None
    for c in df.columns:
        if "date" in c or "datetime" in c:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    if "close" not in df.columns:
        raise HTTPException(status_code=500, detail="No close column in data")

    # Sample to max 200 points
    if len(df) > 200:
        step = len(df) // 200
        df = df.iloc[::step]

    records = []
    for _, row in df.iterrows():
        date_val = row[date_col]
        if hasattr(date_val, "strftime"):
            date_str = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)[:10]
        try:
            close_val = float(row["close"])
            if close_val == close_val:  # NaN check
                records.append({"date": date_str, "close": round(close_val, 2)})
        except (ValueError, TypeError):
            continue

    return {"ticker": ticker, "timeframe": timeframe, "data": records}


@app.get("/exports/{filename:path}")
async def download_export(filename: str):
    """Serve exported CSV files from possible exports folders."""
    # Search common locations where CSVs may be written
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "exports"),
        os.path.join(os.path.dirname(__file__), "exports"),
        os.path.join(os.path.dirname(__file__), "technobot_data", "exports"),
        os.path.abspath(os.path.join(os.getcwd(), "exports")),
    ]

    for base in candidates:
        path = os.path.abspath(os.path.join(base, filename))
        if os.path.exists(path) and os.path.isfile(path):
            return FileResponse(path, media_type="text/csv", filename=os.path.basename(path))

    raise HTTPException(status_code=404, detail=f"Export not found: {filename}")


@app.post("/exports")
async def create_export(req: ExportRequest):
    """Create an exported CSV for a given instrument/ticker and timeframe.

    This endpoint triggers the same CSV export logic used elsewhere but
    accepts explicit instrument/ticker/timeframe to avoid relying on
    conversational confirm paths.
    """
    instrument = req.instrument
    ticker = req.ticker
    timeframe = req.timeframe

    # Resolve ticker from instrument if needed
    if not ticker and instrument:
        _, ticker = detect_instrument(instrument)

    # Provide a sensible default when nothing was passed (avoid None going into yfinance)
    if not ticker:
        # default to NIFTY if no ticker provided
        ticker = "^NSEI"
        if not instrument:
            instrument = "NIFTY 50"

    # Resolve timeframe if not provided
    if not timeframe:
        timeframe = detect_timeframe(instrument or ticker or "") or "1d"

    try:
        df, meta = fetch_historical_data(ticker, timeframe)
    except Exception as e:
        # Return a clearer error to the client to aid debugging
        raise HTTPException(status_code=500, detail=f"Failed to fetch data for ticker '{ticker}': {e}")

    try:
        path = export_csv(df, ticker, timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export CSV: {e}")

    filename = os.path.basename(path)
    return {"path": path, "filename": filename}


@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    """Remove a chat session from the in-memory SESSION_STORE.

    Called by the frontend when the user deletes a chat from the sidebar.
    Chat history lives in the browser's localStorage — this endpoint only
    cleans up the server-side conversation context so it doesn't linger.
    """
    from technobot_router import SESSION_STORE
    existed = chat_id in SESSION_STORE
    SESSION_STORE.pop(chat_id, None)
    print(f"[TechnoBot] DELETE /chat/{chat_id} — {'removed' if existed else 'not found (already gone)'}")
    return {"deleted": chat_id, "existed": existed}


@app.get("/sessions/{chat_id}/exports")
async def get_export_history(chat_id: str):
    """D8: Return the export log for a given chat session."""
    from technobot_router import SESSION_STORE
    sess = SESSION_STORE.get(chat_id, {})
    return {"exports": sess.get("export_history", [])}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept a CSV upload and save it to the server's uploads directory.

    Returns the absolute file path which can be sent back to the `/chat`
    endpoint so backend conversational logic treats it as an uploaded file.
    """
    uploads_dir = os.path.join(os.path.dirname(__file__), "technobot_data", "uploads")
    pathlib.Path(uploads_dir).mkdir(parents=True, exist_ok=True)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    safe_name = os.path.basename(file.filename)
    dest_path = os.path.abspath(os.path.join(uploads_dir, safe_name))

    try:
        with open(dest_path, "wb") as dest:
            shutil.copyfileobj(file.file, dest)
    finally:
        await file.close()

    return {"path": dest_path, "filename": safe_name}

if __name__ == "__main__":
    import uvicorn
    import os

    module_name = os.path.splitext(os.path.basename(__file__))[0]
    pkg = __package__ or ""
    if pkg:
        import_path = f"{pkg}.{module_name}:app"
    else:
        import_path = f"{module_name}:app"

    port_env = int(os.environ.get("PORT", 8000))

    # 2. Pass the port variable into uvicorn, and disable reload on production
    is_local = os.environ.get("PORT") is None
    uvicorn.run(import_path, host="0.0.0.0", port=port_env, reload=is_local)