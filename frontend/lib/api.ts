// ─── Backend URL ─────────────────────────────────────────────────────────────
// Priority: NEXT_PUBLIC_API_BASE env var → production backend default.
// Set NEXT_PUBLIC_API_BASE in .env.local for local dev overrides.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "https://technobot-82sa.onrender.com";

// ─── Shared types ─────────────────────────────────────────────────────────────

export type SendPayload = string | { message: string; acknowledge?: boolean };

/**
 * Shape of every /chat response (5-field contract).
 *   status   — "ok" | "partial" | "unavailable"
 *   analysis — bullet-point TA text or markdown comparison table
 *   evidence — { source, ticker, instrument, timeframe, link }
 *   insight  — { regime, pattern, implication }
 *   followup — single contextual follow-up question ("" when absent)
 */
export interface ChatApiResponse {
  status: "ok" | "partial" | "unavailable";
  analysis: string;
  evidence: Record<string, string>;
  insight: Record<string, string>;
  followup: string;
}

// ─── Non-streaming send (used for hidden/file-path messages) ──────────────────

/**
 * POST /chat — synchronous, returns the `analysis` text.
 *
 * Used by sendHiddenMessage() to inject a file-path context message before
 * the user's follow-up query arrives via the SSE streaming path.
 *
 * Task 3: Full try/catch with console.error logging.
 * Task 4: Reads `analysis` from the 5-field response (not the old `response` key).
 */
export async function sendMessage(
  chatId: string,
  messageOrPayload: SendPayload,
): Promise<string> {
  const isObject = typeof messageOrPayload !== "string";
  const body: Record<string, unknown> = {
    chat_id: chatId,
    message: isObject
      ? (messageOrPayload as { message: string }).message
      : messageOrPayload,
  };
  if (isObject && (messageOrPayload as { acknowledge?: boolean }).acknowledge) {
    body.acknowledge = true;
  }

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      // Task 3: Log HTTP-level errors before re-throwing
      console.error(`[TechnoBot/api] POST /chat HTTP error ${res.status}`);
      throw new Error(`HTTP error ${res.status}`);
    }

    // Task 4: Backend returns the 5-field contract — read `analysis`, not `response`
    const data = (await res.json()) as ChatApiResponse;
    return data.analysis ?? "";
  } catch (err) {
    // Task 3: Distinguish connection errors from HTTP errors
    if (err instanceof TypeError && err.message === "Failed to fetch") {
      console.error(
        `[TechnoBot/api] POST /chat — cannot reach backend at ${API_BASE}. ` +
          "Is the backend running on port 8000?",
      );
    } else {
      console.error("[TechnoBot/api] POST /chat error:", err);
    }
    throw err;
  }
}

// ─── SSE streaming send (used for all normal chat messages) ───────────────────

/**
 * POST /chat-stream — SSE, invokes onToken for each arriving token.
 *
 * Resolves with the full accumulated response when `done: true` arrives.
 * Falls back to a single-response parse when ReadableStream is unavailable (SSR).
 *
 * Task 3: Full error logging on HTTP failures and parse errors.
 */
export async function sendMessageStream(
  chatId: string,
  message: string,
  onToken: (token: string) => void,
): Promise<string> {
  let res: Response;

  try {
    res = await fetch(`${API_BASE}/chat-stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chat_id: chatId, message }),
    });
  } catch (err) {
    if (err instanceof TypeError && err.message === "Failed to fetch") {
      console.error(
        `[TechnoBot/api] POST /chat-stream — cannot reach backend at ${API_BASE}. ` +
          "Is the backend running on port 8000?",
      );
    } else {
      console.error("[TechnoBot/api] POST /chat-stream connection error:", err);
    }
    throw err;
  }

  if (!res.ok) {
    console.error(`[TechnoBot/api] POST /chat-stream HTTP error ${res.status}`);
    throw new Error(`HTTP error ${res.status}`);
  }

  // Graceful fallback for environments without ReadableStream (e.g. SSR)
  if (!res.body) {
    const data = (await res.json()) as { response?: string; analysis?: string };
    // Task 4: prefer `analysis`, fall back to legacy `response` key
    const full = data.analysis ?? data.response ?? "";
    onToken(full);
    return full;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let accumulated = "";
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by double-newlines
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const dataLine = frame.split("\n").find((l) => l.startsWith("data: "));
      if (!dataLine) continue;
      try {
        const payload = JSON.parse(dataLine.slice(6)) as {
          token?: string;
          done?: boolean;
        };
        const token = payload.token ?? "";
        if (token) {
          accumulated += token;
          onToken(token);
        }
        if (payload.done) return accumulated;
      } catch {
        // Malformed SSE frame — skip silently
      }
    }
  }

  return accumulated;
}
