"use client";

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useRef,
} from "react";
import { sendMessage as apiSendMessage, sendMessageStream, API_BASE } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

// ─── Types ────────────────────────────────────────────────────────────────────

export type ChatMessage = {
  id?: string;
  role: "user" | "assistant";
  content: string;
  timestamp?: number;
  // ── Structured response fields (Task 1) ──────────────────────────────────
  // Stamped after streaming completes or after /chat JSON response is parsed.
  status?:   "ok" | "partial" | "unavailable";
  evidence?: Record<string, string>;
  insight?:  Record<string, string>;
  followup?: string;
};

export type Chat = {
  id: string;
  title: string;
  timestamp: string;
  messages: ChatMessage[];
};

type ChatContextValue = {
  chats: Chat[];
  currentChatId: string;
  messages: ChatMessage[];
  isLoading: boolean;
  streamingMessageId: string | null;
  addMessage: (m: ChatMessage) => void;
  sendUserMessage: (content: string) => Promise<string>;
  sendMessage: (m: ChatMessage) => Promise<string>;
  sendHiddenMessage: (content: string) => Promise<string>;
  setLoading: (v: boolean) => void;
  startNewChat: () => string;
  loadChat: (id: string) => void;
  updateChatTitle: (chatId: string, title: string) => void;
  deleteChat: (chatId: string) => void;
  awaitingExport: boolean;
  setAwaitingExport: (v: boolean) => void;
  lastFollowUp: string | null;
  setLastFollowUp: (v: string | null) => void;
};

const ChatContext = createContext<ChatContextValue | null>(null);

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeId(prefix = "") {
  return `${prefix}${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 9)}`;
}

/** Safely read a value from localStorage; returns null on any error. */
function lsGet(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}

/** Safely write a value to localStorage; silently swallows quota errors. */
function lsSet(key: string, value: string): void {
  try { localStorage.setItem(key, value); } catch { /* quota exceeded or SSR */ }
}

/** Safely remove a key from localStorage. */
function lsRemove(key: string): void {
  try { localStorage.removeItem(key); } catch { /* ignore */ }
}

/**
 * Read the currently-logged-in user's ID directly from localStorage.
 * Used for the lazy useState initializer which runs before effects fire —
 * at that point useAuth() hasn't yet run its own useEffect to populate state.
 */
function readStoredUserId(): string {
  try {
    const stored = lsGet("user");
    if (stored) {
      const parsed = JSON.parse(stored) as { id?: string };
      if (parsed?.id) return parsed.id;
    }
  } catch { /* fall through */ }
  return "guest";
}

/**
 * A2: Chat title derived directly from the first user message.
 * Truncates to 40 characters so the sidebar stays readable.
 * LLM-generated title from /generate-title takes priority when available.
 */
function generateChatTitle(message: string): string {
  const m = message.trim();
  if (m.length <= 40) return m;
  return m.slice(0, 40).replace(/\s+\S*$/, "").trimEnd() + "…";
}

// ─── Streaming text parsers ───────────────────────────────────────────────────
//
// These mirror the backend's _build_evidence_dict / _parse_insight_text exactly
// so that the structured fields on each message come from THAT message's stream,
// never from a previous message or a cached context dict.

/**
 * Parse the "Evidence summary:" text block produced by format_evidence_summary()
 * into a structured dict.  Returns undefined when no evidence block is present.
 *
 * Task 1 / Task 2 — structured, immutable, per-message.
 */
function parseEvidenceDictFromContent(
  text: string
): Record<string, string> | undefined {
  const lines = text.split("\n");
  const startIdx = lines.findIndex((l) =>
    /^(Evidence Summary|Evidence summary|📊 Evidence)[:\s]/i.test(l.trim())
  );
  if (startIdx === -1) return undefined;

  const result: Record<string, string> = {};
  for (let i = startIdx + 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) break; // evidence block ends at blank line
    // Matches "• Source: Yahoo Finance", "• Ticker: INFY.NS", etc.
    const m = line.match(/^[•\-*]\s*([^:]+):\s*(.+)$/);
    if (!m) continue;
    // "Last Close" → "last_close", "Data Range" → "data_range"
    const key = m[1].trim().toLowerCase().replace(/\s+/g, "_");
    result[key] = m[2].trim();
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

/**
 * Parse the "📌 Contextual Insight:" block produced by generate_insight_layer()
 * into {regime, pattern, implication}.  Mirrors backend _parse_insight_text().
 *
 * Task 4 — structured, immutable, per-message.
 */
function parseInsightDictFromContent(
  text: string
): Record<string, string> | undefined {
  const result: Record<string, string> = {
    regime: "", pattern: "", implication: "",
  };
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (s.startsWith("• Current regime:"))
      result.regime      = s.slice("• Current regime:".length).trim();
    else if (s.startsWith("• Pattern:"))
      result.pattern     = s.slice("• Pattern:".length).trim();
    else if (s.startsWith("• Implication:"))
      result.implication = s.slice("• Implication:".length).trim();
    else if (s.startsWith("🔍 Relative Insight:")) {
      // Comparison insight — flatten into pattern field
      result.regime      = "comparative";
      result.pattern     = s.slice("🔍 Relative Insight:".length).trim();
      result.implication = "";
    }
  }
  return (result.regime || result.pattern || result.implication)
    ? result
    : undefined;
}

// ─── Post-processing ──────────────────────────────────────────────────────────

function postProcessResponse(raw: string): string {
  let out = raw;

  // ── 1. Comparison sentinel ─────────────────────────────────────────────────
  const compLineIdx = out.split("\n").findIndex((l) =>
    l.startsWith("__COMPARISON__")
  );
  if (compLineIdx !== -1) {
    const lines = out.split("\n");
    const compLine = lines[compLineIdx];
    const jsonStr = compLine.slice("__COMPARISON__".length);
    try {
      const payload = JSON.parse(jsonStr) as {
        inst1: string;
        inst2: string;
        rows: Array<{ aspect: string; a: string; b: string }>;
      };
      const tableRows = payload.rows.map((r) => ({
        Aspect: r.aspect,
        [payload.inst1]: r.a,
        [payload.inst2]: r.b,
      }));
      lines[compLineIdx] = `**COMPARISON**${JSON.stringify(tableRows)}`;
    } catch {
      // JSON parse failed — leave raw line so it renders as prose
    }
    out = lines.join("\n");
  }

  // ── 2. 📌 Pattern: → 📌 Contextual Insight: ───────────────────────────────
  out = out.replace(/^📌\s*Pattern:/gm, "📌 Contextual Insight:");

  // ── 3. Last bare question → Follow-up: <question> ─────────────────────────
  const outLines = out.split("\n");
  for (let i = outLines.length - 1; i >= 0; i--) {
    const trimmed = outLines[i].trim();
    if (
      trimmed.endsWith("?") &&
      !trimmed.startsWith("•") &&
      !trimmed.startsWith("*") &&
      !trimmed.startsWith("-") &&
      !trimmed.startsWith("Follow") &&
      !trimmed.startsWith("📌") &&
      !trimmed.startsWith("🔭") &&
      trimmed.length > 15
    ) {
      outLines[i] = `Follow-up: ${trimmed}`;
      break;
    }
  }
  out = outLines.join("\n");

  return out;
}

// ─── Instrument → ticker map (for ticker injection) ──────────────────────────

const INSTRUMENT_TICKER_MAP: Record<string, string> = {
  "tata steel": "TATASTEEL.NS", reliance: "RELIANCE.NS",
  infosys: "INFY.NS", "hdfc bank": "HDFCBANK.NS", "icici bank": "ICICIBANK.NS",
  sbi: "SBIN.NS", tcs: "TCS.NS", "bajaj auto": "BAJAJ-AUTO.NS",
  titan: "TITAN.NS", "axis bank": "AXISBANK.NS",
  "kotak mahindra": "KOTAKBANK.NS", maruti: "MARUTI.NS", ntpc: "NTPC.NS",
  nifty: "^NSEI", sensex: "^BSESN", wipro: "WIPRO.NS",
  "hcl tech": "HCLTECH.NS", "sun pharma": "SUNPHARMA.NS",
  "tata motors": "TATAMOTORS.NS", "tech mahindra": "TECHM.NS",
};

// ─── Storage key helpers (user-scoped) ────────────────────────────────────────

function chatsKey(userId: string)     { return `technobot_chats_${userId}`; }
function currentIdKey(userId: string) { return `technobot_current_id_${userId}`; }

function loadChatsForUser(userId: string): Chat[] {
  try {
    const saved = lsGet(chatsKey(userId));
    if (saved) {
      const parsed: Chat[] = JSON.parse(saved);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch { /* corrupt — fall through */ }
  const id = makeId("chat-");
  return [{ id, title: "New Chat", timestamp: new Date().toLocaleString(), messages: [] }];
}

function loadCurrentIdForUser(userId: string, chats: Chat[]): string {
  try {
    const savedId = lsGet(currentIdKey(userId));
    if (savedId && chats.find((c) => c.id === savedId)) return savedId;
  } catch { /* fall through */ }
  return chats[0]?.id ?? makeId("chat-");
}

// ─── Provider ─────────────────────────────────────────────────────────────────

export function ChatProvider({ children }: { children: React.ReactNode }) {
  // Read auth state — ChatProvider is always rendered inside AuthProvider so
  // useAuth() is available.  The user object starts as null while auth loads
  // from localStorage via useEffect; we also read it synchronously below for
  // the lazy useState initializer.
  const { user } = useAuth();

  // The initial render may happen before AuthProvider's useEffect fires, so we
  // read the stored user directly from localStorage to get the right scoped key.
  const initUserId = readStoredUserId();

  const [chats, setChats] = useState<Chat[]>(() => loadChatsForUser(initUserId));

  const [currentChatId, setCurrentChatId] = useState<string>(() => {
    const initialChats = loadChatsForUser(initUserId);
    return loadCurrentIdForUser(initUserId, initialChats);
  });

  const [isLoading, setIsLoading] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [awaitingExport, setAwaitingExport] = useState(false);
  const [lastFollowUp, setLastFollowUp] = useState<string | null>(null);

  // Stable per-session backend chat_id — scopes server conversation context
  const [backendChatId] = useState<string>(() =>
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : makeId("backend-")
  );

  // Track the previous user id so we can detect login / logout transitions.
  const prevUserIdRef = useRef<string>(initUserId);

  // ── User isolation: reload chats whenever the logged-in user changes ────────
  // This fires when:
  //   • The app first loads and AuthProvider resolves the user from localStorage
  //   • A different user logs in (user.id changes)
  //   • The user logs out (user becomes null → userId becomes "guest")
  useEffect(() => {
    const userId = user?.id ?? "guest";
    if (userId === prevUserIdRef.current) return;   // no real change

    prevUserIdRef.current = userId;
    const freshChats = loadChatsForUser(userId);
    setChats(freshChats);
    setCurrentChatId(loadCurrentIdForUser(userId, freshChats));
  }, [user?.id]);

  // ── Persistence: write to user-scoped keys on every change ──────────────────
  useEffect(() => {
    const userId = user?.id ?? initUserId;
    lsSet(chatsKey(userId), JSON.stringify(chats));
  }, [chats, user?.id, initUserId]);

  useEffect(() => {
    const userId = user?.id ?? initUserId;
    lsSet(currentIdKey(userId), currentChatId);
  }, [currentChatId, user?.id, initUserId]);

  // ── Helpers ────────────────────────────────────────────────────────────────

  const getCurrentChat = useCallback(
    (id = currentChatId) => chats.find((c) => c.id === id),
    [chats, currentChatId]
  );

  const messages = getCurrentChat()?.messages ?? [];

  const persistChatUpdate = useCallback(
    (chatId: string, updater: (c: Chat) => Chat) => {
      setChats((prev) => prev.map((c) => (c.id === chatId ? updater(c) : c)));
    },
    []
  );

  const updateChatTitle = useCallback(
    (chatId: string, title: string) => {
      persistChatUpdate(chatId, (c) => ({ ...c, title: title || c.title }));
    },
    [persistChatUpdate]
  );

  const addMessage = useCallback(
    (m: ChatMessage) => {
      const msgWithMeta: ChatMessage = {
        ...m,
        id: m.id ?? makeId(m.role === "user" ? "u-" : "a-"),
        timestamp: m.timestamp ?? Date.now(),
      };
      persistChatUpdate(currentChatId, (c) => ({
        ...c,
        messages: [...c.messages, msgWithMeta],
      }));
    },
    [currentChatId, persistChatUpdate]
  );

  const startNewChat = useCallback(() => {
    const id = makeId("chat-");
    setChats((s) => [
      { id, title: "New Chat", timestamp: new Date().toLocaleString(), messages: [] },
      ...s,
    ]);
    setCurrentChatId(id);
    return id;
  }, []);

  const loadChat = useCallback(
    (id: string) => {
      if (!chats.find((c) => c.id === id)) return;
      setCurrentChatId(id);
    },
    [chats]
  );

  // ── Delete chat ────────────────────────────────────────────────────────────

  const deleteChat = useCallback(
    (chatId: string) => {
      setChats((prev) => {
        const next = prev.filter((c) => c.id !== chatId);
        if (next.length === 0) {
          // Always keep at least one chat so the UI doesn't go blank
          const newId = makeId("chat-");
          const newChat: Chat = {
            id: newId,
            title: "New Chat",
            timestamp: new Date().toLocaleString(),
            messages: [],
          };
          setCurrentChatId(newId);
          return [newChat];
        }
        // If the deleted chat was active, switch to the first remaining chat
        if (chatId === currentChatId) {
          setCurrentChatId(next[0].id);
        }
        return next;
      });

      // Fire-and-forget backend cleanup — removes in-memory session context.
      // Non-critical: if the request fails the chat is already gone from the UI.
      fetch(`${API_BASE}/chat/${chatId}`, { method: "DELETE" }).catch(() => {});
    },
    [currentChatId]
  );

  // ── Core send ──────────────────────────────────────────────────────────────

  const sendUserMessage = useCallback(
    async (content: string): Promise<string> => {
      const currentChat = getCurrentChat();
      const isFirstMessage = currentChat !== undefined && currentChat.messages.length === 0;

      addMessage({ role: "user", content, timestamp: Date.now() });

      if (isFirstMessage) {
        const chatIdForTitle = currentChatId;
        fetch(`${API_BASE}/generate-title`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: content }),
        })
          .then((r) => r.json())
          .then((data: { title?: string }) => {
            const llmTitle = data?.title?.trim();
            updateChatTitle(chatIdForTitle, llmTitle || generateChatTitle(content));
          })
          .catch(() => {
            updateChatTitle(chatIdForTitle, generateChatTitle(content));
          });
      }

      setIsLoading(true);

      // Ticker injection — ensures backend reliably detects the instrument
      let outgoing = content;
      const lower = content.toLowerCase();
      for (const [name, ticker] of Object.entries(INSTRUMENT_TICKER_MAP)) {
        if (lower.includes(name)) {
          if (!outgoing.includes(ticker)) outgoing = `${outgoing} ${ticker}`;
          break;
        }
      }

      const targetChatId = currentChatId;
      const assistantMsgId = makeId("a-");

      persistChatUpdate(targetChatId, (c) => ({
        ...c,
        messages: [
          ...c.messages,
          { id: assistantMsgId, role: "assistant", content: "", timestamp: Date.now() },
        ],
      }));
      setStreamingMessageId(assistantMsgId);

      try {
        const rawResponse = await sendMessageStream(
          backendChatId,
          outgoing,
          (token) => {
            setChats((prev) =>
              prev.map((c) => {
                if (c.id !== targetChatId) return c;
                return {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === assistantMsgId
                      ? { ...m, content: m.content + token }
                      : m
                  ),
                };
              })
            );
          }
        );

        const finalContent = postProcessResponse(rawResponse);

        // ── Task 1 & 2: Parse structured fields from this message's SSE content ──
        // Each call creates a FRESH object from THIS response's text — no shared
        // references, no mutation, no stale data from previous messages.
        const fupMatch     = finalContent.match(/^Follow-up:\s*(.+)$/m);
        const followupText = fupMatch ? fupMatch[1].trim() : undefined;
        const evidenceData = parseEvidenceDictFromContent(finalContent);  // Task 2
        const insightData  = parseInsightDictFromContent(finalContent);   // Task 4

        // ── Debug logs — strict binding verification ──────────────────────────
        // RAW BACKEND: what parseEvidenceDictFromContent extracted from THIS message's
        // SSE stream text.  If this says "Yahoo Finance", the backend sent "Yahoo Finance"
        // — the frontend never injects or overrides evidence values.
        // RENDERED: (chat-message.tsx) must match this exactly. Any difference means a
        // stale prop reference reached the component — investigate setChats update path.
        console.log("RAW BACKEND:", evidenceData ?? null);
        console.log("BACKEND:", {
          contentPreview: finalContent.slice(0, 200),
          insight:        insightData  ?? null,
          followup:       followupText ?? null,
          status:         "ok",
        });

        // ── Task 1 & 7: Single immutable update — spread creates new object ──
        // No field is mutated; every key is either spread from `m` (immutable
        // string / primitive) or a fresh object/value built above.
        setChats((prev) =>
          prev.map((c) => {
            if (c.id !== targetChatId) return c;
            return {
              ...c,
              messages: c.messages.map((m) =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      content:  finalContent,                              // post-processed text
                      status:   "ok",                                       // streaming path → LLM always ran
                      // Task 5: explicit spread — guarantees a fresh object reference
                      // on every update so React always sees a new value, preventing
                      // stale renders from object identity equality checks.
                      evidence: evidenceData ? { ...evidenceData } : undefined,
                      insight:  insightData  ? { ...insightData  } : undefined,
                      followup: followupText,                               // Task 5: preserved, not changed
                    }
                  : m
              ),
            };
          })
        );

        setLastFollowUp(followupText ?? null);

        return finalContent;
      } catch (err) {
        // Task 7: Show a friendly server-connectivity message instead of the raw
        // "Failed to fetch" browser error.  Log the real error to console for debugging.
        console.error("[ChatContext] sendUserMessage error:", err);
        const isConnectionError =
          err instanceof TypeError && err.message === "Failed to fetch";
        const errText = isConnectionError
          ? "⚠️ Unable to connect to server\nPlease ensure backend is running on port 8000"
          : err instanceof Error
          ? `⚠️ Error: ${err.message}`
          : String(err);

        setChats((prev) =>
          prev.map((c) => {
            if (c.id !== targetChatId) return c;
            return {
              ...c,
              messages: c.messages.map((m) =>
                m.id === assistantMsgId ? { ...m, content: errText } : m
              ),
            };
          })
        );
        return errText;
      } finally {
        setStreamingMessageId(null);
        setIsLoading(false);
      }
    },
    [
      getCurrentChat,
      currentChatId,
      addMessage,
      updateChatTitle,
      persistChatUpdate,
      backendChatId,
    ]
  );

  const sendHiddenMessage = useCallback(
    async (content: string): Promise<string> => {
      setIsLoading(true);
      try {
        const response = await apiSendMessage(backendChatId, content);
        return postProcessResponse(response);
      } catch (err) {
        console.error("[ChatContext] sendHiddenMessage error:", err);
        const isConnectionError =
          err instanceof TypeError && err.message === "Failed to fetch";
        const errText = isConnectionError
          ? "⚠️ Unable to connect to server\nPlease ensure backend is running on port 8000"
          : err instanceof Error
          ? `⚠️ Error: ${err.message}`
          : String(err);
        addMessage({ role: "assistant", content: errText, timestamp: Date.now() });
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [backendChatId, addMessage]
  );

  const sendMessage = useCallback(
    async (m: ChatMessage): Promise<string> => {
      if (m.role !== "user") { addMessage(m); return ""; }
      return sendUserMessage(m.content);
    },
    [addMessage, sendUserMessage]
  );

  // ── Context value ──────────────────────────────────────────────────────────

  const value: ChatContextValue = {
    chats,
    currentChatId,
    messages,
    isLoading,
    streamingMessageId,
    addMessage,
    sendUserMessage,
    sendMessage,
    sendHiddenMessage,
    setLoading: setIsLoading,
    startNewChat,
    loadChat,
    updateChatTitle,
    deleteChat,
    awaitingExport,
    setAwaitingExport,
    lastFollowUp,
    setLastFollowUp,
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export const useChat = () => {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChat must be used inside <ChatProvider>");
  return ctx;
};
