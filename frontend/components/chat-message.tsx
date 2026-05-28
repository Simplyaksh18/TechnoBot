"use client";

import React, { useMemo } from "react";
import { Pin, ArrowRight, Telescope, ChevronDown } from "lucide-react";

// ─── Types ───────────────────────────────────────────────────────────────────

export interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  onFollowUp?: (query: string) => void;
  /** Backend response status — "partial" triggers a qualitative-mode badge. */
  status?: "ok" | "partial" | "unavailable";
  /**
   * Task 2 — structured evidence dict parsed from this message's SSE text.
   * When present (and has `ticker`), rendering uses this dict directly so
   * evidence is ALWAYS bound to THIS message's backend response — never stale.
   */
  evidence?: Record<string, string>;
  /**
   * Task 4 — structured insight dict parsed from this message's SSE text.
   * When present (and has `regime`), rendering uses this dict directly.
   */
  insight?: Record<string, string>;
}

type BlockType =
  | "chart"
  | "comparison"
  | "evidence"
  | "export_cta"
  | "followup"
  | "bullets"
  | "broader_context"
  | "contextual_insight"
  | "prose";

interface Block {
  type: BlockType;
  raw: string;
}

// ─── Constants ───────────────────────────────────────────────────────────────

const SENTINEL_CHART = /^__CHART__(.+):(.+)$/;
const SENTINEL_COMPARISON = /^\*\*COMPARISON\*\*(\[[\s\S]+\])$/;
const SENTINEL_EVIDENCE =
  /^(Evidence Summary|Evidence summary|📊 Evidence)[:\s]/i;
const SENTINEL_EXPORT_CTA = /^(Export|Download|Get report)/i;
const SENTINEL_FOLLOWUP = /^(Follow.?up[:\s]|💬|Q:|Question:)/i;
const SENTINEL_BROADER = /^🔭\s*Broader Context/i;
const SENTINEL_INSIGHT = /^📌\s*Contextual Insight/i;
const SENTINEL_BULLET = /^[-•*]\s+/;

// ─── Block Classifier ────────────────────────────────────────────────────────

function classifyLine(line: string): BlockType {
  if (SENTINEL_CHART.test(line)) return "chart";
  if (SENTINEL_COMPARISON.test(line)) return "comparison";
  if (SENTINEL_EVIDENCE.test(line)) return "evidence";
  if (SENTINEL_EXPORT_CTA.test(line)) return "export_cta";
  if (SENTINEL_FOLLOWUP.test(line)) return "followup";
  if (SENTINEL_BROADER.test(line)) return "broader_context";
  if (SENTINEL_INSIGHT.test(line)) return "contextual_insight";
  if (SENTINEL_BULLET.test(line)) return "bullets";
  return "prose";
}

const STOP_TYPES = new Set([
  "chart",
  "comparison",
  "followup",
  "broader_context",
  "contextual_insight",
]);

function parseBlocks(content: string): Block[] {
  const lines = content.split("\n");
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i].trim();
    if (!line) {
      i++;
      continue;
    }

    const type = classifyLine(line);

    // ── Multi-line accumulators ──────────────────────────────────────────────
    if (type === "evidence") {
      let raw = line;
      i++;
      while (
        i < lines.length &&
        lines[i].trim() &&
        !STOP_TYPES.has(classifyLine(lines[i].trim()))
      ) {
        raw += "\n" + lines[i];
        i++;
      }
      blocks.push({ type: "evidence", raw });
      continue;
    }

    if (type === "broader_context") {
      let raw = line;
      i++;
      while (
        i < lines.length &&
        lines[i].trim() &&
        !STOP_TYPES.has(classifyLine(lines[i].trim())) &&
        classifyLine(lines[i].trim()) !== "evidence"
      ) {
        raw += "\n" + lines[i];
        i++;
      }
      blocks.push({ type: "broader_context", raw });
      continue;
    }

    if (type === "contextual_insight") {
      let raw = line;
      i++;
      while (
        i < lines.length &&
        lines[i].trim() &&
        !STOP_TYPES.has(classifyLine(lines[i].trim())) &&
        classifyLine(lines[i].trim()) !== "evidence"
      ) {
        raw += "\n" + lines[i];
        i++;
      }
      blocks.push({ type: "contextual_insight", raw });
      continue;
    }

    if (type === "bullets") {
      let raw = line;
      i++;
      while (i < lines.length && SENTINEL_BULLET.test(lines[i])) {
        raw += "\n" + lines[i];
        i++;
      }
      blocks.push({ type: "bullets", raw });
      continue;
    }

    if (type === "prose") {
      let raw = line;
      i++;
      while (
        i < lines.length &&
        classifyLine(lines[i].trim()) === "prose" &&
        lines[i].trim()
      ) {
        raw += "\n" + lines[i];
        i++;
      }
      blocks.push({ type: "prose", raw });
      continue;
    }

    blocks.push({ type, raw: line });
    i++;
  }

  return blocks;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function BulletItem({ text }: { text: string }) {
  const clean = text.replace(SENTINEL_BULLET, "").trim();
  return (
    <li className="flex items-start gap-2 text-sm leading-relaxed">
      <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-primary shrink-0" />
      <span>{clean}</span>
    </li>
  );
}

/** Linkify a single text node: returns mixed string + <a> elements. */
function linkifyLine(text: string): React.ReactNode {
  const urlRegex = /(https?:\/\/[^\s,)"']+)/g;
  const parts = text.split(urlRegex);
  return parts.map((part, i) =>
    urlRegex.test(part) ? (
      <a
        key={i}
        href={part}
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-500 dark:text-blue-400 underline underline-offset-2 hover:opacity-80 transition-opacity break-all"
      >
        {part}
      </a>
    ) : (
      <React.Fragment key={i}>{part}</React.Fragment>
    ),
  );
}

function EvidenceBlock({ raw }: { raw: string }) {
  const evidenceLines = raw.split("\n");
  const header = evidenceLines[0];
  const bodyLines = evidenceLines.slice(1).filter((l) => l.trim());

  return (
    <div className="mt-8 border rounded-lg bg-muted/50 text-sm shadow-sm overflow-hidden">
      <details open className="group">
        <summary className="cursor-pointer font-medium px-4 py-3 flex items-center justify-between select-none list-none">
          <span>📊 {header}</span>
          <ChevronDown
            size={14}
            className="transition-transform group-open:rotate-180 text-muted-foreground"
          />
        </summary>
        {bodyLines.length > 0 && (
          <div className="px-4 pb-4 pt-1 text-muted-foreground leading-relaxed border-t border-border">
            {bodyLines.map((line, i) => (
              <div key={i} className="py-0.5">
                {linkifyLine(line)}
              </div>
            ))}
          </div>
        )}
      </details>
    </div>
  );
}

function BroaderContextBlock({ raw }: { raw: string }) {
  // Strip the "🔭 Broader Context" header prefix; body follows on next line
  // or is embedded after a colon on the same line
  const firstNewline = raw.indexOf("\n");
  let body: string;
  if (firstNewline !== -1) {
    body = raw.slice(firstNewline + 1).trim();
  } else {
    // single-line: strip the sentinel prefix
    body = raw.replace(/^🔭\s*Broader Context[:\s]*/i, "").trim();
  }

  return (
    <div className="mt-6 p-3 rounded-lg bg-purple-50 dark:bg-purple-500/10 border border-purple-200 dark:border-purple-500/30 space-y-1">
      <div className="flex items-center gap-1.5 text-xs font-semibold text-purple-700 dark:text-purple-400 uppercase tracking-wide">
        <Telescope size={13} />
        Broader Context
      </div>
      <p className="text-sm text-purple-900 dark:text-purple-200/90 leading-relaxed">
        {body}
      </p>
    </div>
  );
}

function ContextualInsightBlock({ raw }: { raw: string }) {
  // Strip the "📌 Contextual Insight:" sentinel from the first line
  const body = raw.replace(/^📌\s*Contextual Insight[:\s]*/i, "").trim();

  // Detect multi-line bullet format (• Current regime / • Pattern / • Implication)
  const bulletLines = body
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.startsWith("•") || l.startsWith("-") || l.startsWith("*"));

  const isBulletFormat = bulletLines.length >= 2;

  return (
    <div className="mt-6 p-4 rounded-lg bg-amber-50 dark:bg-yellow-500/10 border border-amber-200 dark:border-yellow-500/30">
      <div className="flex items-start gap-2.5">
        <Pin
          size={14}
          className="mt-1 text-amber-600 dark:text-yellow-400 shrink-0"
        />
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-amber-700 dark:text-yellow-400 uppercase tracking-wide mb-2">
            Contextual Insight
          </p>
          {isBulletFormat ? (
            // Render structured bullet insight
            <ul className="space-y-1.5">
              {bulletLines.map((line, i) => {
                const clean = line.replace(/^[•\-*]\s*/, "").trim();
                // Bold the label before the colon (e.g. "Current regime:", "Pattern:", "Implication:")
                const colonIdx = clean.indexOf(":");
                const label =
                  colonIdx !== -1 ? clean.slice(0, colonIdx + 1) : null;
                const rest =
                  colonIdx !== -1 ? clean.slice(colonIdx + 1).trim() : clean;
                return (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-sm text-amber-900 dark:text-yellow-100/90 leading-relaxed"
                  >
                    <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-amber-500 dark:bg-yellow-400 shrink-0" />
                    <span>
                      {label && (
                        <span className="font-semibold text-amber-800 dark:text-yellow-300">
                          {label}{" "}
                        </span>
                      )}
                      {rest}
                    </span>
                  </li>
                );
              })}
            </ul>
          ) : (
            // Legacy prose format fallback
            <p className="text-sm italic text-amber-900 dark:text-yellow-100/90 leading-relaxed">
              {body}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function FollowUpBlock({
  raw,
  onFollowUp,
}: {
  raw: string;
  onFollowUp?: (q: string) => void;
}) {
  const question = raw
    .replace(/^(Follow.?up[:\s]|💬|Q:|Question:)/i, "")
    .trim();

  return (
    <div className="mt-6">
      <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wide font-medium">
        Suggested follow-up
      </p>
      <button
        onClick={() => onFollowUp?.(question)}
        className="px-4 py-2 border border-border rounded-md text-sm flex items-center gap-2 hover:bg-muted hover:scale-[1.01] transition-all text-foreground"
      >
        <ArrowRight size={14} className="text-primary shrink-0" />
        {question}
      </button>
    </div>
  );
}

function ChartBlock({ raw }: { raw: string }) {
  const match = raw.match(SENTINEL_CHART);
  const ticker = match?.[1] ?? "";
  const timeframe = match?.[2] ?? "";

  return (
    <div className="mt-6 rounded-lg border border-border overflow-hidden">
      {/* Swap this div for <PriceLineChart ticker={ticker} timeframe={timeframe} /> */}
      <div className="h-48 bg-muted/30 flex items-center justify-center text-muted-foreground text-sm">
        📈 Chart: {ticker} · {timeframe}
      </div>
    </div>
  );
}

function ComparisonBlock({ raw }: { raw: string }) {
  const match = raw.match(SENTINEL_COMPARISON);
  if (!match) return <pre className="text-sm overflow-x-auto">{raw}</pre>;

  let rows: Record<string, string>[] = [];
  try {
    rows = JSON.parse(match[1]) as Record<string, string>[];
  } catch {
    return <pre className="text-sm overflow-x-auto">{raw}</pre>;
  }

  if (!rows.length) return null;
  const headers = Object.keys(rows[0]);

  return (
    <div className="mt-6 overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted/60">
          <tr>
            {headers.map((h) => (
              <th
                key={h}
                className="px-4 py-2 text-left font-semibold text-muted-foreground whitespace-nowrap"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIdx) => (
            <tr
              key={rowIdx}
              className="border-t border-border hover:bg-muted/20 transition-colors"
            >
              {headers.map((h) => (
                <td key={h} className="px-4 py-2 align-top">
                  {String(row[h] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExportCtaBlock({ raw }: { raw: string }) {
  return (
    <div className="mt-6">
      <button className="px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm font-medium hover:opacity-90 transition-opacity">
        {raw}
      </button>
    </div>
  );
}

function ProseBlock({ raw }: { raw: string }) {
  return (
    <p className="text-sm leading-relaxed text-foreground/90 whitespace-pre-wrap">
      {raw}
    </p>
  );
}

// ─── Structured (dict-driven) block components ────────────────────────────────
//
// Task 2 & 4: These render from the per-message structured dicts instead of
// from the raw text content.  This guarantees the displayed ticker / regime
// always matches THIS message's backend response — no stale data leakage.

/**
 * Task 2 — StructuredEvidenceBlock
 * Renders evidence from the typed dict.  Only shown when `evidence.ticker` is
 * present (Task 3 conditional rendering guard).
 */
function StructuredEvidenceBlock({
  evidence,
}: {
  evidence: Record<string, string>;
}) {
  // Display EXACTLY 5 canonical fields from the backend — nothing more, nothing less.
  //
  //   source     → what the backend set (e.g. "TwelveData", "Alpha Vantage",
  //                "Qualitative fallback").  Frontend NEVER overrides this.
  //   ticker     → the detected instrument symbol exactly as the backend resolved it.
  //   instrument → human-readable instrument name.
  //   timeframe  → the label the backend produced (e.g. "Weekly · last 2 years").
  //                Frontend does NOT compute, derive, or replace this value.
  //   link       → evidence.reference  (text-stream path: "• Reference: https://…")
  //             OR evidence.link       (structured /chat path: { link: "https://…" })
  //
  // Fields that are INTENTIONALLY excluded from display:
  //   data_range  ("~125 sessions") — backend-generated approximation, not user-facing
  //   last_close  — price display is handled by the analysis bullets, not evidence card
  //   note / data_note / data_used — internal backend annotations
  //
  // All four text fields use `|| "N/A"` so a missing backend value surfaces explicitly
  // rather than silently rendering nothing (which looks identical to a hidden field).

  // link: handles both key names produced by the two backend paths
  const link = evidence.reference || evidence.link;

  return (
    <div className="mt-8 border rounded-lg bg-muted/50 text-sm shadow-sm overflow-hidden">
      <details open className="group">
        <summary className="cursor-pointer font-medium px-4 py-3 flex items-center justify-between select-none list-none">
          <span>📊 Evidence Summary</span>
          <ChevronDown
            size={14}
            className="transition-transform group-open:rotate-180 text-muted-foreground"
          />
        </summary>
        <div className="px-4 pb-4 pt-1 text-muted-foreground leading-relaxed border-t border-border">
          {/* Source — exact backend string, no default injected */}
          <div className="py-0.5">
            <span className="font-medium text-foreground/70">Source:</span>{" "}
            {evidence.source || "N/A"}
          </div>

          {/* Instrument — human-readable name */}
          <div className="py-0.5">
            <span className="font-medium text-foreground/70">Instrument:</span>{" "}
            {evidence.instrument || "N/A"}
          </div>

          {/* Timeframe — backend value only; frontend never derives or replaces this */}
          <div className="py-0.5">
            <span className="font-medium text-foreground/70">Timeframe:</span>{" "}
            {evidence.timeframe || "N/A"}
          </div>

          {evidence.requested_scope && (
            <div className="py-0.5">
              <span className="font-medium text-foreground/70">
                Requested scope:
              </span>{" "}
              {evidence.requested_scope}
            </div>
          )}

          {evidence.rows && (
            <div className="py-0.5">
              <span className="font-medium text-foreground/70">Rows:</span>{" "}
              {evidence.rows}
            </div>
          )}

          {evidence.data_coverage && (
            <div className="py-0.5">
              <span className="font-medium text-foreground/70">
                Data coverage:
              </span>{" "}
              {evidence.data_coverage}
            </div>
          )}

          {/* Ticker link — backend URL is preserved as-is */}
          {link && (
            <div className="py-0.5 mt-1">
              <span className="font-medium text-foreground/70">Ticker:</span>{" "}
              {evidence.ticker || evidence.instrument || "N/A"}
              <div className="mt-0.5">
                <span className="font-medium text-foreground/70">Link :</span>{" "}
                <a
                  href={link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-blue-500 dark:text-blue-400 underline underline-offset-2 hover:opacity-80 transition-opacity"
                >
                  {evidence.ticker || evidence.instrument || "Open ticker"}
                </a>
              </div>
            </div>
          )}
        </div>
      </details>
    </div>
  );
}

/**
 * Task 4 — StructuredInsightBlock
 * Renders insight from the typed dict.  Only shown when `insight.regime` is
 * present (Task 4 conditional rendering guard).
 */
function StructuredInsightBlock({
  insight,
}: {
  insight: Record<string, string>;
}) {
  const isComparative = insight.regime === "comparative";

  const bullets: Array<[string, string]> = (
    [
      ["Current regime", insight.regime],
      ["Pattern", insight.pattern],
      ["Implication", insight.implication],
    ] as Array<[string, string]>
  ).filter(([, v]) => !!v && v !== "comparative");

  return (
    <div className="mt-6 p-4 rounded-lg bg-amber-50 dark:bg-yellow-500/10 border border-amber-200 dark:border-yellow-500/30">
      <div className="flex items-start gap-2.5">
        <Pin
          size={14}
          className="mt-1 text-amber-600 dark:text-yellow-400 shrink-0"
        />
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-amber-700 dark:text-yellow-400 uppercase tracking-wide mb-2">
            Contextual Insight
          </p>
          {isComparative ? (
            <p className="text-sm italic text-amber-900 dark:text-yellow-100/90 leading-relaxed">
              {insight.pattern}
            </p>
          ) : (
            <ul className="space-y-1.5">
              {bullets.map(([label, value]) => (
                <li
                  key={label}
                  className="flex items-start gap-2 text-sm text-amber-900 dark:text-yellow-100/90 leading-relaxed"
                >
                  <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-amber-500 dark:bg-yellow-400 shrink-0" />
                  <span>
                    <span className="font-semibold text-amber-800 dark:text-yellow-300">
                      {label}:{" "}
                    </span>
                    {value}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Fallback warning ─────────────────────────────────────────────────────────

/**
 * Task 2 — FallbackWarningBlock
 * Shown in place of the evidence card when the backend returned a qualitative
 * fallback response (source includes "Qualitative").  Prevents fake structured
 * market data from appearing when live feeds were unavailable.
 */
function FallbackWarningBlock() {
  return (
    <div className="mt-6 flex items-start gap-2.5 px-4 py-3 rounded-lg border border-amber-200 dark:border-amber-500/30 bg-amber-50/60 dark:bg-amber-500/10 text-sm text-amber-800 dark:text-amber-300">
      <span className="shrink-0 leading-none mt-0.5 text-base">⚠️</span>
      <span className="leading-relaxed">
        Live market data unavailable — showing qualitative analysis only.
        Results may not reflect current price action or real OHLCV indicators.
      </span>
    </div>
  );
}

function StreamingIndicator() {
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground animate-pulse mt-2">
      <span className="inline-flex gap-0.5">
        <span className="w-1 h-1 rounded-full bg-primary animate-bounce [animation-delay:0ms]" />
        <span className="w-1 h-1 rounded-full bg-primary animate-bounce [animation-delay:150ms]" />
        <span className="w-1 h-1 rounded-full bg-primary animate-bounce [animation-delay:300ms]" />
      </span>
      Analysing structure…
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function ChatMessage({
  role,
  content,
  streaming = false,
  onFollowUp,
  status,
  evidence, // Task 2: structured evidence dict — bound to THIS message only
  insight, // Task 4: structured insight dict — bound to THIS message only
}: ChatMessageProps) {
  const blocks = useMemo(
    () => (role === "assistant" ? parseBlocks(content) : []),
    [content, role],
  );

  // ── User bubble ───────────────────────────────────────────────────────────
  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] px-4 py-2.5 rounded-2xl rounded-br-sm bg-primary text-primary-foreground text-sm leading-relaxed">
          {content}
        </div>
      </div>
    );
  }

  // ── Assistant bubble ──────────────────────────────────────────────────────

  // Task 1 — Fallback detector.
  // Backend sets evidence.source = "Qualitative fallback (live feed unavailable)"
  // when _lightweight=True.  Checking the source field is the single reliable
  // signal — status="partial" is equivalent but may lag during streaming.
  const isFallback = !!evidence?.source?.includes("Qualitative");

  // Task 6 / Step 5 — Debug logs (assistant messages only, after streaming completes).
  // "EVIDENCE RENDERED:" logs exactly what THIS message's component receives from
  // state — use this to verify source/ticker/timeframe match the backend's response.
  if (!streaming) {
    console.log("RENDER:", {
      isFallback,
      status,
      contentPreview: content.slice(0, 120),
    });
    // Strict binding check: compare "RAW BACKEND:" (ChatContext) vs "RENDERED:" (here).
    // If UI shows "Yahoo Finance" → RAW BACKEND must also show { source: "Yahoo Finance" }
    // If not → frontend is overriding. Both logs must match exactly.
    console.log("RENDERED:", evidence ?? null);
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-180 w-full space-y-2">
        {/* Task 1: Partial-mode badge — shown when backend ran a qualitative LLM fallback */}
        {status === "partial" && !streaming && (
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-400/30 text-xs text-amber-600 dark:text-amber-400 mb-1">
            <span className="leading-none">⚡</span>
            <span>Qualitative analysis — live data unavailable</span>
          </div>
        )}

        {/* Strict backend binding — evidence and insight render ONLY from their props.   */}
        {/* No text fallback, no ticker guard, no defaults. If the prop isn't set,        */}
        {/* the block is suppressed entirely (returns null). This prevents any stale      */}
        {/* or hardcoded value from reaching the UI.                                      */}
        {(() => {
          // Insight guard: structured dict must have regime to render StructuredInsightBlock
          const hasStructuredInsight = !!insight?.regime;

          type OrderedItem = Block | Block[];
          const orderedItems: OrderedItem[] = [];
          let currentBullets: Block[] = [];

          for (const block of blocks) {
            if (block.type === "bullets") {
              currentBullets.push(block);
            } else {
              if (currentBullets.length) {
                orderedItems.push([...currentBullets]);
                currentBullets = [];
              }
              orderedItems.push(block);
            }
          }
          if (currentBullets.length) orderedItems.push([...currentBullets]);

          return orderedItems.map((item, idx) => {
            // Bullet group
            if (Array.isArray(item)) {
              const allLines = item.flatMap((b) =>
                b.raw.split("\n").filter((l) => SENTINEL_BULLET.test(l)),
              );
              return (
                <div key={idx} className="p-3 rounded-md bg-muted/30">
                  <ul className="space-y-2 pl-1">
                    {allLines.map((line, li) => (
                      <BulletItem key={li} text={line} />
                    ))}
                  </ul>
                </div>
              );
            }

            const block = item as Block;
            switch (block.type) {
              case "chart":
                return <ChartBlock key={idx} raw={block.raw} />;
              case "comparison":
                // Task 3: Suppress comparison tables in fallback mode — qualitative
                // responses contain no valid comparison data; rendering them would
                // present fabricated market structure.
                if (isFallback) return null;
                return <ComparisonBlock key={idx} raw={block.raw} />;
              case "evidence":
                // Strict backend binding:
                //   isFallback=true  → FallbackWarningBlock
                //   evidence prop set → StructuredEvidenceBlock (exact backend values)
                //   evidence prop absent → null (never fall back to raw text)
                //
                // The EvidenceBlock (raw text) path is intentionally removed.
                // Raw SSE text may contain stale source/timeframe from a previous
                // cache hit. Only the parsed, per-message evidence dict is shown.
                if (isFallback) {
                  return (
                    <React.Fragment key={idx}>
                      <div className="border-t border-border/40 mt-8 -mb-2" />
                      <FallbackWarningBlock />
                    </React.Fragment>
                  );
                }
                if (!evidence) return null;
                return (
                  <React.Fragment key={idx}>
                    <div className="border-t border-border/40 mt-8 -mb-2" />
                    <StructuredEvidenceBlock evidence={evidence} />
                  </React.Fragment>
                );
              case "export_cta":
                return <ExportCtaBlock key={idx} raw={block.raw} />;
              case "followup":
                return (
                  <React.Fragment key={idx}>
                    <div className="border-t border-border/40 mt-6 -mb-2" />
                    <FollowUpBlock
                      key={idx}
                      raw={block.raw}
                      onFollowUp={onFollowUp}
                    />
                  </React.Fragment>
                );
              case "broader_context":
                return <BroaderContextBlock key={idx} raw={block.raw} />;
              case "contextual_insight":
                // Task 4: Suppress insight in fallback mode — qualitative analysis
                // has no real OHLCV-derived regime data; showing it would mislead.
                // For real-data responses: prefer structured dict (regime-guarded),
                // fall back to text parsing for old persisted messages.
                if (isFallback) return null;
                return hasStructuredInsight ? (
                  <StructuredInsightBlock key={idx} insight={insight!} />
                ) : (
                  <ContextualInsightBlock key={idx} raw={block.raw} />
                );
              case "prose":
                return <ProseBlock key={idx} raw={block.raw} />;
              default:
                return null;
            }
          });
        })()}

        {/* Streaming dots */}
        {streaming && <StreamingIndicator />}
      </div>
    </div>
  );
}

export default ChatMessage;
