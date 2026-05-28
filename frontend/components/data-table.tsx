"use client";

import React, { useMemo, useState } from "react";

interface DataRow {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface DataTableProps {
  data: DataRow[];
  instrument: string;
  timeframe: string;
}

export function DataTable({ data, instrument, timeframe }: DataTableProps) {
  const [sortKey, setSortKey] = useState<keyof DataRow | "date">("date");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [pageSize, setPageSize] = useState<number>(10);
  const [page, setPage] = useState<number>(1);

  const sorted = useMemo(() => {
    const copy = data.slice();
    copy.sort((a: any, b: any) => {
      let va: any = a[sortKey];
      let vb: any = b[sortKey];
      if (sortKey === "date") {
        va = new Date(a.date).getTime() || 0;
        vb = new Date(b.date).getTime() || 0;
      }
      if (va < vb) return sortDir === "asc" ? -1 : 1;
      if (va > vb) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return copy;
  }, [data, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const pageData = useMemo(() => {
    const start = (page - 1) * pageSize;
    return sorted.slice(start, start + pageSize);
  }, [sorted, page, pageSize]);

  const toggleSort = (key: keyof DataRow | "date") => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
    setPage(1);
  };

  return (
    <div className="bg-card rounded-lg border border-border overflow-hidden">
      <div className="px-4 py-3 bg-muted border-b border-border flex items-center justify-between">
        <h3 className="font-semibold text-sm">
          {instrument} • {timeframe} • {data.length} rows
        </h3>
        <div className="flex items-center gap-3">
          <label className="text-xs text-muted-foreground">Show</label>
          <select
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setPage(1);
            }}
            className="bg-card border border-border rounded px-2 py-1 text-sm"
          >
            <option value={5}>5</option>
            <option value={10}>10</option>
            <option value={25}>25</option>
            <option value={50}>50</option>
          </select>
        </div>
      </div>

      <div className="overflow-auto">
        <table className="w-full text-sm table-fixed">
          <thead>
            <tr className="border-b border-border bg-muted/50">
              <th
                className="px-4 py-2 text-left font-semibold text-muted-foreground cursor-pointer"
                onClick={() => toggleSort("date")}
              >
                Date {sortKey === "date" ? (sortDir === "asc" ? "↑" : "↓") : ""}
              </th>
              <th
                className="px-4 py-2 text-right font-semibold text-muted-foreground cursor-pointer"
                onClick={() => toggleSort("open")}
              >
                Open {sortKey === "open" ? (sortDir === "asc" ? "↑" : "↓") : ""}
              </th>
              <th
                className="px-4 py-2 text-right font-semibold text-muted-foreground cursor-pointer"
                onClick={() => toggleSort("high")}
              >
                High {sortKey === "high" ? (sortDir === "asc" ? "↑" : "↓") : ""}
              </th>
              <th
                className="px-4 py-2 text-right font-semibold text-muted-foreground cursor-pointer"
                onClick={() => toggleSort("low")}
              >
                Low {sortKey === "low" ? (sortDir === "asc" ? "↑" : "↓") : ""}
              </th>
              <th
                className="px-4 py-2 text-right font-semibold text-muted-foreground cursor-pointer"
                onClick={() => toggleSort("close")}
              >
                Close{" "}
                {sortKey === "close" ? (sortDir === "asc" ? "↑" : "↓") : ""}
              </th>
              <th
                className="px-4 py-2 text-right font-semibold text-muted-foreground cursor-pointer"
                onClick={() => toggleSort("volume")}
              >
                Volume{" "}
                {sortKey === "volume" ? (sortDir === "asc" ? "↑" : "↓") : ""}
              </th>
            </tr>
          </thead>

          <tbody>
            {pageData.map((row, idx) => (
              <tr
                key={idx}
                className="border-b border-border hover:bg-muted/30 transition-colors"
              >
                <td className="px-4 py-3 text-foreground">{row.date}</td>
                <td className="px-4 py-3 text-right text-foreground">
                  {formatNumber(row.open)}
                </td>
                <td className="px-4 py-3 text-right text-foreground">
                  {formatNumber(row.high)}
                </td>
                <td className="px-4 py-3 text-right text-foreground">
                  {formatNumber(row.low)}
                </td>
                <td className="px-4 py-3 text-right font-medium text-primary">
                  {formatNumber(row.close)}
                </td>
                <td className="px-4 py-3 text-right text-muted-foreground">
                  {formatVolume(row.volume)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="px-4 py-3 bg-muted border-t border-border flex items-center justify-between">
        <div className="text-xs text-muted-foreground">
          Showing {Math.min((page - 1) * pageSize + 1, data.length)} -{" "}
          {Math.min(page * pageSize, data.length)} of {data.length}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            className="px-2 py-1 border border-border rounded disabled:opacity-50"
            disabled={page <= 1}
          >
            Prev
          </button>
          <div className="text-sm">
            {page} / {totalPages}
          </div>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            className="px-2 py-1 border border-border rounded disabled:opacity-50"
            disabled={page >= totalPages}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}

function formatNumber(n: number) {
  if (n == null || Number.isNaN(n)) return "-";
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function formatVolume(v: number) {
  if (v == null || Number.isNaN(v)) return "-";
  if (Math.abs(v) >= 1_000_000) return (v / 1_000_000).toFixed(1) + "M";
  if (Math.abs(v) >= 1_000) return (v / 1_000).toFixed(1) + "K";
  return v.toString();
}
