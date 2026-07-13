"use client";

import { ChangeEvent, DragEvent, useCallback, useEffect, useMemo, useState } from "react";

type Job = {
  id: string;
  status: "uploading" | "queued" | "processing" | "complete" | "failed";
  original_name: string;
  page: number;
  pages: number | null;
  row_count: number | null;
  error: string | null;
};

type Point = { x: number; y: number };
type Evidence = { page_number: number; polygon: { points: Point[] } };
type Row = {
  id: string;
  description: string | null;
  quantity: string | null;
  unit_price: string | null;
  discount: string | null;
  net_amount: string | null;
  review_disposition: string;
  page_number: number;
  evidence: Evidence[];
};
type PageAsset = { page_number: number; width: number; height: number };
type Result = { document_id: string; pages: number; page_assets: PageAsset[]; rows: Row[] };

const money = (value: string | null) =>
  value === null
    ? "—"
    : new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR" }).format(Number(value));

export default function Home() {
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [selected, setSelected] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const upload = useCallback(async (file: File) => {
    setError(null);
    setResult(null);
    setUploading(true);
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch("/api/v2/documents", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Upload failed");
      setJob(payload);
      localStorage.setItem("gmoney-demo-job", payload.id);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, []);

  useEffect(() => {
    const previous = localStorage.getItem("gmoney-demo-job");
    if (!previous) return;
    fetch(`/api/v2/documents/${previous}`)
      .then((response) => (response.ok ? response.json() : Promise.reject()))
      .then(setJob)
      .catch(() => localStorage.removeItem("gmoney-demo-job"));
  }, []);

  useEffect(() => {
    if (!job || job.status === "complete" || job.status === "failed") return;
    const poll = window.setInterval(async () => {
      const response = await fetch(`/api/v2/documents/${job.id}`);
      if (response.ok) setJob(await response.json());
    }, 1500);
    return () => window.clearInterval(poll);
  }, [job]);

  useEffect(() => {
    if (!job || job.status !== "complete" || result) return;
    fetch(`/api/v2/documents/${job.id}/rows`)
      .then((response) => response.json())
      .then((payload) => {
        setResult(payload);
        setSelected(0);
      })
      .catch(() => setError("The result could not be loaded."));
  }, [job, result]);

  const selectedRow = result?.rows[selected] ?? null;
  const pageNumber = selectedRow?.evidence[0]?.page_number ?? selectedRow?.page_number ?? 1;
  const pageAsset = result?.page_assets.find((asset) => asset.page_number === pageNumber);
  const polygon = selectedRow?.evidence[0]?.polygon.points ?? [];
  const polygonString = polygon.map((point) => `${point.x},${point.y}`).join(" ");
  const progress = useMemo(() => {
    if (!job) return 0;
    if (job.status === "complete") return 100;
    if (!job.pages) return job.status === "queued" ? 8 : 14;
    return Math.max(14, Math.round((job.page / job.pages) * 94));
  }, [job]);

  const accept = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) void upload(file);
  };
  const drop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) void upload(file);
  };

  return (
    <main>
      <header className="masthead">
        <div className="brand-mark">G</div>
        <div>
          <p className="eyebrow">Evidence studio · V2 preview</p>
          <h1>Read every charge.<br />See every source.</h1>
        </div>
        <div className="mast-status"><span /> D16 inference online</div>
      </header>

      {!result && (
        <section className="arrival">
          <div className="arrival-copy">
            <p className="folio">01 / Intake</p>
            <h2>A bill enters.<br /><em>A ledger emerges.</em></h2>
            <p className="lede">Upload a hospital bill PDF. The engine finds borderless tables, reconstructs rows, and anchors every accepted value to the source page.</p>
            <div className="proof-strip">
              <div><b>300</b><span>DPI evidence</span></div>
              <div><b>3×</b><span>parallel lanes</span></div>
              <div><b>6h</b><span>auto cleanup</span></div>
            </div>
          </div>
          <label
            className={`drop-zone ${dragging ? "is-dragging" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={drop}
          >
            <input type="file" accept="application/pdf,.pdf" onChange={accept} disabled={uploading} />
            <span className="drop-index">PDF</span>
            <span className="drop-cross">+</span>
            <strong>{uploading ? "Transferring…" : "Place bill here"}</strong>
            <small>or click to choose · 25 MiB maximum</small>
          </label>
        </section>
      )}

      {job && !result && (
        <section className="processing-card">
          <div className="processing-meta">
            <p className="folio">02 / Reconstruction</p>
            <span>{job.original_name}</span>
          </div>
          <div className="processing-number">{String(progress).padStart(2, "0")}<sup>%</sup></div>
          <div className="progress-track"><i style={{ width: `${progress}%` }} /></div>
          <div className="processing-foot">
            <strong>{job.status === "queued" ? "Waiting for an inference lane" : job.status === "failed" ? "Extraction stopped" : "Reading tables and grounding evidence"}</strong>
            <span>{job.pages ? `page ${job.page} of ${job.pages}` : "preparing pages"}</span>
          </div>
          {job.error && <p className="error-note">{job.error}</p>}
        </section>
      )}

      {result && job && (
        <section className="workspace">
          <div className="workspace-head">
            <div><p className="folio">03 / Evidence ledger</p><h2>{result.rows.length} billable rows</h2></div>
            <button className="new-bill" onClick={() => { setJob(null); setResult(null); localStorage.removeItem("gmoney-demo-job"); }}>New bill ↗</button>
          </div>
          <div className="split-view">
            <div className="ledger-pane">
              <div className="ledger-caption"><span>{job.original_name}</span><span>{result.pages} pages</span></div>
              <div className="table-shell">
                <table>
                  <thead><tr><th>#</th><th>Charge description</th><th>Qty</th><th>Rate</th><th>Net amount</th></tr></thead>
                  <tbody>
                    {result.rows.map((row, index) => (
                      <tr key={row.id} className={selected === index ? "selected" : ""} onClick={() => setSelected(index)}>
                        <td>{String(index + 1).padStart(2, "0")}</td>
                        <td><b>{row.description || "Unlabelled row"}</b><small><i className={row.review_disposition} /> {row.review_disposition} · p.{row.page_number}</small></td>
                        <td>{row.quantity ?? "—"}</td>
                        <td>{money(row.unit_price)}</td>
                        <td>{money(row.net_amount)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <aside className="evidence-pane">
              <div className="evidence-head"><span>Source page {pageNumber}</span><span className="evidence-key"><i /> linked evidence</span></div>
              <div className="page-frame">
                {/* The API serves the immutable 300-DPI artifact used by extraction. */}
                {/* eslint-disable-next-line @next/next/no-img-element -- immutable evidence is not optimized or cached by Next */}
                <img src={`/api/v2/documents/${job.id}/pages/${pageNumber}`} alt={`Rendered bill page ${pageNumber}`} />
                {pageAsset && polygonString && (
                  <svg viewBox={`0 0 ${pageAsset.width} ${pageAsset.height}`} preserveAspectRatio="none" aria-hidden="true">
                    <polygon points={polygonString} />
                  </svg>
                )}
              </div>
              {selectedRow && <div className="evidence-note"><span>{String(selected + 1).padStart(2, "0")}</span><div><strong>{selectedRow.description}</strong><p>{money(selectedRow.net_amount)} · grounded on page {pageNumber}</p></div></div>}
            </aside>
          </div>
        </section>
      )}

      {error && <div className="toast" role="alert">{error}<button onClick={() => setError(null)}>×</button></div>}
      <footer><span>GMoney / evidence-grounded extraction</span><span>Demo artifacts expire automatically</span></footer>
    </main>
  );
}
