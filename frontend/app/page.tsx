"use client";

import {
  ChangeEvent,
  DragEvent,
  PointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

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
type Evidence = {
  page_number: number;
  polygon: { points: Point[] };
  artifact_sha256: string;
};
type ReviewMeta = {
  source: "machine" | "reviewer";
  modified: boolean;
  reason: string | null;
  machine_values: Record<string, unknown>;
};
type Row = {
  id: string;
  description: string | null;
  section: string | null;
  quantity: string | null;
  unit_price: string | null;
  discount: string | null;
  net_amount: string | null;
  role: string;
  review_disposition: string;
  page_number: number;
  evidence: Evidence[];
  field_evidence: Record<string, Evidence[]>;
  review: ReviewMeta;
};
type PageAsset = {
  page_number: number;
  artifact_sha256: string;
  width: number;
  height: number;
};
type RowsResult = {
  document_id: string;
  pages: number;
  page_assets: PageAsset[];
  review_revision: number;
  total: number;
  offset: number;
  limit: number;
  rows: Row[];
};
type ReviewIssue = {
  id: string;
  page_number: number;
  table_id: string;
  table_type: string;
  reason_codes: string[];
  status: "open" | "resolved";
  resolution_reason: string | null;
};
type ReviewSummary = {
  revision: number;
  rows_total: number;
  rows_active: number;
  rows_modified: number;
  rows_pending: number;
  issues_open: number;
  issues: ReviewIssue[];
  approval: { status: string; approved_at: string; review_revision: number } | null;
};
type Health = { active_jobs: number; worker_capacity: number; queue_capacity: number };
type EditValues = {
  description: string;
  quantity: string;
  unit_price: string;
  discount: string;
  net_amount: string;
  role: string;
  review_disposition: string;
};

const JOBS_KEY = "gmoney-demo-jobs-v2";
const LEGACY_JOB_KEY = "gmoney-demo-job";
const PAGE_SIZE = 150;
const emptyEdit: EditValues = {
  description: "",
  quantity: "",
  unit_price: "",
  discount: "",
  net_amount: "",
  role: "detail",
  review_disposition: "accepted",
};

const money = (value: string | null) =>
  value === null || value === ""
    ? "—"
    : new Intl.NumberFormat("en-IN", {
        style: "currency",
        currency: "INR",
        maximumFractionDigits: 2,
      }).format(Number(value));

const detail = (payload: unknown) => {
  if (typeof payload === "string") return payload;
  if (payload && typeof payload === "object" && "detail" in payload) {
    const value = (payload as { detail: unknown }).detail;
    if (typeof value === "string") return value;
    if (value && typeof value === "object" && "code" in value) {
      return String((value as { code: unknown }).code).replaceAll("_", " ");
    }
  }
  return "The request could not be completed.";
};

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const payload = response.status === 204 ? null : await response.json().catch(() => null);
  if (!response.ok) throw new Error(detail(payload));
  return payload as T;
}

const points = (evidence: Evidence | undefined) =>
  evidence?.polygon.points.map((point) => `${point.x},${point.y}`).join(" ") ?? "";

export default function Home() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [rowsResult, setRowsResult] = useState<RowsResult | null>(null);
  const [review, setReview] = useState<ReviewSummary | null>(null);
  const [selectedRowId, setSelectedRowId] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [query, setQuery] = useState("");
  const [disposition, setDisposition] = useState("");
  const [pageFilter, setPageFilter] = useState("");
  const [offset, setOffset] = useState(0);
  const [viewPage, setViewPage] = useState(1);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editMode, setEditMode] = useState(false);
  const [editValues, setEditValues] = useState<EditValues>(emptyEdit);
  const [reason, setReason] = useState("");
  const [addMode, setAddMode] = useState(false);
  const [addDescription, setAddDescription] = useState("");
  const [addAmount, setAddAmount] = useState("");
  const [issueReason, setIssueReason] = useState("");
  const [drawMode, setDrawMode] = useState<"add" | "relink" | null>(null);
  const [draftPolygon, setDraftPolygon] = useState<Point[] | null>(null);
  const drawStart = useRef<Point | null>(null);
  const pageFrame = useRef<HTMLDivElement | null>(null);

  const selectedJob = jobs.find((job) => job.id === selectedJobId) ?? null;
  const selectedRow = rowsResult?.rows.find((row) => row.id === selectedRowId) ?? null;
  const pageAsset = rowsResult?.page_assets.find((asset) => asset.page_number === viewPage);
  const activeJobs = jobs.filter((job) => ["queued", "processing"].includes(job.status));

  const remember = useCallback((next: Job[]) => {
    const ids = next.map((job) => job.id).slice(0, 20);
    localStorage.setItem(JOBS_KEY, JSON.stringify(ids));
  }, []);

  const replaceJobs = useCallback(
    (updater: (current: Job[]) => Job[]) => {
      setJobs((current) => {
        const next = updater(current);
        remember(next);
        return next;
      });
    },
    [remember],
  );

  const refreshHealth = useCallback(() => {
    request<Health>("/api/v2/health/ready").then(setHealth).catch(() => setHealth(null));
  }, []);

  const refreshJobs = useCallback(async (ids?: string[]) => {
    const targets = ids ?? jobs.map((job) => job.id);
    if (!targets.length) return;
    const states = await Promise.all(
      targets.map((id) => request<Job>(`/api/v2/documents/${id}`).catch(() => null)),
    );
    const live = states.filter((state): state is Job => state !== null);
    setJobs(live);
    remember(live);
    setSelectedJobId((current) =>
      current && live.some((job) => job.id === current) ? current : (live[0]?.id ?? null),
    );
  }, [jobs, remember]);

  useEffect(() => {
    let ids: string[] = [];
    try {
      ids = JSON.parse(localStorage.getItem(JOBS_KEY) ?? "[]") as string[];
    } catch {
      localStorage.removeItem(JOBS_KEY);
    }
    const legacy = localStorage.getItem(LEGACY_JOB_KEY);
    if (legacy && !ids.includes(legacy)) ids.unshift(legacy);
    localStorage.removeItem(LEGACY_JOB_KEY);
    if (ids.length) void refreshJobs(ids);
    refreshHealth();
    // Initial hydration deliberately runs once using the persisted UUID set.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!activeJobs.length) return;
    const poll = window.setInterval(() => {
      void Promise.all(
        activeJobs.map((job) =>
          request<Job>(`/api/v2/documents/${job.id}`).then((state) => {
            replaceJobs((current) => current.map((item) => (item.id === state.id ? state : item)));
          }),
        ),
      );
      refreshHealth();
    }, 1500);
    return () => window.clearInterval(poll);
  }, [activeJobs, refreshHealth, replaceJobs]);

  const loadWorkspace = useCallback(async () => {
    if (!selectedJob || selectedJob.status !== "complete") return;
    const params = new URLSearchParams({ offset: String(offset), limit: String(PAGE_SIZE) });
    if (query.trim()) params.set("query", query.trim());
    if (disposition) params.set("disposition", disposition);
    if (pageFilter) params.set("source_page", pageFilter);
    try {
      const [rows, summary] = await Promise.all([
        request<RowsResult>(`/api/v2/documents/${selectedJob.id}/rows?${params}`),
        request<ReviewSummary>(`/api/v2/documents/${selectedJob.id}/review`),
      ]);
      setRowsResult(rows);
      setReview(summary);
      setSelectedRowId((current) =>
        current && rows.rows.some((row) => row.id === current)
          ? current
          : (rows.rows[0]?.id ?? null),
      );
      if (!rows.rows.length && offset > 0) setOffset(Math.max(0, offset - PAGE_SIZE));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The result could not be loaded.");
    }
  }, [selectedJob, offset, query, disposition, pageFilter]);

  useEffect(() => {
    setRowsResult(null);
    setReview(null);
    setOffset(0);
    setQuery("");
    setDisposition("");
    setPageFilter("");
    setEditMode(false);
    setAddMode(false);
  }, [selectedJobId]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadWorkspace(), 220);
    return () => window.clearTimeout(timer);
  }, [loadWorkspace]);

  useEffect(() => {
    if (selectedRow) setViewPage(selectedRow.page_number);
  }, [selectedRow]);

  const uploadFiles = useCallback(
    async (files: File[]) => {
      if (!files.length) return;
      setError(null);
      setUploading(true);
      try {
        const uploaded = await Promise.all(
          files.map(async (file) => {
            const body = new FormData();
            body.append("file", file);
            return request<Job>("/api/v2/documents", { method: "POST", body });
          }),
        );
        replaceJobs((current) => [...uploaded, ...current.filter((job) => !uploaded.some((item) => item.id === job.id))]);
        setSelectedJobId(uploaded[0].id);
        refreshHealth();
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [refreshHealth, replaceJobs],
  );

  const acceptFiles = (event: ChangeEvent<HTMLInputElement>) => {
    void uploadFiles(Array.from(event.target.files ?? []));
    event.target.value = "";
  };
  const drop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    void uploadFiles(Array.from(event.dataTransfer.files ?? []));
  };

  const beginEdit = () => {
    if (!selectedRow) return;
    setEditValues({
      description: selectedRow.description ?? "",
      quantity: selectedRow.quantity ?? "",
      unit_price: selectedRow.unit_price ?? "",
      discount: selectedRow.discount ?? "",
      net_amount: selectedRow.net_amount ?? "",
      role: selectedRow.role,
      review_disposition: selectedRow.review_disposition,
    });
    setReason(selectedRow.review.reason ?? "");
    setEditMode(true);
    setAddMode(false);
    setDraftPolygon(null);
    setDrawMode(null);
  };

  const mutate = useCallback(
    async <T,>(url: string, init: RequestInit): Promise<T | null> => {
      if (!review) return null;
      try {
        const value = await request<T>(url, {
          ...init,
          headers: {
            "Content-Type": "application/json",
            "If-Match": String(review.revision),
            ...init.headers,
          },
        });
        await loadWorkspace();
        return value;
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Review action failed");
        await loadWorkspace();
        return null;
      }
    },
    [loadWorkspace, review],
  );

  const saveEdit = async () => {
    if (!selectedJob || !selectedRow || reason.trim().length < 3) {
      setError("Add a short correction reason before saving.");
      return;
    }
    const payload: Record<string, unknown> = { changes: editValues, reason };
    if (draftPolygon) {
      payload.page_number = viewPage;
      payload.polygon = { points: draftPolygon };
    }
    const saved = await mutate(`/api/v2/documents/${selectedJob.id}/rows/${selectedRow.id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
    if (saved) {
      setEditMode(false);
      setDraftPolygon(null);
      setDrawMode(null);
    }
  };

  const rejectSelected = async () => {
    if (!selectedJob || !selectedRow || reason.trim().length < 3) {
      setError("Add a short rejection reason first.");
      return;
    }
    const rejected = await mutate(
      `/api/v2/documents/${selectedJob.id}/rows/${selectedRow.id}?reason=${encodeURIComponent(reason)}`,
      { method: "DELETE" },
    );
    if (rejected) setEditMode(false);
  };

  const addRow = async () => {
    if (!selectedJob || !draftPolygon || addDescription.trim().length < 2 || reason.trim().length < 3) {
      setError("Draw the source row and provide description, amount, and reason.");
      return;
    }
    const added = await mutate(`/api/v2/documents/${selectedJob.id}/rows`, {
      method: "POST",
      body: JSON.stringify({
        values: { description: addDescription, net_amount: addAmount },
        page_number: viewPage,
        polygon: { points: draftPolygon },
        reason,
      }),
    });
    if (added) {
      setAddMode(false);
      setAddDescription("");
      setAddAmount("");
      setReason("");
      setDraftPolygon(null);
      setDrawMode(null);
    }
  };

  const updateIssue = async (issue: ReviewIssue) => {
    if (!selectedJob || issueReason.trim().length < 3) {
      setError("Add a review note before changing an issue.");
      return;
    }
    const nextStatus = issue.status === "open" ? "resolved" : "open";
    const updated = await mutate(`/api/v2/documents/${selectedJob.id}/issues/${issue.id}`, {
      method: "PATCH",
      body: JSON.stringify({ status: nextStatus, reason: issueReason }),
    });
    if (updated) setIssueReason("");
  };

  const approve = async () => {
    if (!selectedJob) return;
    await mutate(`/api/v2/documents/${selectedJob.id}/approval`, { method: "POST" });
  };

  const deleteJob = async (job: Job) => {
    try {
      await request(`/api/v2/documents/${job.id}`, { method: "DELETE" });
      replaceJobs((current) => current.filter((item) => item.id !== job.id));
      if (selectedJobId === job.id) setSelectedJobId(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Document could not be deleted");
    }
  };

  const framePoint = (event: PointerEvent<SVGSVGElement>): Point | null => {
    if (!pageAsset || !pageFrame.current) return null;
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(pageAsset.width, ((event.clientX - rect.left) / rect.width) * pageAsset.width)),
      y: Math.max(0, Math.min(pageAsset.height, ((event.clientY - rect.top) / rect.height) * pageAsset.height)),
    };
  };
  const startDraw = (event: PointerEvent<SVGSVGElement>) => {
    if (!drawMode) return;
    const point = framePoint(event);
    if (!point) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    drawStart.current = point;
    setDraftPolygon([point, point, point, point]);
  };
  const moveDraw = (event: PointerEvent<SVGSVGElement>) => {
    if (!drawStart.current || !drawMode) return;
    const point = framePoint(event);
    if (!point) return;
    const start = drawStart.current;
    setDraftPolygon([
      { x: start.x, y: start.y },
      { x: point.x, y: start.y },
      { x: point.x, y: point.y },
      { x: start.x, y: point.y },
    ]);
  };
  const endDraw = () => {
    drawStart.current = null;
    setDrawMode(null);
  };

  const progress = (job: Job) => {
    if (job.status === "complete") return 100;
    if (!job.pages) return job.status === "queued" ? 8 : 14;
    return Math.max(14, Math.round((job.page / job.pages) * 94));
  };

  const evidence = useMemo(() => {
    if (!selectedRow || selectedRow.page_number !== viewPage) return { description: "", amount: "" };
    return {
      description: points(selectedRow.field_evidence?.description?.[0] ?? selectedRow.evidence?.[0]),
      amount: points(selectedRow.field_evidence?.amount?.[0]),
    };
  }, [selectedRow, viewPage]);

  const arrival = !jobs.length;

  return (
    <main>
      <div className="risk-ribbon">
        Public HTTP demo · no login · uploads are unencrypted and removed six hours after completion
      </div>
      <header className="masthead">
        <div className="brand-mark">G</div>
        <div>
          <p className="eyebrow">Evidence studio · review edition</p>
          <h1>Read every charge.<br />Resolve every doubt.</h1>
        </div>
        <div className={`mast-status ${health ? "online" : "offline"}`}>
          <span /> {health ? `${health.worker_capacity} inference lanes · ${health.active_jobs} active` : "Inference unavailable"}
        </div>
      </header>

      {arrival ? (
        <section className="arrival">
          <div className="arrival-copy">
            <p className="folio">01 / Intake</p>
            <h2>A bill enters.<br /><em>An auditable ledger emerges.</em></h2>
            <p className="lede">
              Upload one or more hospital bill PDFs. Two documents run in parallel; every accepted value remains linked to the exact source page.
            </p>
            <div className="proof-strip">
              <div><b>300</b><span>DPI evidence</span></div>
              <div><b>2×</b><span>parallel lanes</span></div>
              <div><b>6h</b><span>private cleanup</span></div>
            </div>
          </div>
          <label
            className={`drop-zone ${dragging ? "is-dragging" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={drop}
          >
            <input type="file" multiple accept="application/pdf,.pdf" onChange={acceptFiles} disabled={uploading} />
            <span className="drop-index">PDF × MULTI</span>
            <span className="drop-cross">+</span>
            <strong>{uploading ? "Transferring…" : "Place bills here"}</strong>
            <small>or click to choose · 25 MiB and 200 pages maximum each</small>
          </label>
        </section>
      ) : (
        <section className="review-desk">
          <aside className="queue-rail">
            <div className="rail-head">
              <div><p className="folio">01 / Queue</p><h2>{jobs.length} documents</h2></div>
              <label className="compact-upload">
                <input type="file" multiple accept="application/pdf,.pdf" onChange={acceptFiles} disabled={uploading} />
                {uploading ? "…" : "+"}
              </label>
            </div>
            <div className="lane-meter">
              <span>{activeJobs.filter((job) => job.status === "processing").length} / {health?.worker_capacity ?? 2} lanes occupied</span>
              <i style={{ width: `${Math.min(100, (activeJobs.filter((job) => job.status === "processing").length / (health?.worker_capacity ?? 2)) * 100)}%` }} />
            </div>
            <div className="job-list">
              {jobs.map((job, index) => (
                <button key={job.id} className={`job-card ${selectedJobId === job.id ? "selected" : ""}`} onClick={() => setSelectedJobId(job.id)}>
                  <span className={`job-state ${job.status}`} />
                  <span className="job-index">{String(index + 1).padStart(2, "0")}</span>
                  <span className="job-copy"><b>{job.original_name}</b><small>{job.status === "processing" ? `page ${job.page} of ${job.pages ?? "?"}` : job.status}{job.row_count !== null ? ` · ${job.row_count} rows` : ""}</small></span>
                  <span className="job-progress"><i style={{ width: `${progress(job)}%` }} /></span>
                </button>
              ))}
            </div>
            <div className="rail-note">UUID-scoped · no public document listing</div>
          </aside>

          <div className="desk-main">
            {selectedJob && selectedJob.status !== "complete" && (
              <section className="processing-card compact-processing">
                <div className="processing-meta"><p className="folio">02 / Reconstruction</p><span>{selectedJob.original_name}</span></div>
                <div className="processing-number">{String(progress(selectedJob)).padStart(2, "0")}<sup>%</sup></div>
                <div className="progress-track"><i style={{ width: `${progress(selectedJob)}%` }} /></div>
                <div className="processing-foot">
                  <strong>{selectedJob.status === "queued" ? "Waiting for an inference lane" : selectedJob.status === "failed" ? "Extraction stopped" : "Reading tables and grounding evidence"}</strong>
                  <span>{selectedJob.pages ? `page ${selectedJob.page} of ${selectedJob.pages}` : "preparing pages"}</span>
                </div>
                {selectedJob.error && <p className="error-note">{selectedJob.error}</p>}
                {selectedJob.status === "failed" && <button className="text-action" onClick={() => void deleteJob(selectedJob)}>Remove failed document</button>}
              </section>
            )}

            {selectedJob?.status === "complete" && rowsResult && review && (
              <section className="workspace">
                <div className="workspace-head">
                  <div><p className="folio">03 / Evidence ledger</p><h2>{review.rows_active.toLocaleString("en-IN")} active rows</h2></div>
                  <div className="summary-strip">
                    <span><b>{review.rows_modified}</b> corrected</span>
                    <span className={review.issues_open ? "warn" : ""}><b>{review.issues_open}</b> open issues</span>
                    <span className={review.rows_pending ? "warn" : ""}><b>{review.rows_pending}</b> pending</span>
                    <span className={review.approval ? "approved" : ""}><b>{review.approval ? "✓" : "—"}</b> {review.approval ? "approved" : "draft"}</span>
                  </div>
                </div>

                <div className="review-toolbar">
                  <input aria-label="Search rows" placeholder="Search charge, section, code…" value={query} onChange={(event) => { setQuery(event.target.value); setOffset(0); }} />
                  <select aria-label="Filter disposition" value={disposition} onChange={(event) => { setDisposition(event.target.value); setOffset(0); }}>
                    <option value="">All dispositions</option><option value="accepted">Accepted</option><option value="pending">Pending</option><option value="rejected">Rejected</option><option value="unreadable">Unreadable</option>
                  </select>
                  <input className="page-filter" aria-label="Filter page" type="number" min="1" max={rowsResult.pages} placeholder="Page" value={pageFilter} onChange={(event) => { setPageFilter(event.target.value); setOffset(0); }} />
                  <button className="toolbar-action" onClick={() => { setAddMode(true); setEditMode(false); setReason(""); setDraftPolygon(null); setDrawMode("add"); }}>+ Add grounded row</button>
                  <button className="toolbar-action subtle" onClick={() => void deleteJob(selectedJob)}>Delete document</button>
                </div>

                <div className="split-view">
                  <div className="ledger-pane">
                    <div className="ledger-caption"><span>{selectedJob.original_name}</span><span>{rowsResult.total.toLocaleString("en-IN")} matching · revision {review.revision}</span></div>
                    <div className="table-shell">
                      <table>
                        <thead><tr><th>#</th><th>Charge description</th><th>Qty</th><th>Rate</th><th>Net amount</th></tr></thead>
                        <tbody>
                          {rowsResult.rows.map((row, index) => (
                            <tr key={row.id} className={`${selectedRowId === row.id ? "selected" : ""} ${row.review_disposition} ${row.review.modified ? "modified" : ""}`} onClick={() => { setSelectedRowId(row.id); setEditMode(false); setAddMode(false); setDraftPolygon(null); }}>
                              <td>{String(offset + index + 1).padStart(2, "0")}</td>
                              <td><b>{row.description || "Unlabelled row"}</b><small><i className={row.review_disposition} /> {row.review_disposition} · p.{row.page_number}{row.review.modified ? " · reviewer changed" : ""}</small></td>
                              <td>{row.quantity ?? "—"}</td><td>{money(row.unit_price)}</td><td>{money(row.net_amount)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {!rowsResult.rows.length && <div className="empty-ledger">No rows match this view.</div>}
                    </div>
                    <div className="pagination">
                      <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>← Previous</button>
                      <span>{rowsResult.total ? `${offset + 1}–${Math.min(offset + PAGE_SIZE, rowsResult.total)} of ${rowsResult.total}` : "0 rows"}</span>
                      <button disabled={offset + PAGE_SIZE >= rowsResult.total} onClick={() => setOffset(offset + PAGE_SIZE)}>Next →</button>
                    </div>
                  </div>

                  <aside className="evidence-pane">
                    <div className="evidence-head">
                      <span>Source page {viewPage} / {rowsResult.pages}</span>
                      <span className="page-switch"><button disabled={viewPage <= 1} onClick={() => setViewPage(viewPage - 1)}>←</button><button disabled={viewPage >= rowsResult.pages} onClick={() => setViewPage(viewPage + 1)}>→</button></span>
                    </div>
                    <div className={`page-frame ${drawMode ? "drawing" : ""}`} ref={pageFrame}>
                      <div className="page-canvas" style={pageAsset ? { aspectRatio: `${pageAsset.width} / ${pageAsset.height}` } : undefined}>
                        {/* eslint-disable-next-line @next/next/no-img-element -- immutable evidence is intentionally not optimized */}
                        <img src={`/api/v2/documents/${selectedJob.id}/pages/${viewPage}`} alt={`Rendered bill page ${viewPage}`} />
                        {pageAsset && (
                          <svg viewBox={`0 0 ${pageAsset.width} ${pageAsset.height}`} preserveAspectRatio="none" aria-label={drawMode ? "Draw row evidence" : "Linked row evidence"} onPointerDown={startDraw} onPointerMove={moveDraw} onPointerUp={endDraw}>
                            {evidence.description && <polygon className="description-evidence" points={evidence.description} />}
                            {evidence.amount && evidence.amount !== evidence.description && <polygon className="amount-evidence" points={evidence.amount} />}
                            {draftPolygon && <polygon className="draft-evidence" points={draftPolygon.map((point) => `${point.x},${point.y}`).join(" ")} />}
                          </svg>
                        )}
                      </div>
                    </div>
                    {drawMode && <div className="draw-instruction">Drag across the visible source row to anchor the reviewer evidence.</div>}
                    {selectedRow && !addMode && (
                      <div className="evidence-note">
                        <span>{String((rowsResult.rows.findIndex((row) => row.id === selectedRow.id) + offset + 1)).padStart(2, "0")}</span>
                        <div><strong>{selectedRow.description}</strong><p>{money(selectedRow.net_amount)} · grounded on page {selectedRow.page_number}</p></div>
                        <button onClick={beginEdit}>Review row ↗</button>
                      </div>
                    )}
                  </aside>
                </div>

                {(editMode || addMode) && (
                  <div className="review-sheet">
                    <div className="sheet-title"><p className="folio">04 / Human review</p><h3>{addMode ? "Add a missing grounded row" : "Correct without erasing the machine record"}</h3></div>
                    {addMode ? (
                      <div className="edit-grid">
                        <label><span>Description</span><input value={addDescription} onChange={(event) => setAddDescription(event.target.value)} /></label>
                        <label><span>Net amount</span><input inputMode="decimal" value={addAmount} onChange={(event) => setAddAmount(event.target.value)} /></label>
                        <label className="wide"><span>Review reason</span><input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Why was this row added?" /></label>
                      </div>
                    ) : (
                      <div className="edit-grid">
                        <label className="wide"><span>Description</span><input value={editValues.description} onChange={(event) => setEditValues({ ...editValues, description: event.target.value })} /></label>
                        <label><span>Quantity</span><input value={editValues.quantity} onChange={(event) => setEditValues({ ...editValues, quantity: event.target.value })} /></label>
                        <label><span>Unit rate</span><input value={editValues.unit_price} onChange={(event) => setEditValues({ ...editValues, unit_price: event.target.value })} /></label>
                        <label><span>Discount</span><input value={editValues.discount} onChange={(event) => setEditValues({ ...editValues, discount: event.target.value })} /></label>
                        <label><span>Net amount</span><input value={editValues.net_amount} onChange={(event) => setEditValues({ ...editValues, net_amount: event.target.value })} /></label>
                        <label><span>Role</span><select value={editValues.role} onChange={(event) => setEditValues({ ...editValues, role: event.target.value })}><option value="detail">Detail</option><option value="category_rollup">Category rollup</option><option value="refund">Refund</option></select></label>
                        <label><span>Disposition</span><select value={editValues.review_disposition} onChange={(event) => setEditValues({ ...editValues, review_disposition: event.target.value })}><option value="accepted">Accepted</option><option value="pending">Pending</option><option value="unreadable">Unreadable</option></select></label>
                        <label className="wide"><span>Review reason</span><input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="What did you verify or change?" /></label>
                      </div>
                    )}
                    <div className="sheet-actions">
                      <button className="secondary" onClick={() => { setDrawMode(addMode ? "add" : "relink"); setDraftPolygon(null); }}>{draftPolygon ? "Redraw evidence" : addMode ? "Draw evidence above" : "Relink evidence"}</button>
                      <button className="secondary" onClick={() => { setEditMode(false); setAddMode(false); setDrawMode(null); setDraftPolygon(null); }}>Cancel</button>
                      {!addMode && <button className="danger" onClick={() => void rejectSelected()}>Reject row</button>}
                      <button className="primary" onClick={() => void (addMode ? addRow() : saveEdit())}>{addMode ? "Add grounded row" : "Save correction"}</button>
                    </div>
                  </div>
                )}

                <div className="approval-zone">
                  <div className="issues-panel">
                    <div className="zone-head"><div><p className="folio">05 / Structural review</p><h3>{review.issues.length ? `${review.issues_open} of ${review.issues.length} issues open` : "No structural issues"}</h3></div></div>
                    {review.issues.map((issue) => (
                      <div className={`issue-card ${issue.status}`} key={issue.id}>
                        <span>p.{issue.page_number}</span><div><b>{issue.table_type.replaceAll("_", " ")}</b><small>{issue.reason_codes.join(" · ").replaceAll("_", " ")}</small></div>
                        <button onClick={() => { setViewPage(issue.page_number); setIssueReason(issue.resolution_reason ?? ""); }}>{issue.status}</button>
                        <button className="issue-action" onClick={() => void updateIssue(issue)}>{issue.status === "open" ? "Resolve" : "Reopen"}</button>
                      </div>
                    ))}
                    {!!review.issues.length && <input className="issue-reason" placeholder="Issue review note…" value={issueReason} onChange={(event) => setIssueReason(event.target.value)} />}
                  </div>
                  <div className="approval-card">
                    <p className="folio">06 / Approval & export</p>
                    {review.approval ? (
                      <><h3>Review sealed.</h3><p>Approved at {new Date(review.approval.approved_at).toLocaleString()}.</p><div className="export-grid"><a href={`/api/v2/documents/${selectedJob.id}/exports/csv`}>CSV ↗</a><a href={`/api/v2/documents/${selectedJob.id}/exports/json`}>JSON ↗</a><a href={`/api/v2/documents/${selectedJob.id}/exports/evidence.zip`}>Evidence ZIP ↗</a></div></>
                    ) : (
                      <><h3>Machine draft.</h3><p>Approval requires zero pending rows, zero open structural issues, and grounded description and amount evidence.</p><button className="approve-button" onClick={() => void approve()}>Approve reviewed ledger →</button></>
                    )}
                  </div>
                </div>
              </section>
            )}
          </div>
        </section>
      )}

      {error && <div className="toast" role="alert">{error}<button onClick={() => setError(null)}>×</button></div>}
      <footer><span>GMoney / evidence-grounded extraction</span><span>Machine output remains immutable · reviewer changes are revisioned</span></footer>
    </main>
  );
}
