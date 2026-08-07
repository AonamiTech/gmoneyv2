"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type {
  CSSProperties,
  ChangeEvent,
  DragEvent,
  KeyboardEvent,
  PointerEvent,
} from "react";

type Job = {
  id: string;
  status: "uploading" | "queued" | "processing" | "complete" | "failed";
  original_name: string;
  page: number;
  pages: number | null;
  row_count: number | null;
  hospital_name: string | null;
  hospital_confidence: number | null;
  hospital_name_source: "machine" | "reviewer" | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  last_activity_at: string;
  expires_at: string | null;
};
type JobsResult = {
  total: number;
  offset: number;
  limit: number;
  has_more: boolean;
  documents: Job[];
};
type TrainedHospital = {
  hospital_id: string;
  hospital_name: string;
  active_profile_count: number;
};
type TrainedHospitalsResult = {
  total: number;
  hospitals: TrainedHospital[];
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
type Hospital = {
  name: string;
  source: "machine" | "reviewer";
  machine_name: string | null;
  confidence: number | null;
  page_number: number;
  evidence: Evidence;
  reason: string | null;
};
type Row = {
  id: string;
  description: string | null;
  service_date_raw: string | null;
  service_date_iso: string | null;
  section: string | null;
  quantity: string | null;
  unit_price: string | null;
  gross_amount: string | null;
  discount: string | null;
  net_amount: string | null;
  role: string;
  review_disposition: string;
  page_number: number;
  evidence: Evidence[];
  field_evidence: Record<string, Evidence[]>;
  validation_flags: string[];
  review: ReviewMeta;
};
type PrintedTotal = {
  amount: string;
  label: string;
  kind: string | null;
  scope: string | null;
  page_number: number;
  evidence: Evidence;
  is_primary?: boolean;
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
  hospital: Hospital | null;
  review_revision: number;
  totals: {
    items_total: string;
    bill_total: PrintedTotal | null;
    printed_totals: PrintedTotal[];
    difference: string | null;
    comparison: "match" | "mismatch" | "items_partial" | "bill_total_missing" | "multiple_printed_totals";
    missing_item_amounts: number;
  };
  total: number;
  offset: number;
  limit: number;
  rows: Row[];
};
type SourceColumn = {
  id: string;
  label: string;
  order: number;
  canonical_field: string | null;
  evidence: Evidence[];
};
type SourceCell = {
  column_id: string;
  raw_value: string | null;
  evidence: Evidence[];
  validation_flags: string[];
};
type SourceRow = {
  id: string;
  order: number;
  ordinal: number;
  canonical_row_id: string | null;
  cells: SourceCell[];
  validation_flags: string[];
};
type SourceTable = {
  id: string;
  page_number: number;
  table_id: string;
  table_type: string;
  columns: SourceColumn[];
  rows: SourceRow[];
  validation_flags: string[];
};
type SourceTablesResult = {
  document_id: string;
  available: boolean;
  unavailable_reason: "legacy_result" | "no_source_tables" | null;
  total: number;
  offset: number;
  limit: number;
  tables: SourceTable[];
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
  hospital: Hospital | null;
  approval: { status: string; approved_at: string; review_revision: number } | null;
};
type Health = {
  active_jobs: number;
  worker_capacity: number;
  queue_capacity: number;
  retention_hours: number;
  storage_total_bytes: number;
  storage_free_bytes: number;
  storage_min_free_bytes: number;
};
type EditValues = {
  description: string;
  service_date_iso: string;
  quantity: string;
  unit_price: string;
  gross_amount: string;
  discount: string;
  net_amount: string;
  role: string;
  review_disposition: string;
};

const PAGE_SIZE = 150;
const emptyEdit: EditValues = {
  description: "",
  service_date_iso: "",
  quantity: "",
  unit_price: "",
  gross_amount: "",
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

const differenceCopy = (value: string | null, comparison: RowsResult["totals"]["comparison"]) => {
  if (comparison === "multiple_printed_totals") return "Multiple document totals found · review printed totals";
  if (value === null) return "Comparison unavailable";
  const difference = Number(value);
  if (Math.abs(difference) <= 0.01) return "Items and bill agree";
  return `Items are ${money(String(Math.abs(difference)))} ${difference > 0 ? "above" : "below"} bill`;
};

const serviceDate = (value: string | null, raw: string | null) => {
  if (!value) return raw || "Not printed";
  const [year, month, day] = value.split("-").map(Number);
  if (!year || !month || !day) return value;
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  }).format(new Date(Date.UTC(year, month - 1, day)));
};

const dateTime = (value: string | null) =>
  value
    ? new Intl.DateTimeFormat("en-IN", {
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      }).format(new Date(value))
    : "—";

const storageSize = (value: number | undefined) => {
  if (value === undefined) return "—";
  return `${(value / 1024 ** 3).toFixed(value >= 100 * 1024 ** 3 ? 0 : 1)} GB`;
};

const detail = (payload: unknown, status: number, statusText: string) => {
  if (typeof payload === "string") return payload;
  if (payload && typeof payload === "object" && "detail" in payload) {
    const value = (payload as { detail: unknown }).detail;
    if (typeof value === "string") return value;
    if (value && typeof value === "object" && "code" in value) {
      return String((value as { code: unknown }).code).replaceAll("_", " ");
    }
  }
  if (status === 413) return "The PDF is larger than the allowed upload size.";
  if (status === 429) return "Too many requests are in progress. Please try again shortly.";
  if ([502, 503, 504].includes(status)) {
    return "The service is temporarily unavailable. Please try again shortly.";
  }
  const suffix = statusText ? ` ${statusText}` : "";
  return `The request failed (HTTP ${status}${suffix}).`;
};

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  let payload: unknown = null;
  if (response.status !== 204) {
    const body = await response.text();
    if (body) {
      try {
        payload = JSON.parse(body);
      } catch {
        payload = null;
      }
    }
  }
  if (!response.ok) throw new Error(detail(payload, response.status, response.statusText));
  return payload as T;
}

const points = (evidence: Evidence | undefined) =>
  evidence?.polygon.points.map((point) => `${point.x},${point.y}`).join(" ") ?? "";

export default function Home() {
  const [activeJobs, setActiveJobs] = useState<Job[]>([]);
  const [historyJobs, setHistoryJobs] = useState<Job[]>([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyQuery, setHistoryQuery] = useState("");
  const [trainedHospitals, setTrainedHospitals] = useState<TrainedHospital[]>([]);
  const [trainedHospitalsError, setTrainedHospitalsError] = useState<string | null>(null);
  const [railView, setRailView] = useState<"active" | "history" | "hospitals">("history");
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [abortingJobIds, setAbortingJobIds] = useState<Set<string>>(new Set());
  const [rowsResult, setRowsResult] = useState<RowsResult | null>(null);
  const [sourceTablesResult, setSourceTablesResult] = useState<SourceTablesResult | null>(null);
  const [review, setReview] = useState<ReviewSummary | null>(null);
  const [selectedRowId, setSelectedRowId] = useState<string | null>(null);
  const [selectedSourceRowId, setSelectedSourceRowId] = useState<string | null>(null);
  const [ledgerMode, setLedgerMode] = useState<"printed" | "normalized">("printed");
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
  const [hospitalEditMode, setHospitalEditMode] = useState(false);
  const [focusedPrintedTotal, setFocusedPrintedTotal] = useState<PrintedTotal | null>(null);
  const totalEvidenceActive = focusedPrintedTotal !== null;
  const [hospitalName, setHospitalName] = useState("");
  const [hospitalReason, setHospitalReason] = useState("");
  const [drawMode, setDrawMode] = useState<"add" | "relink" | "hospital" | null>(null);
  const [draftPolygon, setDraftPolygon] = useState<Point[] | null>(null);
  const [splitPercent, setSplitPercent] = useState(55);
  const [resizingSplit, setResizingSplit] = useState(false);
  const [viewMode, setViewMode] = useState<"fit-width" | "fit-page" | "zoom">("fit-width");
  const [zoom, setZoom] = useState(150);
  const drawStart = useRef<Point | null>(null);
  const pageFrame = useRef<HTMLDivElement | null>(null);
  const pageCanvas = useRef<HTMLDivElement | null>(null);
  const splitView = useRef<HTMLDivElement | null>(null);
  const evidencePane = useRef<HTMLElement | null>(null);
  const previousActiveIds = useRef<Set<string>>(new Set());

  const jobs = useMemo(
    () => [
      ...activeJobs,
      ...historyJobs.filter(
        (historyJob) => !activeJobs.some((activeJob) => activeJob.id === historyJob.id),
      ),
    ],
    [activeJobs, historyJobs],
  );
  const visibleJobs = useMemo(
    () => (railView === "active" ? activeJobs : railView === "history" ? historyJobs : []),
    [activeJobs, historyJobs, railView],
  );
  const selectedJob = jobs.find((job) => job.id === selectedJobId) ?? null;
  const selectedRow = rowsResult?.rows.find((row) => row.id === selectedRowId) ?? null;
  const selectedRowPage = selectedRow?.page_number ?? null;
  const selectedSource = useMemo(() => {
    for (const table of sourceTablesResult?.tables ?? []) {
      const row = table.rows.find((candidate) => candidate.id === selectedSourceRowId);
      if (row) return { table, row };
    }
    return null;
  }, [selectedSourceRowId, sourceTablesResult]);
  const selectedSourcePage = selectedSource?.table.page_number ?? null;
  const selectedSourceDescriptionEvidence =
    selectedSource?.row.cells.find(
      (cell) =>
        selectedSource.table.columns.find((column) => column.id === cell.column_id)
          ?.canonical_field === "description",
    )?.evidence[0] ??
    selectedSource?.row.cells.find((cell) => cell.evidence.length)?.evidence[0];
  const selectedSourceAmountEvidence =
    [...(selectedSource?.row.cells ?? [])]
      .reverse()
      .find((cell) => cell.evidence.length)?.evidence[0];
  const pageAsset = rowsResult?.page_assets.find((asset) => asset.page_number === viewPage);
  const hospital = rowsResult?.hospital ?? review?.hospital ?? null;
  const refreshHealth = useCallback(() => {
    request<Health>("/api/v2/health/ready").then(setHealth).catch(() => setHealth(null));
  }, []);

  const refreshActiveJobs = useCallback(async () => {
    const result = await request<JobsResult>("/api/v2/documents?scope=active&limit=200");
    setActiveJobs(result.documents);
    return result.documents;
  }, []);

  const refreshHistoryJobs = useCallback(async (historyOffset = 0, append = false) => {
    const params = new URLSearchParams({
      scope: "history",
      offset: String(historyOffset),
      limit: "200",
    });
    if (historyQuery.trim()) params.set("query", historyQuery.trim());
    const result = await request<JobsResult>(`/api/v2/documents?${params}`);
    setHistoryTotal(result.total);
    setHistoryJobs((current) => {
      if (!append) return result.documents;
      const known = new Set(current.map((job) => job.id));
      return [...current, ...result.documents.filter((job) => !known.has(job.id))];
    });
    return result.documents;
  }, [historyQuery]);

  const refreshTrainedHospitals = useCallback(async () => {
    try {
      const result = await request<TrainedHospitalsResult>("/api/v2/hospitals/trained");
      setTrainedHospitals(result.hospitals);
      setTrainedHospitalsError(null);
      return result.hospitals;
    } catch (cause) {
      setTrainedHospitalsError(
        cause instanceof Error ? cause.message : "The trained hospital list could not be loaded.",
      );
      return [];
    }
  }, []);

  useEffect(() => {
    void refreshActiveJobs().catch(() => setError("The active bill queue could not be loaded."));
    refreshHealth();
  }, [refreshActiveJobs, refreshHealth]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refreshHistoryJobs().catch(() => setError("Bill history could not be loaded."));
    }, 220);
    return () => window.clearTimeout(timer);
  }, [refreshHistoryJobs]);

  useEffect(() => {
    if (railView === "hospitals") void refreshTrainedHospitals();
  }, [railView, refreshTrainedHospitals]);

  useEffect(() => {
    const poll = window.setInterval(() => {
      void refreshActiveJobs().catch(() => undefined);
      refreshHealth();
    }, 3000);
    return () => window.clearInterval(poll);
  }, [refreshActiveJobs, refreshHealth]);

  useEffect(() => {
    const poll = window.setInterval(() => {
      void refreshHistoryJobs().catch(() => undefined);
    }, 30000);
    return () => window.clearInterval(poll);
  }, [refreshHistoryJobs]);

  useEffect(() => {
    const currentIds = new Set(activeJobs.map((job) => job.id));
    const jobLeftActiveQueue = [...previousActiveIds.current].some(
      (jobId) => !currentIds.has(jobId),
    );
    previousActiveIds.current = currentIds;
    if (jobLeftActiveQueue) {
      void refreshHistoryJobs().catch(() => undefined);
    }
  }, [activeJobs, refreshHistoryJobs]);

  useEffect(() => {
    if (railView === "hospitals") return;
    if (railView === "active" && !activeJobs.length && historyJobs.length) {
      setRailView("history");
      return;
    }
    setSelectedJobId((current) =>
      current && visibleJobs.some((job) => job.id === current)
        ? current
        : (visibleJobs[0]?.id ?? null),
    );
  }, [activeJobs.length, historyJobs.length, railView, visibleJobs]);

  const loadWorkspace = useCallback(async () => {
    if (!selectedJob || selectedJob.status !== "complete") return;
    const params = new URLSearchParams({ offset: String(offset), limit: String(PAGE_SIZE) });
    const sourceParams = new URLSearchParams({
      offset: String(offset),
      limit: String(PAGE_SIZE),
    });
    if (query.trim()) params.set("query", query.trim());
    if (query.trim()) sourceParams.set("query", query.trim());
    if (disposition) params.set("disposition", disposition);
    if (pageFilter) params.set("source_page", pageFilter);
    if (pageFilter) sourceParams.set("source_page", pageFilter);
    try {
      const [rows, sources, summary] = await Promise.all([
        request<RowsResult>(`/api/v2/documents/${selectedJob.id}/rows?${params}`),
        request<SourceTablesResult>(
          `/api/v2/documents/${selectedJob.id}/source-tables?${sourceParams}`,
        ),
        request<ReviewSummary>(`/api/v2/documents/${selectedJob.id}/review`),
      ]);
      setRowsResult(rows);
      setSourceTablesResult(sources);
      setReview(summary);
      setSelectedRowId((current) =>
        current && rows.rows.some((row) => row.id === current)
          ? current
          : (rows.rows[0]?.id ?? null),
      );
      const sourceRows = sources.tables.flatMap((table) => table.rows);
      setSelectedSourceRowId((current) =>
        current && sourceRows.some((row) => row.id === current)
          ? current
          : (sourceRows[0]?.id ?? null),
      );
      setLedgerMode((current) => (sources.available ? current : "normalized"));
      if (!rows.rows.length && !sourceRows.length && offset > 0) {
        setOffset(Math.max(0, offset - PAGE_SIZE));
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The result could not be loaded.");
    }
  }, [selectedJob, offset, query, disposition, pageFilter]);

  useEffect(() => {
    setRowsResult(null);
    setSourceTablesResult(null);
    setReview(null);
    setOffset(0);
    setQuery("");
    setDisposition("");
    setPageFilter("");
    setEditMode(false);
    setAddMode(false);
    setHospitalEditMode(false);
    setFocusedPrintedTotal(null);
    setDrawMode(null);
    setDraftPolygon(null);
    setSelectedSourceRowId(null);
    setLedgerMode("printed");
  }, [selectedJobId]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadWorkspace(), 220);
    return () => window.clearTimeout(timer);
  }, [loadWorkspace]);

  useEffect(() => {
    if (selectedRowPage) setViewPage(selectedRowPage);
  }, [selectedRowId, selectedRowPage]);

  useEffect(() => {
    if (ledgerMode === "printed" && selectedSourcePage) {
      setViewPage(selectedSourcePage);
    }
  }, [ledgerMode, selectedSourceRowId, selectedSourcePage]);

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
        setActiveJobs((current) => [
          ...uploaded,
          ...current.filter((job) => !uploaded.some((item) => item.id === job.id)),
        ]);
        setRailView("active");
        setSelectedJobId(uploaded[0].id);
        void refreshActiveJobs();
        refreshHealth();
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [refreshActiveJobs, refreshHealth],
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
    setFocusedPrintedTotal(null);
    setEditValues({
      description: selectedRow.description ?? "",
      service_date_iso: selectedRow.service_date_iso ?? "",
      quantity: selectedRow.quantity ?? "",
      unit_price: selectedRow.unit_price ?? "",
      gross_amount: selectedRow.gross_amount ?? "",
      discount: selectedRow.discount ?? "",
      net_amount: selectedRow.net_amount ?? "",
      role: selectedRow.role,
      review_disposition: selectedRow.review_disposition,
    });
    setReason(selectedRow.review.reason ?? "");
    setEditMode(true);
    setAddMode(false);
    setHospitalEditMode(false);
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

  const beginHospitalEdit = () => {
    if (!rowsResult) return;
    setFocusedPrintedTotal(null);
    const source = rowsResult.hospital;
    setHospitalName(source?.name ?? "");
    setHospitalReason(source?.reason ?? "");
    setHospitalEditMode(true);
    setEditMode(false);
    setAddMode(false);
    setViewPage(source?.page_number ?? 1);
    setDraftPolygon(source?.evidence?.polygon.points ?? null);
    setDrawMode(null);
  };

  const saveHospital = async () => {
    if (!selectedJob || !draftPolygon || hospitalName.trim().length < 2 || hospitalReason.trim().length < 3) {
      setError("Confirm the hospital name, header evidence, and a short review reason.");
      return;
    }
    const saved = await mutate(`/api/v2/documents/${selectedJob.id}/metadata`, {
      method: "PATCH",
      body: JSON.stringify({
        hospital_name: hospitalName,
        page_number: viewPage,
        polygon: { points: draftPolygon },
        reason: hospitalReason,
      }),
    });
    if (saved) {
      setHospitalEditMode(false);
      setDraftPolygon(null);
      setDrawMode(null);
      void refreshHistoryJobs();
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
      setActiveJobs((current) => current.filter((item) => item.id !== job.id));
      setHistoryJobs((current) => current.filter((item) => item.id !== job.id));
      setHistoryTotal((current) => Math.max(0, current - 1));
      if (selectedJobId === job.id) setSelectedJobId(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Document could not be deleted");
    }
  };

  const abortJob = async (job: Job) => {
    if (!window.confirm(`Abort ${job.original_name}? Its partial extraction will be deleted.`)) {
      return;
    }
    setAbortingJobIds((current) => new Set(current).add(job.id));
    try {
      await request<{ id: string; status: "cancelling" }>(
        `/api/v2/documents/${job.id}/abort`,
        { method: "POST" },
      );
      setActiveJobs((current) => current.filter((item) => item.id !== job.id));
      if (selectedJobId === job.id) setSelectedJobId(null);
      refreshHealth();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The bill could not be aborted.");
      void refreshActiveJobs();
    } finally {
      setAbortingJobIds((current) => {
        const next = new Set(current);
        next.delete(job.id);
        return next;
      });
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

  const resize = (event: PointerEvent<HTMLButtonElement>) => {
    if (!resizingSplit || !splitView.current) return;
    const bounds = splitView.current.getBoundingClientRect();
    const percentage = ((event.clientX - bounds.left) / bounds.width) * 100;
    setSplitPercent(Math.max(44, Math.min(68, percentage)));
  };

  const resizeWithKeyboard = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    setSplitPercent((current) =>
      Math.max(44, Math.min(68, current + (event.key === "ArrowLeft" ? -2 : 2))),
    );
  };

  const focusEvidenceSource = (source: Evidence | undefined) => {
    const sourceAsset = rowsResult?.page_assets.find(
      (asset) => asset.page_number === source?.page_number,
    );
    if (!source || !sourceAsset) return;
    setViewPage(source.page_number);
    setViewMode("zoom");
    setZoom(175);
    window.setTimeout(() => {
      const frame = pageFrame.current;
      const canvas = pageCanvas.current;
      if (!frame || !canvas) return;
      const polygon = source.polygon.points;
      const centerX = polygon.reduce((sum, point) => sum + point.x, 0) / polygon.length;
      const centerY = polygon.reduce((sum, point) => sum + point.y, 0) / polygon.length;
      frame.scrollTo({
        left: Math.max(0, canvas.offsetLeft + (centerX / sourceAsset.width) * canvas.clientWidth - frame.clientWidth / 2),
        top: Math.max(0, canvas.offsetTop + (centerY / sourceAsset.height) * canvas.clientHeight - frame.clientHeight / 2),
        behavior: "smooth",
      });
    }, 80);
  };

  const focusEvidence = () => {
    const source = totalEvidenceActive
      ? focusedPrintedTotal?.evidence
      : hospitalEditMode
        ? hospital?.evidence
        : ledgerMode === "printed"
          ? selectedSourceDescriptionEvidence
          : selectedRow?.field_evidence?.description?.[0] ?? selectedRow?.evidence?.[0];
    focusEvidenceSource(source);
  };

  const focusBillTotal = (total: PrintedTotal | null = rowsResult?.totals.bill_total ?? null) => {
    const source = total?.evidence;
    if (!source) return;
    setFocusedPrintedTotal(total);
    setHospitalEditMode(false);
    setEditMode(false);
    setAddMode(false);
    focusEvidenceSource(source);
  };

  const toggleFullscreen = async () => {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
      return;
    }
    await evidencePane.current?.requestFullscreen();
  };

  const progress = (job: Job) => {
    if (job.status === "complete") return 100;
    if (!job.pages) return job.status === "queued" ? 8 : 14;
    return Math.max(14, Math.round((job.page / job.pages) * 94));
  };

  const evidence = useMemo(() => {
    const total = focusedPrintedTotal;
    if (totalEvidenceActive) {
      return {
        description: "",
        amount: "",
        total: total?.page_number === viewPage ? points(total.evidence) : "",
      };
    }
    if (ledgerMode === "printed") {
      if (!selectedSource || selectedSource.table.page_number !== viewPage) {
        return { description: "", amount: "", total: "" };
      }
      return {
        description: points(selectedSourceDescriptionEvidence),
        amount: points(selectedSourceAmountEvidence),
        total: "",
      };
    }
    if (!selectedRow || selectedRow.page_number !== viewPage) {
      return { description: "", amount: "", total: "" };
    }
    return {
      description: points(selectedRow.field_evidence?.description?.[0] ?? selectedRow.evidence?.[0]),
      amount: points(selectedRow.field_evidence?.amount?.[0]),
      total: "",
    };
  }, [
    focusedPrintedTotal,
    ledgerMode,
    selectedRow,
    selectedSource,
    selectedSourceAmountEvidence,
    selectedSourceDescriptionEvidence,
    totalEvidenceActive,
    viewPage,
  ]);
  const hospitalEvidence =
    hospitalEditMode && hospital?.page_number === viewPage ? points(hospital.evidence) : "";

  const printedAvailable = sourceTablesResult?.available ?? false;
  const printedUnavailableCopy =
    sourceTablesResult?.unavailable_reason === "legacy_result"
      ? "Printed columns are unavailable for this older extraction. Reprocess this bill to enable them."
      : sourceTablesResult?.unavailable_reason === "no_source_tables"
        ? "No printed table structure was detected; normalized rows are shown."
        : null;
  const ledgerTotal =
    ledgerMode === "printed"
      ? (sourceTablesResult?.total ?? 0)
      : (rowsResult?.total ?? 0);
  const arrival = !jobs.length && railView !== "hospitals";
  const workerCapacity = health?.worker_capacity ?? 1;

  return (
    <main>
      <div className="risk-ribbon">
        Public HTTP demo · no login · shared 30-day bill history · uploads are unencrypted
      </div>
      <header className="masthead">
        <div className="brand-mark">G</div>
        <div>
          <p className="eyebrow">Evidence studio · review edition</p>
          <h1>Read every charge.<br />Resolve every doubt.</h1>
        </div>
        <div className={`mast-status ${health ? "online" : "offline"}`}>
          <span /> {health ? `${health.worker_capacity} lanes · ${health.active_jobs} active · ${storageSize(health.storage_free_bytes)} free` : "Inference unavailable"}
        </div>
      </header>

      {arrival ? (
        <section className="arrival">
          <div className="arrival-copy">
            <p className="folio">01 / Intake</p>
            <h2>A bill enters.<br /><em>An auditable ledger emerges.</em></h2>
            <p className="lede">
              Upload one or more hospital bill PDFs. Up to {workerCapacity} {workerCapacity === 1 ? "document runs" : "documents run"} at once; every accepted value remains linked to the exact source page.
            </p>
            <div className="proof-strip">
              <div><b>300</b><span>DPI evidence</span></div>
              <div><b>{workerCapacity}×</b><span>inference {workerCapacity === 1 ? "lane" : "lanes"}</span></div>
              <div><b>30d</b><span>searchable history</span></div>
            </div>
            <button className="hospital-directory-link" onClick={() => setRailView("hospitals")}>View trained hospitals →</button>
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
            <small>or click to choose</small>
          </label>
        </section>
      ) : (
        <section className="review-desk">
          <aside className="queue-rail">
            <div className="rail-head">
              <div>
                <p className="folio">01 / {railView === "hospitals" ? "Training index" : "Document index"}</p>
                <h2>
                  {railView === "active" ? activeJobs.length : railView === "history" ? historyTotal : trainedHospitals.length}
                  {railView === "hospitals" ? " hospitals" : " documents"}
                </h2>
              </div>
              <label className="compact-upload">
                <input type="file" multiple accept="application/pdf,.pdf" onChange={acceptFiles} disabled={uploading} />
                {uploading ? "…" : "+"}
              </label>
            </div>
            {railView === "hospitals" ? (
              <div className="profile-meter">Active, hospital-specific layout profiles</div>
            ) : (
              <div className="lane-meter">
                <span>{activeJobs.filter((job) => job.status === "processing").length} / {health?.worker_capacity ?? 2} lanes occupied</span>
                <i style={{ width: `${Math.min(100, (activeJobs.filter((job) => job.status === "processing").length / (health?.worker_capacity ?? 2)) * 100)}%` }} />
              </div>
            )}
            <div className="rail-tabs" role="tablist" aria-label="Document index">
              <button className={railView === "active" ? "active" : ""} onClick={() => setRailView("active")}>
                Active <span>{activeJobs.length}</span>
              </button>
              <button className={railView === "history" ? "active" : ""} onClick={() => setRailView("history")}>
                History <span>{historyTotal}</span>
              </button>
              <button className={railView === "hospitals" ? "active" : ""} onClick={() => setRailView("hospitals")}>
                Hospitals <span>{trainedHospitals.length}</span>
              </button>
            </div>
            {railView === "history" && (
              <input
                className="history-search"
                aria-label="Search bill history"
                placeholder="Hospital or filename"
                value={historyQuery}
                onChange={(event) => setHistoryQuery(event.target.value)}
              />
            )}
            <div className="job-list">
              {visibleJobs.map((job, index) => (
                <div className="job-card-frame" key={job.id}>
                  <button className={`job-card ${selectedJobId === job.id ? "selected" : ""}`} onClick={() => setSelectedJobId(job.id)}>
                    <span className={`job-state ${job.status}`} />
                    <span className="job-index">{String(index + 1).padStart(2, "0")}</span>
                    <span className="job-copy">
                      <b>{job.hospital_name ?? (job.status === "complete" ? "Hospital not identified" : "Identifying hospital…")}</b>
                      <small className="job-file">{job.original_name}</small>
                      <small>
                        {job.status === "processing" ? `page ${job.page} of ${job.pages ?? "?"}` : job.status}
                        {job.row_count !== null ? ` · ${job.row_count} rows` : ""}
                      </small>
                      <small className="job-time">
                        {job.status === "complete"
                          ? `updated ${dateTime(job.last_activity_at)} · expires ${dateTime(job.expires_at)}`
                          : `received ${dateTime(job.created_at)}`}
                      </small>
                    </span>
                    <span className="job-progress"><i style={{ width: `${progress(job)}%` }} /></span>
                  </button>
                  {(job.status === "queued" || job.status === "processing") && (
                    <button className="job-abort" disabled={abortingJobIds.has(job.id)} onClick={() => void abortJob(job)}>
                      {abortingJobIds.has(job.id) ? "Aborting…" : "Abort"}
                    </button>
                  )}
                </div>
              ))}
              {railView === "hospitals" && trainedHospitals.map((item, index) => (
                <article className="hospital-rail-card" key={item.hospital_id}>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <div><b>{item.hospital_name}</b><small>{item.active_profile_count} active {item.active_profile_count === 1 ? "profile" : "profiles"}</small></div>
                </article>
              ))}
              {railView !== "hospitals" && !visibleJobs.length && (
                <div className="rail-empty">
                  {railView === "active" ? "No bills are running." : "No historical bills match."}
                </div>
              )}
              {railView === "hospitals" && !trainedHospitals.length && (
                <div className="rail-empty">{trainedHospitalsError ?? "No active hospital profiles."}</div>
              )}
              {railView === "history" && historyJobs.length < historyTotal && (
                <button className="load-history" onClick={() => void refreshHistoryJobs(historyJobs.length, true)}>
                  Load older bills
                </button>
              )}
            </div>
            <div className="rail-note">{railView === "hospitals" ? "Shared active-profile registry" : "Public shared index · full evidence retained for 30 days"}</div>
          </aside>

          <div className="desk-main">
            {railView === "hospitals" && (
              <section className="hospital-directory">
                <div className="directory-heading">
                  <div><p className="folio">02 / Training registry</p><h2>Hospitals the system has profiles for.</h2></div>
                  <button onClick={() => void refreshTrainedHospitals()}>Refresh registry</button>
                </div>
                <p className="directory-lede">Only active, hospital-specific layout profiles appear here. Candidate and archived profiles remain outside the production directory.</p>
                {trainedHospitalsError ? (
                  <div className="directory-empty">{trainedHospitalsError}</div>
                ) : trainedHospitals.length ? (
                  <div className="hospital-grid">
                    {trainedHospitals.map((item, index) => (
                      <article key={item.hospital_id}>
                        <span>{String(index + 1).padStart(2, "0")}</span>
                        <h3>{item.hospital_name}</h3>
                        <p>{item.active_profile_count} active layout {item.active_profile_count === 1 ? "profile" : "profiles"}</p>
                        <small>{item.hospital_id}</small>
                      </article>
                    ))}
                  </div>
                ) : (
                  <div className="directory-empty">No active hospital-specific profiles have been published.</div>
                )}
              </section>
            )}

            {railView !== "hospitals" && selectedJob && selectedJob.status !== "complete" && (
              <section className="processing-card compact-processing">
                <div className="processing-meta"><p className="folio">02 / Reconstruction</p><span>{selectedJob.original_name}</span></div>
                <div className="processing-number">{String(progress(selectedJob)).padStart(2, "0")}<sup>%</sup></div>
                <div className="progress-track"><i style={{ width: `${progress(selectedJob)}%` }} /></div>
                <div className="processing-foot">
                  <strong>{selectedJob.status === "queued" ? "Waiting for an inference lane" : selectedJob.status === "failed" ? "Extraction stopped" : "Reading tables and grounding evidence"}</strong>
                  <span>{selectedJob.pages ? `page ${selectedJob.page} of ${selectedJob.pages}` : "preparing pages"}</span>
                </div>
                {selectedJob.error && <p className="error-note">{selectedJob.error}</p>}
                {(selectedJob.status === "queued" || selectedJob.status === "processing") && (
                  <button className="text-action abort-action" disabled={abortingJobIds.has(selectedJob.id)} onClick={() => void abortJob(selectedJob)}>
                    {abortingJobIds.has(selectedJob.id) ? "Aborting and deleting…" : "Abort this bill"}
                  </button>
                )}
                {selectedJob.status === "failed" && <button className="text-action" onClick={() => void deleteJob(selectedJob)}>Remove failed document</button>}
              </section>
            )}

            {railView !== "hospitals" && selectedJob?.status === "complete" && rowsResult && review && (
              <section className="workspace">
                <div className="workspace-head">
                  <div className="hospital-heading">
                    <p className="folio">03 / Evidence ledger</p>
                    <div className="hospital-title-line">
                      <h2>{hospital?.name ?? "Hospital not identified"}</h2>
                      <button onClick={beginHospitalEdit}>Correct label</button>
                    </div>
                    <p className="document-identity">
                      {selectedJob.original_name} · {review.rows_active.toLocaleString("en-IN")} active rows
                      {hospital && (
                        <span className={`identity-source ${hospital.source}`}>
                          {hospital.source === "reviewer" ? "reviewer verified" : "machine identified"}
                        </span>
                      )}
                    </p>
                  </div>
                  <div className="summary-strip">
                    <span><b>{review.rows_modified}</b> corrected</span>
                    <span className={review.issues_open ? "warn" : ""}><b>{review.issues_open}</b> open issues</span>
                    <span className={review.rows_pending ? "warn" : ""}><b>{review.rows_pending}</b> pending</span>
                    <span className={review.approval ? "approved" : ""}><b>{review.approval ? "✓" : "—"}</b> {review.approval ? "approved" : "draft"}</span>
                  </div>
                </div>

                <section className={`totals-band ${rowsResult.totals.comparison}`} aria-label="Bill totals comparison" aria-live="polite">
                  <article>
                    <p>Billable items total</p>
                    <strong>{money(rowsResult.totals.items_total)}</strong>
                    <small>
                      {rowsResult.totals.missing_item_amounts
                        ? `Partial · ${rowsResult.totals.missing_item_amounts} missing ${rowsResult.totals.missing_item_amounts === 1 ? "amount" : "amounts"}`
                        : "Informational rows excluded"}
                    </small>
                  </article>
                  <article className="printed-total">
                    <p>Printed bill total</p>
                    <strong>{money(rowsResult.totals.bill_total?.amount ?? null)}</strong>
                    {rowsResult.totals.bill_total ? (
                      <button type="button" onClick={() => focusBillTotal()}>
                        {rowsResult.totals.bill_total.label} · source p.{rowsResult.totals.bill_total.page_number} ↗
                      </button>
                    ) : (
                      <small>Not extracted from an explicit final-total label</small>
                    )}
                    {rowsResult.totals.printed_totals.length > 1 && (
                      <details className="printed-totals-list">
                        <summary>{rowsResult.totals.printed_totals.length} explicit totals found</summary>
                        {rowsResult.totals.printed_totals.map((total, index) => (
                          <button
                            type="button"
                            key={`${total.page_number}-${total.label}-${total.amount}-${index}`}
                            onClick={() => focusBillTotal(total)}
                          >
                            {total.is_primary ? "Primary · " : ""}{total.label} · {money(total.amount)} · p.{total.page_number}
                          </button>
                        ))}
                      </details>
                    )}
                  </article>
                  <article className="difference-total">
                    <p>Difference</p>
                    <strong>
                      {rowsResult.totals.difference === null
                        ? "—"
                        : money(String(Math.abs(Number(rowsResult.totals.difference))))}
                    </strong>
                    <small>{differenceCopy(rowsResult.totals.difference, rowsResult.totals.comparison)}</small>
                  </article>
                </section>

                <div className="review-toolbar">
                  <input aria-label="Search rows" placeholder="Search charge, section, code…" value={query} onChange={(event) => { setQuery(event.target.value); setOffset(0); }} />
                  <select aria-label="Filter disposition" disabled={ledgerMode === "printed"} title={ledgerMode === "printed" ? "Disposition applies to normalized rows" : undefined} value={disposition} onChange={(event) => { setDisposition(event.target.value); setOffset(0); }}>
                    <option value="">All dispositions</option><option value="accepted">Accepted</option><option value="pending">Pending</option><option value="rejected">Rejected</option><option value="unreadable">Unreadable</option>
                  </select>
                  <input className="page-filter" aria-label="Filter page" type="number" min="1" max={rowsResult.pages} placeholder="Page" value={pageFilter} onChange={(event) => { setPageFilter(event.target.value); setOffset(0); }} />
                  <button className="toolbar-action" onClick={() => { setAddMode(true); setEditMode(false); setHospitalEditMode(false); setReason(""); setDraftPolygon(null); setDrawMode("add"); }}>+ Add grounded row</button>
                  <button className="toolbar-action subtle" onClick={() => void deleteJob(selectedJob)}>Delete document</button>
                </div>

                <div
                  className={`split-view ${resizingSplit ? "resizing" : ""}`}
                  ref={splitView}
                  style={{ "--ledger-share": `${splitPercent}%` } as CSSProperties}
                >
                  <div className="ledger-pane">
                    <div className="ledger-caption">
                      <span>{selectedJob.original_name}</span>
                      <span className="ledger-caption-actions">
                        <span>{ledgerTotal.toLocaleString("en-IN")} matching · revision {review.revision}</span>
                        <span className="ledger-mode-switch" role="group" aria-label="Ledger columns">
                          <button
                            type="button"
                            className={ledgerMode === "printed" ? "active" : ""}
                            disabled={!printedAvailable}
                            onClick={() => {
                              setLedgerMode("printed");
                              setOffset(0);
                            }}
                          >
                            Printed columns
                          </button>
                          <button
                            type="button"
                            className={ledgerMode === "normalized" ? "active" : ""}
                            onClick={() => {
                              setLedgerMode("normalized");
                              setOffset(0);
                            }}
                          >
                            Normalized
                          </button>
                        </span>
                      </span>
                    </div>
                    {printedUnavailableCopy && (
                      <div className="printed-unavailable" role="note">
                        {printedUnavailableCopy}
                      </div>
                    )}
                    <div className="table-shell">
                      {ledgerMode === "printed" ? (
                        <>
                          {sourceTablesResult?.tables.map((table) => (
                            <section className="source-table-group" key={table.id}>
                              <div className="source-table-label">
                                <span>Page {table.page_number} · {table.table_type.replaceAll("_", " ")}</span>
                                <span>
                                  {table.columns.some((column) => column.canonical_field === null)
                                    ? "Unmapped columns retained"
                                    : "All columns mapped"}
                                </span>
                              </div>
                              <table
                                className="source-table"
                                style={{ minWidth: `${Math.max(720, 56 + table.columns.length * 140)}px` }}
                              >
                                <thead>
                                  <tr>
                                    <th>#</th>
                                    {table.columns.map((column) => (
                                      <th key={column.id}>{column.label}</th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {table.rows.map((row) => (
                                    <tr
                                      key={row.id}
                                      className={selectedSourceRowId === row.id ? "selected" : ""}
                                      onClick={() => {
                                        setSelectedSourceRowId(row.id);
                                        setSelectedRowId(row.canonical_row_id);
                                        setViewPage(table.page_number);
                                        setEditMode(false);
                                        setAddMode(false);
                                        setHospitalEditMode(false);
                                        setFocusedPrintedTotal(null);
                                        setDraftPolygon(null);
                                      }}
                                    >
                                      <td>{String(row.ordinal).padStart(2, "0")}</td>
                                      {table.columns.map((column) => {
                                        const cell = row.cells.find(
                                          (candidate) => candidate.column_id === column.id,
                                        );
                                        return <td key={column.id}>{cell?.raw_value || "—"}</td>;
                                      })}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </section>
                          ))}
                          {!sourceTablesResult?.tables.length && (
                            <div className="empty-ledger">
                              {printedAvailable
                                ? "No printed rows match this view."
                                : printedUnavailableCopy}
                            </div>
                          )}
                        </>
                      ) : (
                        <>
                          <table>
                            <colgroup>
                              <col className="number-column" />
                              <col className="description-column" />
                              <col className="date-column" />
                              <col className="quantity-column" />
                              <col className="money-column" />
                              <col className="money-column" />
                              <col className="money-column" />
                            </colgroup>
                            <thead><tr><th>#</th><th>Service / charge</th><th>Date</th><th>Qty</th><th>Rate</th><th>Gross</th><th>Net amount</th></tr></thead>
                            <tbody>
                              {rowsResult.rows.map((row, index) => (
                                <tr key={row.id} className={`${selectedRowId === row.id ? "selected" : ""} ${row.role} ${row.review_disposition} ${row.review.modified ? "modified" : ""}`} onClick={() => { setSelectedRowId(row.id); setEditMode(false); setAddMode(false); setHospitalEditMode(false); setFocusedPrintedTotal(null); setDraftPolygon(null); }}>
                                  <td>{String(offset + index + 1).padStart(2, "0")}</td>
                                  <td>
                                    <b>{row.description || "Unlabelled row"}</b>
                                    <small>
                                      <i className={row.review_disposition} /> {row.role === "informational" ? "included in package · informational" : row.review_disposition} · p.{row.page_number}
                                      {row.review.modified ? " · reviewer changed" : ""}
                                      {row.validation_flags?.some((flag) => ["missing_labeled_quantity", "missing_labeled_unit_price", "line_arithmetic_mismatch"].includes(flag))
                                        ? " · field warning"
                                        : ""}
                                    </small>
                                  </td>
                                  <td className="service-date" title={row.service_date_raw ?? undefined}>{serviceDate(row.service_date_iso, row.service_date_raw)}</td>
                                  <td>{row.quantity ?? "—"}</td><td>{money(row.unit_price)}</td><td>{money(row.gross_amount)}</td><td>{money(row.net_amount)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {!rowsResult.rows.length && <div className="empty-ledger">No rows match this view.</div>}
                        </>
                      )}
                    </div>
                    <div className="pagination">
                      <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>← Previous</button>
                      <span>{ledgerTotal ? `${offset + 1}–${Math.min(offset + PAGE_SIZE, ledgerTotal)} of ${ledgerTotal}` : "0 rows"}</span>
                      <button disabled={offset + PAGE_SIZE >= ledgerTotal} onClick={() => setOffset(offset + PAGE_SIZE)}>Next →</button>
                    </div>
                  </div>

                  <button
                    type="button"
                    className="split-resizer"
                    aria-label="Resize ledger and evidence panels"
                    aria-valuemin={44}
                    aria-valuemax={68}
                    aria-valuenow={Math.round(splitPercent)}
                    role="separator"
                    onPointerDown={(event) => {
                      event.currentTarget.setPointerCapture(event.pointerId);
                      setResizingSplit(true);
                    }}
                    onPointerMove={resize}
                    onPointerUp={(event) => {
                      setResizingSplit(false);
                      event.currentTarget.releasePointerCapture(event.pointerId);
                    }}
                    onPointerCancel={() => setResizingSplit(false)}
                    onKeyDown={resizeWithKeyboard}
                  ><span /></button>

                  <aside className="evidence-pane" ref={evidencePane}>
                    <div className="evidence-head">
                      <span>Source page {viewPage} / {rowsResult.pages}</span>
                      <span className="evidence-controls">
                        <span className="page-switch"><button aria-label="Previous page" disabled={viewPage <= 1} onClick={() => setViewPage(viewPage - 1)}>←</button><button aria-label="Next page" disabled={viewPage >= rowsResult.pages} onClick={() => setViewPage(viewPage + 1)}>→</button></span>
                        <span className="view-switch">
                          <button className={viewMode === "fit-page" ? "active" : ""} onClick={() => setViewMode("fit-page")}>Page</button>
                          <button className={viewMode === "fit-width" ? "active" : ""} onClick={() => setViewMode("fit-width")}>Width</button>
                          <button aria-label="Zoom out" onClick={() => { setViewMode("zoom"); setZoom((value) => Math.max(75, value - 25)); }}>−</button>
                          <output>{viewMode === "zoom" ? `${zoom}%` : "Fit"}</output>
                          <button aria-label="Zoom in" onClick={() => { setViewMode("zoom"); setZoom((value) => Math.min(300, value + 25)); }}>+</button>
                          <button onClick={focusEvidence}>Focus</button>
                          <button aria-label="Toggle fullscreen evidence" onClick={() => void toggleFullscreen()}>⛶</button>
                        </span>
                      </span>
                    </div>
                    <div className={`page-frame ${viewMode} ${drawMode ? "drawing" : ""}`} ref={pageFrame}>
                      <div
                        className="page-canvas"
                        ref={pageCanvas}
                        style={pageAsset ? { aspectRatio: `${pageAsset.width} / ${pageAsset.height}`, "--page-zoom": `${zoom}%` } as CSSProperties : undefined}
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element -- immutable evidence is intentionally not optimized */}
                        <img src={`/api/v2/documents/${selectedJob.id}/pages/${viewPage}`} alt={`Rendered bill page ${viewPage}`} />
                        {pageAsset && (
                          <svg viewBox={`0 0 ${pageAsset.width} ${pageAsset.height}`} preserveAspectRatio="none" aria-label={drawMode ? "Draw row evidence" : "Linked row evidence"} onPointerDown={startDraw} onPointerMove={moveDraw} onPointerUp={endDraw}>
                            {evidence.description && <polygon className="description-evidence" points={evidence.description} />}
                            {evidence.amount && evidence.amount !== evidence.description && <polygon className="amount-evidence" points={evidence.amount} />}
                            {evidence.total && <polygon className="total-evidence" points={evidence.total} />}
                            {hospitalEvidence && !draftPolygon && <polygon className="hospital-evidence" points={hospitalEvidence} />}
                            {draftPolygon && <polygon className="draft-evidence" points={draftPolygon.map((point) => `${point.x},${point.y}`).join(" ")} />}
                          </svg>
                        )}
                      </div>
                    </div>
                    {drawMode && <div className="draw-instruction">Drag across the visible source row to anchor the reviewer evidence.</div>}
                    {hospitalEditMode && hospital && (
                      <div className="evidence-note hospital-note">
                        <span>H</span>
                        <div><strong>{hospital.name}</strong><p>Hospital header · grounded on page {hospital.page_number}</p></div>
                        <button onClick={focusEvidence}>Focus header ↗</button>
                      </div>
                    )}
                    {totalEvidenceActive && focusedPrintedTotal && (
                      <div className="evidence-note total-note">
                        <span>Σ</span>
                        <div><strong>{focusedPrintedTotal.label}</strong><p>{money(focusedPrintedTotal.amount)} · printed on page {focusedPrintedTotal.page_number}</p></div>
                        <button onClick={focusEvidence}>Focus total ↗</button>
                      </div>
                    )}
                    {ledgerMode === "printed" && selectedSource && !addMode && !hospitalEditMode && !totalEvidenceActive && (
                      <div className="evidence-note">
                        <span>P{selectedSource.table.page_number}</span>
                        <div>
                          <strong>{selectedSource.row.cells.find((cell) => cell.raw_value)?.raw_value ?? "Printed row"}</strong>
                          <p>Raw printed row · {selectedSource.table.columns.length} source columns retained</p>
                        </div>
                        <button onClick={focusEvidence}>Focus row ↗</button>
                      </div>
                    )}
                    {ledgerMode === "normalized" && selectedRow && !addMode && !hospitalEditMode && !totalEvidenceActive && (
                      <div className="evidence-note">
                        <span>{String((rowsResult.rows.findIndex((row) => row.id === selectedRow.id) + offset + 1)).padStart(2, "0")}</span>
                        <div><strong>{selectedRow.description}</strong><p>{selectedRow.role === "informational" ? "Included package component · no separate printed amount" : money(selectedRow.net_amount)} · grounded on page {selectedRow.page_number}</p></div>
                        <button onClick={beginEdit}>Review row ↗</button>
                      </div>
                    )}
                  </aside>
                </div>

                {(editMode || addMode || hospitalEditMode) && (
                  <div className="review-sheet">
                    <div className="sheet-title"><p className="folio">04 / Human review</p><h3>{hospitalEditMode ? "Verify the hospital identity" : addMode ? "Add a missing grounded row" : "Correct without erasing the machine record"}</h3></div>
                    {hospitalEditMode ? (
                      <div className="edit-grid hospital-edit-grid">
                        <label className="wide"><span>Hospital name</span><input value={hospitalName} onChange={(event) => setHospitalName(event.target.value)} /></label>
                        <label className="wide"><span>Review reason</span><input value={hospitalReason} onChange={(event) => setHospitalReason(event.target.value)} placeholder="What did you verify in the bill header?" /></label>
                      </div>
                    ) : addMode ? (
                      <div className="edit-grid">
                        <label><span>Description</span><input value={addDescription} onChange={(event) => setAddDescription(event.target.value)} /></label>
                        <label><span>Net amount</span><input inputMode="decimal" value={addAmount} onChange={(event) => setAddAmount(event.target.value)} /></label>
                        <label className="wide"><span>Review reason</span><input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Why was this row added?" /></label>
                      </div>
                    ) : (
                      <div className="edit-grid">
                        <label className="wide"><span>Description</span><input value={editValues.description} onChange={(event) => setEditValues({ ...editValues, description: event.target.value })} /></label>
                        <label><span>Service date</span><input type="date" value={editValues.service_date_iso} onChange={(event) => setEditValues({ ...editValues, service_date_iso: event.target.value })} /></label>
                        <label><span>Quantity</span><input value={editValues.quantity} onChange={(event) => setEditValues({ ...editValues, quantity: event.target.value })} /></label>
                        <label><span>Unit rate</span><input value={editValues.unit_price} onChange={(event) => setEditValues({ ...editValues, unit_price: event.target.value })} /></label>
                        <label><span>Gross amount</span><input value={editValues.gross_amount} onChange={(event) => setEditValues({ ...editValues, gross_amount: event.target.value })} /></label>
                        <label><span>Discount</span><input value={editValues.discount} onChange={(event) => setEditValues({ ...editValues, discount: event.target.value })} /></label>
                        <label><span>Net amount</span><input value={editValues.net_amount} onChange={(event) => setEditValues({ ...editValues, net_amount: event.target.value })} /></label>
                        <label><span>Role</span><select value={editValues.role} onChange={(event) => setEditValues({ ...editValues, role: event.target.value })}><option value="detail">Detail</option><option value="informational">Informational</option><option value="category_rollup">Category rollup</option><option value="refund">Refund</option></select></label>
                        <label><span>Disposition</span><select value={editValues.review_disposition} onChange={(event) => setEditValues({ ...editValues, review_disposition: event.target.value })}><option value="accepted">Accepted</option><option value="pending">Pending</option><option value="unreadable">Unreadable</option></select></label>
                        <label className="wide"><span>Review reason</span><input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="What did you verify or change?" /></label>
                      </div>
                    )}
                    <div className="sheet-actions">
                      <button className="secondary" onClick={() => { setDrawMode(hospitalEditMode ? "hospital" : addMode ? "add" : "relink"); setDraftPolygon(null); }}>{draftPolygon ? "Redraw evidence" : hospitalEditMode ? "Draw header evidence" : addMode ? "Draw evidence above" : "Relink evidence"}</button>
                      <button className="secondary" onClick={() => { setEditMode(false); setAddMode(false); setHospitalEditMode(false); setDrawMode(null); setDraftPolygon(null); }}>Cancel</button>
                      {!addMode && !hospitalEditMode && <button className="danger" onClick={() => void rejectSelected()}>Reject row</button>}
                      <button className="primary" onClick={() => void (hospitalEditMode ? saveHospital() : addMode ? addRow() : saveEdit())}>{hospitalEditMode ? "Save hospital identity" : addMode ? "Add grounded row" : "Save correction"}</button>
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
