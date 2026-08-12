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
  alias_count: number;
  training_sources: string[];
};
type TrainedHospitalsResult = {
  registry_revision: number;
  profile_revision: number;
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
  request_no: string | null;
  service_code: string | null;
  hsn_code: string | null;
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
  bulk_action: "reject" | "restore" | null;
  rejection_provenance: {
    source: "reviewer";
    previous_disposition: string;
    legacy_fallback: boolean;
  } | null;
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
  populated_fields: string[];
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
  hospital_id: string | null;
  approval: { status: string; approved_at: string; review_revision: number } | null;
};
type AliasCandidate = {
  candidate_id: string;
  source_table_id: string;
  source_row_id: string;
  source_column_id: string;
  row_id: string | null;
  source_value: string | null;
  classification: "fillable" | "unchanged" | "conflicting" | "invalid" | "unlinked";
  current_value?: string | null;
  proposed_value?: string | null;
};
type AliasPreview = {
  hospital_id: string;
  source_label: string;
  canonical_field: string;
  review_revision: number;
  registry_revision: number;
  source_digest: string;
  counts: Record<AliasCandidate["classification"], number>;
  candidates: AliasCandidate[];
};
type HospitalAlias = {
  alias_id: string;
  source_label: string;
  canonical_field: string;
  active: boolean;
  reason: string;
};
type Health = {
  profile_revision: number;
  alias_registry_revision: number;
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
type AppView = "dashboard" | "active" | "history" | "hospitals" | "review";
type IconName =
  | "activity"
  | "archive"
  | "building"
  | "chevron-left"
  | "chevron-right"
  | "close"
  | "dashboard"
  | "file"
  | "menu"
  | "plus"
  | "search"
  | "upload";

const iconPaths: Record<IconName, React.ReactNode> = {
  activity: <><path d="M3 12h4l2.4-6 4.2 12 2.4-6h5" /></>,
  archive: <><rect x="3" y="5" width="18" height="15" rx="2" /><path d="M3 9h18M9 13h6" /></>,
  building: <><path d="M4 21V5l8-3 8 3v16M9 21v-4h6v4M8 7h.01M12 7h.01M16 7h.01M8 11h.01M12 11h.01M16 11h.01" /></>,
  "chevron-left": <path d="m15 18-6-6 6-6" />,
  "chevron-right": <path d="m9 18 6-6-6-6" />,
  close: <path d="M6 6l12 12M18 6 6 18" />,
  dashboard: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
  file: <><path d="M6 2h8l4 4v16H6z" /><path d="M14 2v5h5M9 12h6M9 16h6" /></>,
  menu: <path d="M4 7h16M4 12h16M4 17h16" />,
  plus: <path d="M12 5v14M5 12h14" />,
  search: <><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></>,
  upload: <><path d="M12 16V4m0 0L7 9m5-5 5 5" /><path d="M5 14v6h14v-6" /></>,
};

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {iconPaths[name]}
    </svg>
  );
}

function AonamiMark() {
  return <span className="aonami-mark" role="img" aria-label="Aonami" />;
}

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

const editValuesForRow = (row: Row): EditValues => ({
  description: row.description ?? "",
  service_date_iso: row.service_date_iso ?? "",
  quantity: row.quantity ?? "",
  unit_price: row.unit_price ?? "",
  gross_amount: row.gross_amount ?? "",
  discount: row.discount ?? "",
  net_amount: row.net_amount ?? "",
  role: row.role,
  review_disposition: row.review_disposition,
});

const changedEditValues = (row: Row, edited: EditValues) => {
  const original = editValuesForRow(row);
  return Object.fromEntries(
    Object.entries(edited).filter(
      ([field, value]) => original[field as keyof EditValues] !== value,
    ),
  );
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
  const [appView, setAppView] = useState<AppView>("dashboard");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [activeJobs, setActiveJobs] = useState<Job[]>([]);
  const [historyJobs, setHistoryJobs] = useState<Job[]>([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyQuery, setHistoryQuery] = useState("");
  const [trainedHospitals, setTrainedHospitals] = useState<TrainedHospital[]>([]);
  const [aliasRegistryRevision, setAliasRegistryRevision] = useState(0);
  const [profileRegistryRevision, setProfileRegistryRevision] = useState<number | null>(null);
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
  const [disposition, setDisposition] = useState("active");
  const [pageFilter, setPageFilter] = useState("");
  const [offset, setOffset] = useState(0);
  const [viewPage, setViewPage] = useState(1);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editMode, setEditMode] = useState(false);
  const [editValues, setEditValues] = useState<EditValues>(emptyEdit);
  const [reason, setReason] = useState("");
  const [bulkRowIds, setBulkRowIds] = useState<Set<string>>(new Set());
  const [bulkAction, setBulkAction] = useState<"reject" | "restore" | null>(null);
  const [bulkReason, setBulkReason] = useState("");
  const [addMode, setAddMode] = useState(false);
  const [addDescription, setAddDescription] = useState("");
  const [addAmount, setAddAmount] = useState("");
  const [issueReason, setIssueReason] = useState("");
  const [hospitalEditMode, setHospitalEditMode] = useState(false);
  const [focusedPrintedTotal, setFocusedPrintedTotal] = useState<PrintedTotal | null>(null);
  const totalEvidenceActive = focusedPrintedTotal !== null;
  const [hospitalName, setHospitalName] = useState("");
  const [hospitalReason, setHospitalReason] = useState("");
  const [aliasColumn, setAliasColumn] = useState<SourceColumn | null>(null);
  const [aliasPanelOpen, setAliasPanelOpen] = useState(false);
  const [aliasTarget, setAliasTarget] = useState("net_amount");
  const [aliasReason, setAliasReason] = useState("");
  const [hospitalChoice, setHospitalChoice] = useState("create");
  const [aliasPreview, setAliasPreview] = useState<AliasPreview | null>(null);
  const [aliasSelected, setAliasSelected] = useState<Set<string>>(new Set());
  const [hospitalAliases, setHospitalAliases] = useState<HospitalAlias[]>([]);
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
  const selectedJobCache = useRef<Job | null>(null);
  const drawerPanel = useRef<HTMLElement | null>(null);
  const menuTrigger = useRef<HTMLButtonElement | null>(null);
  const drawerWasOpen = useRef(false);

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
  const currentSelectedJob = jobs.find((job) => job.id === selectedJobId) ?? null;
  const selectedJob =
    currentSelectedJob
    ?? (selectedJobCache.current?.id === selectedJobId ? selectedJobCache.current : null);
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

  useEffect(() => {
    if (currentSelectedJob) selectedJobCache.current = currentSelectedJob;
    if (!selectedJobId) selectedJobCache.current = null;
  }, [currentSelectedJob, selectedJobId]);

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
      setAliasRegistryRevision(result.registry_revision);
      setProfileRegistryRevision(result.profile_revision);
      setTrainedHospitalsError(null);
      return result.hospitals;
    } catch (cause) {
      setProfileRegistryRevision(null);
      setTrainedHospitalsError(
        cause instanceof Error ? cause.message : "The trained hospital list could not be loaded.",
      );
      return [];
    }
  }, []);

  useEffect(() => {
    void refreshActiveJobs().catch(() => setError("The active bill queue could not be loaded."));
    void refreshTrainedHospitals();
    refreshHealth();
  }, [refreshActiveJobs, refreshHealth, refreshTrainedHospitals]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refreshHistoryJobs().catch(() => setError("Bill history could not be loaded."));
    }, 220);
    return () => window.clearTimeout(timer);
  }, [refreshHistoryJobs]);

  useEffect(() => {
    if (appView === "hospitals") void refreshTrainedHospitals();
  }, [appView, refreshTrainedHospitals]);

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
    if (!drawerOpen) {
      if (drawerWasOpen.current) menuTrigger.current?.focus();
      drawerWasOpen.current = false;
      return;
    }
    drawerWasOpen.current = true;
    drawerPanel.current?.querySelector<HTMLButtonElement>("nav button")?.focus();
    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") setDrawerOpen(false);
      if (event.key !== "Tab" || !drawerPanel.current) return;
      const controls = [...drawerPanel.current.querySelectorAll<HTMLElement>("button, [href], input, select, [tabindex]:not([tabindex='-1'])")];
      if (!controls.length) return;
      const first = controls[0];
      const last = controls.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [drawerOpen]);

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
    setDisposition("active");
    setPageFilter("");
    setEditMode(false);
    setAddMode(false);
    setHospitalEditMode(false);
    setFocusedPrintedTotal(null);
    setDrawMode(null);
    setDraftPolygon(null);
    setSelectedSourceRowId(null);
    setLedgerMode("printed");
    setBulkRowIds(new Set());
    setBulkAction(null);
    setBulkReason("");
    setAliasColumn(null);
    setAliasPanelOpen(false);
    setAliasPreview(null);
    setAliasSelected(new Set());
    setHospitalAliases([]);
  }, [selectedJobId]);

  useEffect(() => {
    setBulkRowIds(new Set());
    setBulkAction(null);
  }, [offset, query, disposition, pageFilter, ledgerMode]);

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
        setAppView("review");
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
    setEditValues(editValuesForRow(selectedRow));
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
    const changes = changedEditValues(selectedRow, editValues);
    if (!Object.keys(changes).length && !draftPolygon) {
      setError("Change at least one row value or relink its evidence before saving.");
      return;
    }
    const payload: Record<string, unknown> = { changes, reason };
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

  const applyBulkRows = async () => {
    if (!selectedJob || !bulkAction || !bulkRowIds.size || bulkReason.trim().length < 3) {
      setError("Select rows and add a short review reason.");
      return;
    }
    const updated = await mutate(`/api/v2/documents/${selectedJob.id}/rows/bulk`, {
      method: "PATCH",
      body: JSON.stringify({
        row_ids: [...bulkRowIds],
        action: bulkAction,
        reason: bulkReason,
      }),
    });
    if (updated) {
      setBulkRowIds(new Set());
      setBulkAction(null);
      setBulkReason("");
    }
  };

  const linkAliasHospital = async () => {
    if (profileRegistryRevision === null) {
      setError("The trained hospital registry is still loading.");
      return;
    }
    if (!selectedJob || aliasReason.trim().length < 3) {
      setError("Add a short reason for linking this hospital.");
      return;
    }
    const create = hospitalChoice === "create";
    const linked = await mutate<{ hospital_id: string }>(
      `/api/v2/documents/${selectedJob.id}/hospital-link`,
      {
        method: "POST",
        body: JSON.stringify({
          create,
          hospital_id: create ? null : hospitalChoice,
          registry_revision: aliasRegistryRevision,
          profile_revision: profileRegistryRevision,
          reason: aliasReason,
        }),
      },
    );
    if (linked) {
      setHospitalChoice(linked.hospital_id);
      setAliasPreview(null);
      void refreshTrainedHospitals();
    } else {
      void refreshTrainedHospitals();
    }
  };

  const previewColumnAlias = async () => {
    if (!selectedJob || !review?.hospital_id || !aliasColumn) return;
    try {
      const preview = await request<AliasPreview>(
        `/api/v2/documents/${selectedJob.id}/column-aliases/preview`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            hospital_id: review.hospital_id,
            source_label: aliasColumn.label,
            canonical_field: aliasTarget,
          }),
        },
      );
      setAliasPreview(preview);
      setAliasRegistryRevision(preview.registry_revision);
      const selected = new Set<string>();
      const selectedTargets = new Set<string>();
      for (const candidate of preview.candidates) {
        if (!["fillable", "unchanged"].includes(candidate.classification) || !candidate.row_id) continue;
        if (selectedTargets.has(candidate.row_id)) continue;
        selectedTargets.add(candidate.row_id);
        selected.add(candidate.candidate_id);
      }
      setAliasSelected(selected);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Alias preview failed");
    }
  };

  const applyColumnAlias = async () => {
    if (!selectedJob || !aliasPreview || !aliasSelected.size || aliasReason.trim().length < 3) {
      setError("Select at least one grounded row and add a short mapping reason.");
      return;
    }
    const applied = await mutate(
      `/api/v2/documents/${selectedJob.id}/column-aliases/apply`,
      {
        method: "POST",
        body: JSON.stringify({
          hospital_id: aliasPreview.hospital_id,
          source_label: aliasPreview.source_label,
          canonical_field: aliasPreview.canonical_field,
          source_digest: aliasPreview.source_digest,
          registry_revision: aliasPreview.registry_revision,
          selected_candidate_ids: [...aliasSelected],
          reason: aliasReason,
        }),
      },
    );
    if (applied) {
      setAliasColumn(null);
      setAliasPanelOpen(false);
      setAliasPreview(null);
      setAliasSelected(new Set());
      setAliasReason("");
      void refreshTrainedHospitals();
    } else {
      setAliasPreview(null);
      setAliasSelected(new Set());
      void refreshTrainedHospitals();
    }
  };

  const loadHospitalAliases = async () => {
    if (!review?.hospital_id) return;
    try {
      const response = await request<{ registry_revision: number; aliases: HospitalAlias[] }>(
        `/api/v2/hospitals/${review.hospital_id}/aliases`,
      );
      setHospitalAliases(response.aliases);
      setAliasRegistryRevision(response.registry_revision);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Aliases could not be loaded");
    }
  };

  const updateHospitalAlias = async (
    alias: HospitalAlias,
    changes: { active?: boolean; canonical_field?: string },
  ) => {
    if (!review?.hospital_id || aliasReason.trim().length < 3) {
      setError("Add a short reason before changing an alias.");
      return;
    }
    try {
      await request(`/api/v2/hospitals/${review.hospital_id}/aliases/${alias.alias_id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...changes,
          registry_revision: aliasRegistryRevision,
          reason: aliasReason,
        }),
      });
      await loadHospitalAliases();
      void refreshTrainedHospitals();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Alias could not be changed");
      await loadHospitalAliases();
      void refreshTrainedHospitals();
    }
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
      if (selectedJobId === job.id) {
        setSelectedJobId(null);
        setRailView("history");
        setAppView("history");
      }
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
      if (selectedJobId === job.id) {
        setSelectedJobId(null);
        setRailView("active");
        setAppView("active");
      }
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

  const openSection = (view: Exclude<AppView, "review">) => {
    setAppView(view);
    setDrawerOpen(false);
    if (view === "active" || view === "history" || view === "hospitals") {
      setRailView(view);
    }
  };

  const openJob = (job: Job) => {
    setSelectedJobId(job.id);
    setAppView("review");
    setDrawerOpen(false);
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
  const visibleBulkAction: "reject" | "restore" =
    disposition === "rejected" ? "restore" : "reject";
  const visibleRowIds =
    rowsResult?.rows
      .filter((row) => row.bulk_action === visibleBulkAction)
      .map((row) => row.id) ?? [];
  const allVisibleRowsSelected =
    visibleRowIds.length > 0 && visibleRowIds.every((rowId) => bulkRowIds.has(rowId));
  const optionalFields = new Set(rowsResult?.populated_fields ?? []);
  const workerCapacity = health?.worker_capacity ?? 1;
  const activeProcessing = activeJobs.filter((job) => job.status === "processing").length;
  const pageTitle =
    appView === "dashboard"
      ? "Evidence dashboard"
      : appView === "active"
        ? "Active bills"
        : appView === "history"
          ? "Bill history"
          : appView === "hospitals"
            ? "Trained hospitals"
            : (selectedJob?.hospital_name ?? "Bill review");

  return (
    <div className="app-shell">
      <button
        className={`drawer-backdrop ${drawerOpen ? "open" : ""}`}
        aria-label="Close navigation"
        tabIndex={drawerOpen ? 0 : -1}
        onClick={() => setDrawerOpen(false)}
      />
      <aside id="primary-navigation" ref={drawerPanel} className={`shell-sidebar ${drawerOpen ? "open" : ""}`} aria-label="Primary navigation">
        <div className="brand-lockup">
          <AonamiMark />
          <div><strong>GMONEY</strong><span>BY AONAMI</span></div>
        </div>
        <nav className="primary-nav">
          <button aria-current={appView === "dashboard" ? "page" : undefined} onClick={() => openSection("dashboard")}><Icon name="dashboard" />Dashboard</button>
          <button aria-current={appView === "active" ? "page" : undefined} onClick={() => openSection("active")}><Icon name="activity" />Active bills <span>{activeJobs.length}</span></button>
          <button aria-current={appView === "history" ? "page" : undefined} onClick={() => openSection("history")}><Icon name="archive" />History <span>{historyTotal}</span></button>
          <button aria-current={appView === "hospitals" ? "page" : undefined} onClick={() => openSection("hospitals")}><Icon name="building" />Hospitals <span>{trainedHospitals.length}</span></button>
        </nav>
        <div className="sidebar-foot">
          <div className={`service-light ${health ? "online" : ""}`}><i />{health ? "GPU service online" : "Inference unavailable"}</div>
          <span>{health ? `${storageSize(health.storage_free_bytes)} storage free` : "Waiting for health check"}</span>
          <small>Evidence-grounded review</small>
        </div>
      </aside>

      <div className="shell-frame">
        <header className="shell-topbar">
          <button ref={menuTrigger} className="menu-button" aria-label="Open navigation" aria-controls="primary-navigation" aria-expanded={drawerOpen} onClick={() => setDrawerOpen(true)}><Icon name="menu" /></button>
          <div className="topbar-title"><span>GMoney workspace</span><strong>{pageTitle}</strong></div>
          <div className="topbar-actions">
            <div className={`health-chip ${health ? "online" : "offline"}`}><i />{health ? `${health.worker_capacity} lanes · ${health.active_jobs} active` : "Offline"}</div>
            <label className="topbar-upload">
              <input type="file" multiple accept="application/pdf,.pdf" onChange={acceptFiles} disabled={uploading} />
              <Icon name="upload" size={16} />{uploading ? "Uploading…" : "Upload bills"}
            </label>
          </div>
        </header>
        <div className="public-notice"><span>Public demo</span> No login · shared 30-day history · do not upload protected health information</div>

        <main className="app-main">
          {appView === "dashboard" ? (
            <section className="dashboard-view">
              <div className="dashboard-hero">
                <div>
                  <p className="eyebrow">Hospital bill intelligence</p>
                  <h1>Read every charge.<br />Resolve every doubt.</h1>
                  <p className="lede">Turn hospital bill PDFs into a grounded ledger where every accepted value remains linked to its exact source page.</p>
                </div>
                <div className="dashboard-trust"><i /><span>GPU extraction is {health ? "live" : "being checked"}</span><small>300 DPI evidence · revisioned review</small></div>
              </div>

              <div className="kpi-grid" aria-label="Workspace status">
                <article><span>Active lanes</span><strong>{activeProcessing}<small>/{workerCapacity}</small></strong><p>{activeJobs.length} bills in the active queue</p></article>
                <article><span>Retained bills</span><strong>{historyTotal}</strong><p>Searchable with source evidence</p></article>
                <article><span>Trained hospitals</span><strong>{trainedHospitals.length}</strong><p>Active hospital-specific profiles</p></article>
                <article><span>Storage free</span><strong>{storageSize(health?.storage_free_bytes)}</strong><p>Runtime capacity available</p></article>
              </div>

              <div className="dashboard-grid">
                <label
                  className={`drop-zone ${dragging ? "is-dragging" : ""}`}
                  onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={drop}
                >
                  <input type="file" multiple accept="application/pdf,.pdf" onChange={acceptFiles} disabled={uploading} />
                  <span className="upload-icon"><Icon name="upload" size={24} /></span>
                  <span className="eyebrow">PDF intake · multiple files</span>
                  <strong>{uploading ? "Transferring bills…" : "Drop hospital bills here"}</strong>
                  <small>or click to choose PDFs</small>
                  <em>Each bill is processed independently and retained for 30 days.</em>
                </label>
                <section className="recent-panel">
                  <div className="panel-heading"><div><p className="eyebrow">Latest evidence</p><h2>Recent bills</h2></div><button onClick={() => openSection("history")}>View all <Icon name="chevron-right" size={15} /></button></div>
                  <div className="recent-list">
                    {jobs.slice(0, 5).map((job) => (
                      <button key={job.id} onClick={() => openJob(job)}>
                        <span className={`job-state ${job.status}`} />
                        <span><b>{job.hospital_name ?? (job.status === "complete" ? "Hospital not identified" : "Identifying hospital…")}</b><small>{job.original_name}</small></span>
                        <span className="recent-meta">{job.status === "processing" ? `${progress(job)}%` : job.row_count !== null ? `${job.row_count} rows` : job.status}<Icon name="chevron-right" size={15} /></span>
                      </button>
                    ))}
                    {!jobs.length && <div className="dashboard-empty"><Icon name="file" size={28} /><span>No bills yet</span><small>Your first grounded ledger will appear here.</small></div>}
                  </div>
                </section>
              </div>
            </section>
          ) : (
        <section className={`review-desk ${appView === "review" ? "review-only" : appView === "hospitals" ? "hospital-view" : "index-view"}`}>
          {appView !== "hospitals" && appView !== "review" && <aside className="queue-rail">
            <div className="rail-head">
              <div>
                <p className="eyebrow">{railView === "active" ? "Live processing" : "Evidence archive"}</p>
                <h2>
                  {railView === "active" ? activeJobs.length : historyTotal} bills
                </h2>
                <p>{railView === "active" ? "Monitor processing and stop a bill safely." : "Search every retained bill by hospital or filename."}</p>
              </div>
              <label className="compact-upload">
                <input type="file" multiple accept="application/pdf,.pdf" onChange={acceptFiles} disabled={uploading} />
                <Icon name={uploading ? "activity" : "plus"} /> <span>{uploading ? "Uploading" : "Add bills"}</span>
              </label>
            </div>
            <div className="lane-meter">
              <span>{activeProcessing} / {health?.worker_capacity ?? 2} lanes occupied</span>
              <i style={{ width: `${Math.min(100, (activeProcessing / (health?.worker_capacity ?? 2)) * 100)}%` }} />
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
                  <button className={`job-card ${selectedJobId === job.id ? "selected" : ""}`} onClick={() => openJob(job)}>
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
              {!visibleJobs.length && (
                <div className="rail-empty">
                  {railView === "active" ? "No bills are running." : "No historical bills match."}
                </div>
              )}
              {railView === "history" && historyJobs.length < historyTotal && (
                <button className="load-history" onClick={() => void refreshHistoryJobs(historyJobs.length, true)}>
                  Load older bills
                </button>
              )}
            </div>
            <div className="rail-note">Public shared index · full evidence retained for 30 days</div>
          </aside>}

          <div className="desk-main">
            {appView === "hospitals" && (
              <section className="hospital-directory">
                <div className="directory-heading">
                  <div><p className="folio">02 / Training registry</p><h2>Hospitals the system has learned.</h2></div>
                  <button onClick={() => void refreshTrainedHospitals()}>Refresh registry</button>
                </div>
                <p className="directory-lede">Active layout profiles and reviewer-taught column vocabulary appear here. Candidate and archived training remains outside the production directory.</p>
                {trainedHospitalsError ? (
                  <div className="directory-empty">{trainedHospitalsError}</div>
                ) : trainedHospitals.length ? (
                  <div className="hospital-grid">
                    {trainedHospitals.map((item, index) => (
                      <article key={item.hospital_id}>
                        <span>{String(index + 1).padStart(2, "0")}</span>
                        <h3>{item.hospital_name}</h3>
                        <p>{item.active_profile_count} active layout {item.active_profile_count === 1 ? "profile" : "profiles"}</p>
                        <p className="alias-meta">{item.alias_count ?? 0} column {(item.alias_count ?? 0) === 1 ? "alias" : "aliases"}</p>
                        <em>{(item.training_sources ?? ["profile"]).map((source) => source.replaceAll("_", " ")).join(" + ")}</em>
                        <small>{item.hospital_id}</small>
                      </article>
                    ))}
                  </div>
                ) : (
                  <div className="directory-empty">No active hospital-specific profiles have been published.</div>
                )}
              </section>
            )}

            {appView === "review" && selectedJob && selectedJob.status !== "complete" && (
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

            {appView === "review" && selectedJob?.status === "complete" && rowsResult && review && (
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
                    <option value="active">Active rows</option><option value="">All dispositions</option><option value="accepted">Accepted</option><option value="pending">Pending</option><option value="rejected">Rejected</option><option value="unreadable">Unreadable</option>
                  </select>
                  <input className="page-filter" aria-label="Filter page" type="number" min="1" max={rowsResult.pages} placeholder="Page" value={pageFilter} onChange={(event) => { setPageFilter(event.target.value); setOffset(0); }} />
                  <button className="toolbar-action alias-action" onClick={() => { setAliasPanelOpen(true); setAliasColumn(null); setAliasPreview(null); void loadHospitalAliases(); }}>Column aliases</button>
                  <button className="toolbar-action" onClick={() => { setAddMode(true); setEditMode(false); setHospitalEditMode(false); setReason(""); setDraftPolygon(null); setDrawMode("add"); }}>+ Add grounded row</button>
                  <button className="toolbar-action subtle" onClick={() => void deleteJob(selectedJob)}>Delete document</button>
                </div>

                {ledgerMode === "normalized" && bulkRowIds.size > 0 && (
                  <div className="bulk-action-bar" role="region" aria-label="Bulk row actions">
                    <strong>{bulkRowIds.size} selected</strong>
                    <span>Only rows on this visible page are selected.</span>
                    <button onClick={() => setBulkAction(visibleBulkAction)}>
                      {visibleBulkAction === "restore" ? "Restore selected" : "Reject selected"}
                    </button>
                    <button className="quiet" onClick={() => setBulkRowIds(new Set())}>Clear</button>
                  </div>
                )}

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
                                      <th key={column.id} aria-label={column.label}>
                                        <button
                                          className={`column-alias-trigger ${column.canonical_field === null ? "unmapped" : ""}`}
                                          aria-label={column.label}
                                          title="Map this printed header to a normalized field"
                                          onClick={() => {
                                            setAliasColumn(column);
                                            setAliasTarget(column.canonical_field === "service_date_raw" ? "service_date" : (column.canonical_field ?? "net_amount"));
                                            setAliasPreview(null);
                                            setAliasSelected(new Set());
                                            setAliasPanelOpen(true);
                                            void loadHospitalAliases();
                                          }}
                                        >
                                          {column.label}<small>{column.canonical_field?.replaceAll("_", " ") ?? "unmapped"}</small>
                                        </button>
                                      </th>
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
                              <col className="select-column" />
                              <col className="number-column" />
                              <col className="description-column" />
                              {optionalFields.has("section") && <col />}
                              {optionalFields.has("request_no") && <col />}
                              {optionalFields.has("service_code") && <col />}
                              {optionalFields.has("hsn_code") && <col />}
                              <col className="date-column" />
                              <col className="quantity-column" />
                              <col className="money-column" />
                              <col className="money-column" />
                              {optionalFields.has("discount") && <col className="money-column" />}
                              <col className="money-column" />
                            </colgroup>
                            <thead><tr>
                              <th className="select-cell">
                                <input
                                  type="checkbox"
                                  aria-label="Select all rows on this page"
                                  disabled={!visibleRowIds.length}
                                  checked={allVisibleRowsSelected}
                                  onChange={(event) => setBulkRowIds(event.target.checked ? new Set(visibleRowIds) : new Set())}
                                />
                              </th>
                              <th>#</th><th>Service / charge</th>
                              {optionalFields.has("section") && <th>Section</th>}
                              {optionalFields.has("request_no") && <th>Request no.</th>}
                              {optionalFields.has("service_code") && <th>Service code</th>}
                              {optionalFields.has("hsn_code") && <th>HSN / SAC</th>}
                              <th>Date</th><th>Qty</th><th>Rate</th><th>Gross</th>
                              {optionalFields.has("discount") && <th>Discount</th>}
                              <th>Net amount</th>
                            </tr></thead>
                            <tbody>
                              {rowsResult.rows.map((row, index) => (
                                <tr key={row.id} className={`${selectedRowId === row.id ? "selected" : ""} ${bulkRowIds.has(row.id) ? "bulk-marked" : ""} ${row.role} ${row.review_disposition} ${row.review.modified ? "modified" : ""}`} onClick={() => { setSelectedRowId(row.id); setEditMode(false); setAddMode(false); setHospitalEditMode(false); setFocusedPrintedTotal(null); setDraftPolygon(null); }}>
                                  <td className="select-cell" onClick={(event) => event.stopPropagation()}>
                                    <input
                                      type="checkbox"
                                      aria-label={`Select ${row.description || "unlabelled row"}`}
                                      disabled={row.bulk_action !== visibleBulkAction}
                                      checked={bulkRowIds.has(row.id)}
                                      onChange={(event) => setBulkRowIds((current) => {
                                        const next = new Set(current);
                                        if (event.target.checked) next.add(row.id); else next.delete(row.id);
                                        return next;
                                      })}
                                    />
                                  </td>
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
                                  {optionalFields.has("section") && <td>{row.section ?? "—"}</td>}
                                  {optionalFields.has("request_no") && <td>{row.request_no ?? "—"}</td>}
                                  {optionalFields.has("service_code") && <td>{row.service_code ?? "—"}</td>}
                                  {optionalFields.has("hsn_code") && <td>{row.hsn_code ?? "—"}</td>}
                                  <td className="service-date" title={row.service_date_raw ?? undefined}>{serviceDate(row.service_date_iso, row.service_date_raw)}</td>
                                  <td>{row.quantity ?? "—"}</td><td>{money(row.unit_price)}</td><td>{money(row.gross_amount)}</td>
                                  {optionalFields.has("discount") && <td>{money(row.discount)}</td>}
                                  <td>{money(row.net_amount)}</td>
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

                {bulkAction && (
                  <section className="action-sheet bulk-confirm" role="dialog" aria-modal="true" aria-labelledby="bulk-action-title">
                    <div>
                      <p className="folio">Bulk review action</p>
                      <h3 id="bulk-action-title">{bulkAction === "reject" ? "Reject" : "Restore"} {bulkRowIds.size} rows?</h3>
                      <p>{bulkAction === "reject" ? "These rows will leave active totals and exports, while their machine evidence remains recoverable." : "The rows will return to their disposition from before rejection."}</p>
                    </div>
                    <label><span>Review reason</span><input autoFocus value={bulkReason} onChange={(event) => setBulkReason(event.target.value)} placeholder="Why are these rows being changed?" /></label>
                    <div className="sheet-actions">
                      <button className="secondary" onClick={() => { setBulkAction(null); setBulkReason(""); }}>Cancel</button>
                      <button className={bulkAction === "reject" ? "danger" : "primary"} onClick={() => void applyBulkRows()}>{bulkAction === "reject" ? "Reject selected rows" : "Restore selected rows"}</button>
                    </div>
                  </section>
                )}

                {aliasPanelOpen && (
                  <section className="action-sheet alias-sheet" role="dialog" aria-modal="true" aria-labelledby="alias-sheet-title">
                    <div className="alias-sheet-head">
                      <div><p className="folio">Hospital vocabulary</p><h3 id="alias-sheet-title">Column aliases</h3></div>
                      <button aria-label="Close column aliases" onClick={() => { setAliasPanelOpen(false); setAliasPreview(null); }}><Icon name="close" /></button>
                    </div>
                    {!review.hospital_id ? (
                      <div className="alias-link-step">
                        <p>Link this grounded bill identity to a durable hospital before teaching its printed vocabulary.</p>
                        <label><span>Hospital record</span><select value={hospitalChoice} onChange={(event) => setHospitalChoice(event.target.value)}><option value="create">Create from “{hospital?.name ?? "current hospital"}”</option>{trainedHospitals.map((item) => <option key={item.hospital_id} value={item.hospital_id}>{item.hospital_name}</option>)}</select></label>
                        <label><span>Review reason</span><input value={aliasReason} onChange={(event) => setAliasReason(event.target.value)} placeholder="How was this hospital identity verified?" /></label>
                        <button className="primary" disabled={profileRegistryRevision === null} onClick={() => void linkAliasHospital()}>Link hospital</button>
                      </div>
                    ) : aliasColumn ? (
                      <div className="alias-map-step">
                        <div className="alias-route"><span>Printed header</span><strong>{aliasColumn.label}</strong><i>→</i><label><span>Normalized field</span><select value={aliasTarget} onChange={(event) => { setAliasTarget(event.target.value); setAliasPreview(null); }}><option value="description">Service / charge</option><option value="section">Section</option><option value="service_date">Date</option><option value="request_no">Request no.</option><option value="service_code">Service code</option><option value="hsn_code">HSN / SAC</option><option value="quantity">Quantity</option><option value="unit_price">Rate</option><option value="gross_amount">Gross</option><option value="discount">Discount</option><option value="net_amount">Net amount</option></select></label></div>
                        <p className="alias-scope-note">Exact header match · this verified hospital only · original OCR evidence retained</p>
                        {!aliasPreview ? (
                          <button className="primary" onClick={() => void previewColumnAlias()}>Preview affected rows</button>
                        ) : (
                          <>
                            <div className="alias-counts"><span><b>{aliasPreview.counts.fillable}</b> fillable</span><span><b>{aliasPreview.counts.conflicting}</b> conflicts</span><span><b>{aliasPreview.counts.invalid}</b> invalid</span><span><b>{aliasPreview.counts.unlinked}</b> unlinked</span></div>
                            <div className="alias-conflicts"><strong>Select grounded values to apply</strong>{aliasPreview.candidates.map((item) => { const selectable = ["fillable", "unchanged", "conflicting"].includes(item.classification) && Boolean(item.row_id); return <label key={item.candidate_id} className={selectable ? item.classification : "disabled"}><input type="checkbox" disabled={!selectable} checked={aliasSelected.has(item.candidate_id)} onChange={(event) => setAliasSelected((current) => { const next = new Set(current); if (event.target.checked) { for (const candidate of aliasPreview.candidates) { if (candidate.row_id === item.row_id) next.delete(candidate.candidate_id); } next.add(item.candidate_id); } else next.delete(item.candidate_id); return next; })} /><span><b>{item.classification}</b> · {item.current_value || "—"} → {item.proposed_value || item.source_value || "—"}</span></label>; })}</div>
                            <label><span>Mapping reason</span><input value={aliasReason} onChange={(event) => setAliasReason(event.target.value)} placeholder="Why does this header map to this field?" /></label>
                            <div className="sheet-actions"><button className="secondary" onClick={() => setAliasPreview(null)}>Back</button><button className="primary" disabled={!aliasSelected.size} onClick={() => void applyColumnAlias()}>Apply {aliasSelected.size || "no"} selected and teach hospital</button></div>
                          </>
                        )}
                      </div>
                    ) : (
                      <div className="alias-manage-step">
                        <p>Click a header in <b>Printed columns</b> to preview a new mapping. Active aliases are applied to future bills only when this hospital is identified unambiguously.</p>
                        <label><span>Audit reason for alias changes</span><input value={aliasReason} onChange={(event) => setAliasReason(event.target.value)} placeholder="Why is this alias being changed?" /></label>
                        <div className="alias-list">{hospitalAliases.map((alias) => <article key={alias.alias_id} className={alias.active ? "active" : "inactive"}><div><strong>{alias.source_label}</strong><select aria-label={`Normalized field for ${alias.source_label}`} value={alias.canonical_field} onChange={(event) => void updateHospitalAlias(alias, { canonical_field: event.target.value })}><option value="description">Service / charge</option><option value="section">Section</option><option value="service_date">Date</option><option value="request_no">Request no.</option><option value="service_code">Service code</option><option value="hsn_code">HSN / SAC</option><option value="quantity">Quantity</option><option value="unit_price">Rate</option><option value="gross_amount">Gross</option><option value="discount">Discount</option><option value="net_amount">Net amount</option></select></div><button onClick={() => void updateHospitalAlias(alias, { active: !alias.active })}>{alias.active ? "Deactivate" : "Reactivate"}</button></article>)}{!hospitalAliases.length && <p className="alias-empty">No user-trained aliases for this hospital yet.</p>}</div>
                      </div>
                    )}
                  </section>
                )}

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
                        <label><span>Disposition</span><select value={editValues.review_disposition} onChange={(event) => setEditValues({ ...editValues, review_disposition: event.target.value })}>{selectedRow?.review_disposition === "rejected" && <option value="rejected">Rejected</option>}<option value="accepted">Accepted</option><option value="pending">Pending</option><option value="unreadable">Unreadable</option></select></label>
                        <label className="wide"><span>Review reason</span><input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="What did you verify or change?" /></label>
                      </div>
                    )}
                    <div className="sheet-actions">
                      <button className="secondary" onClick={() => { setDrawMode(hospitalEditMode ? "hospital" : addMode ? "add" : "relink"); setDraftPolygon(null); }}>{draftPolygon ? "Redraw evidence" : hospitalEditMode ? "Draw header evidence" : addMode ? "Draw evidence above" : "Relink evidence"}</button>
                      <button className="secondary" onClick={() => { setEditMode(false); setAddMode(false); setHospitalEditMode(false); setDrawMode(null); setDraftPolygon(null); }}>Cancel</button>
                      {!addMode && !hospitalEditMode && selectedRow?.bulk_action === "reject" && <button className="danger" onClick={() => void rejectSelected()}>Reject row</button>}
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
        </main>
        <footer><span>GMoney · BY AONAMI</span><span>Machine output remains immutable · reviewer changes are revisioned</span></footer>
      </div>
      {error && <div className="toast" role="alert"><span>{error}</span><button aria-label="Dismiss notification" onClick={() => setError(null)}><Icon name="close" size={16} /></button></div>}
    </div>
  );
}
