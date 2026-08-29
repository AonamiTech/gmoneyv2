import React from "react";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import Home from "./page";

const job = {
  id: "11111111-1111-4111-8111-111111111111",
  status: "complete",
  original_name: "bill.pdf",
  page: 2,
  pages: 2,
  row_count: 1,
  hospital_name: "Test Hospital",
  hospital_confidence: 0.99,
  hospital_name_source: "machine",
  certification_status: "passed",
  error: null,
  created_at: "2026-07-23T00:00:00Z",
  updated_at: "2026-07-23T00:00:00Z",
  last_activity_at: "2026-07-23T00:00:00Z",
  expires_at: "2026-08-23T00:00:00Z",
};

const evidence = {
  page_number: 1,
  polygon: {
    points: [
      { x: 10, y: 10 },
      { x: 90, y: 10 },
      { x: 90, y: 30 },
      { x: 10, y: 30 },
    ],
  },
  artifact_sha256: "a".repeat(64),
};

const rowsResult = {
  document_id: "d".repeat(64),
  pages: 2,
  page_assets: [
    { page_number: 1, artifact_sha256: "a".repeat(64), width: 100, height: 200 },
    { page_number: 2, artifact_sha256: "b".repeat(64), width: 100, height: 200 },
  ],
  hospital: null,
  review_revision: 0,
  totals: {
    items_total: "100.00",
    bill_total: null,
    printed_totals: [],
    difference: null,
    comparison: "bill_total_missing",
    missing_item_amounts: 0,
  },
  total: 1,
  offset: 0,
  limit: 150,
  rows: [
    {
      id: "row-1",
      description: "Consultation",
      service_date_raw: null,
      service_date_iso: null,
      section: null,
      quantity: "1",
      unit_price: "100.00",
      gross_amount: "100.00",
      discount: null,
      net_amount: "100.00",
      role: "detail",
      review_disposition: "accepted",
      page_number: 1,
      evidence: [evidence],
      field_evidence: { description: [evidence], amount: [evidence] },
      validation_flags: [],
      review: {
        source: "machine",
        modified: false,
        reason: null,
        machine_values: {},
      },
      bulk_action: "reject" as "reject" | "restore" | null,
      rejection_provenance: null,
    },
  ],
};

const review = {
  revision: 0,
  rows_total: 1,
  rows_active: 1,
  rows_modified: 0,
  rows_pending: 0,
  issues_open: 0,
  issues: [],
  hospital: null,
  hospital_id: null,
  approval: null,
  approval_effective: false,
  approval_blockers: [],
  export_eligible: false,
};

const sourceTables = {
  document_id: "d".repeat(64),
  available: true,
  unavailable_reason: null,
  total: 1,
  offset: 0,
  limit: 150,
  tables: [
    {
      id: "p1-t1-s1",
      page_number: 1,
      table_id: "p1-t1",
      table_type: "item_ledger",
      columns: [
        {
          id: "c1",
          label: "Particular",
          order: 0,
          canonical_field: "description",
          evidence: [evidence],
        },
        {
          id: "c2",
          label: "Co-pay %",
          order: 1,
          canonical_field: null,
          evidence: [evidence],
        },
        {
          id: "c3",
          label: "Amount",
          order: 2,
          canonical_field: "net_amount",
          evidence: [evidence],
        },
      ],
      rows: [
        {
          id: "p1-t1-s1-r1",
          order: 0,
          ordinal: 1,
          canonical_row_id: "row-1",
          cells: [
            {
              column_id: "c1",
              raw_value: "Consultation",
              evidence: [evidence],
              validation_flags: [],
            },
            {
              column_id: "c2",
              raw_value: "10",
              evidence: [evidence],
              validation_flags: [],
            },
            {
              column_id: "c3",
              raw_value: "100.00",
              evidence: [evidence],
              validation_flags: [],
            },
          ],
          validation_flags: [],
        },
      ],
      validation_flags: ["unmapped_columns"],
    },
  ],
};

const json = (payload: unknown) =>
  Promise.resolve(
    new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  );

async function advance(milliseconds: number) {
  await act(async () => {
    vi.advanceTimersByTime(milliseconds);
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function openBill() {
  const filename = screen.getByText("bill.pdf");
  const button = filename.closest("button");
  if (!button) throw new Error("Recent bill did not render as an actionable item");
  fireEvent.click(button);
  await advance(250);
}

describe("evidence page navigation", () => {
  let activeRequests = 0;
  let activeJobOverride: Record<string, unknown> | null = null;
  let historyJobOverride: Record<string, unknown> | null = null;
  let historyHospitalName: string | null = "Test Hospital";
  let abortRequests = 0;
  let bulkPayload: Record<string, unknown> | null = null;
  let rowPatchPayload: Record<string, unknown> | null = null;
  let reviewHospitalId: string | null = null;
  let aliasApplyPayload: Record<string, unknown> | null = null;
  let hospitalLinkPayload: Record<string, unknown> | null = null;
  let reviewPayload: Record<string, unknown> = structuredClone(review);
  let validationPayload: Record<string, unknown> = {
    validation_version: "extraction_validation_v5",
    status: "failed",
    issues: [],
  };
  let normalizedRowsPayload = structuredClone(rowsResult);
  let sourcePayload: Omit<typeof sourceTables, "unavailable_reason"> & {
    unavailable_reason: "legacy_result" | "no_source_tables" | null;
  } = sourceTables;

  beforeEach(() => {
    vi.useFakeTimers();
    activeRequests = 0;
    activeJobOverride = null;
    historyJobOverride = null;
    historyHospitalName = "Test Hospital";
    abortRequests = 0;
    bulkPayload = null;
    rowPatchPayload = null;
    reviewHospitalId = null;
    aliasApplyPayload = null;
    hospitalLinkPayload = null;
    reviewPayload = structuredClone(review);
    validationPayload = {
      validation_version: "extraction_validation_v5",
      status: "failed",
      issues: [],
    };
    normalizedRowsPayload = structuredClone(rowsResult);
    sourcePayload = structuredClone(sourceTables);
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.includes("/health/ready")) {
          return json({
            profile_revision: 4,
            alias_registry_revision: 0,
            active_jobs: 0,
            worker_capacity: 2,
            queue_capacity: 20,
            retention_hours: 720,
            storage_total_bytes: 100,
            storage_free_bytes: 80,
            storage_min_free_bytes: 0,
          });
        }
        if (url.includes("scope=active")) {
          activeRequests += 1;
          const documents = activeJobOverride
            ? [activeJobOverride]
            : activeRequests === 1
              ? []
              : [{ ...job }];
          return json({
            total: documents.length,
            offset: 0,
            limit: 200,
            has_more: false,
            documents,
          });
        }
        if (url.includes("scope=history")) {
          return json({
            total: 1,
            offset: 0,
            limit: 50,
            has_more: false,
            documents: [
              historyJobOverride
              ?? { ...job, hospital_name: historyHospitalName },
            ],
          });
        }
        if (url.endsWith("/abort")) {
          abortRequests += 1;
          expect(init?.method).toBe("POST");
          return json({ id: job.id, status: "cancelling" });
        }
        if (url.endsWith("/rows/bulk")) {
          bulkPayload = JSON.parse(String(init?.body));
          return json({ review_revision: 1, updated_count: 1 });
        }
        if (url.endsWith("/rows/row-1")) {
          rowPatchPayload = JSON.parse(String(init?.body));
          return json({ review_revision: 1, row: normalizedRowsPayload.rows[0] });
        }
        if (url.endsWith("/column-aliases/preview")) {
          return json({
            hospital_id: "hospital-1",
            source_label: "Co-pay %",
            canonical_field: "discount",
            review_revision: 0,
            registry_revision: 3,
            source_digest: "d".repeat(64),
            counts: { fillable: 1, unchanged: 0, conflicting: 0, invalid: 0, unlinked: 0 },
            candidates: [{
              candidate_id: "candidate-1",
              source_table_id: "p1-t1-s1",
              source_row_id: "p1-t1-s1-r1",
              source_column_id: "c2",
              row_id: "row-1",
              source_value: "10",
              classification: "fillable",
              current_value: null,
              proposed_value: "10",
            }],
          });
        }
        if (url.endsWith("/hospital-link")) {
          hospitalLinkPayload = JSON.parse(String(init?.body));
          return json({
            review_revision: 1,
            registry_revision: 1,
            profile_revision: 4,
            hospital_id: "created-hospital",
            hospital_name: "Test Hospital",
          });
        }
        if (url.endsWith("/column-aliases/apply")) {
          aliasApplyPayload = JSON.parse(String(init?.body));
          return json({ review_revision: 1, registry_revision: 4, updated_count: 1 });
        }
        if (url.endsWith("/hospitals/hospital-1/aliases")) {
          return json({ registry_revision: 3, aliases: [] });
        }
        if (url.endsWith("/api/v2/hospitals/trained")) {
          return json({
            registry_revision: 0,
            profile_revision: 4,
            total: 2,
            hospitals: [
              { hospital_id: "kamakshi", hospital_name: "Dr. Kamakshi Memorial Hospital", active_profile_count: 2 },
              { hospital_id: "vijaya", hospital_name: "Vijaya Group of Hospitals", active_profile_count: 1 },
            ],
          });
        }
        if (url.includes("/source-tables?")) {
          return json(structuredClone(sourcePayload));
        }
        if (url.includes("/rows?")) return json(structuredClone(normalizedRowsPayload));
        if (url.endsWith("/review")) {
          return json({ ...structuredClone(reviewPayload), hospital_id: reviewHospitalId });
        }
        if (url.endsWith("/validation")) {
          return json(structuredClone(validationPayload));
        }
        throw new Error(`Unexpected request: ${url}`);
      }),
    );
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  test("loads the largest history page so completed bills are immediately visible", async () => {
    render(<Home />);

    await advance(250);

    expect(
      vi.mocked(fetch).mock.calls.some(([input]) => {
        const url = new URL(String(input), "http://localhost");
        return (
          url.pathname === "/api/v2/documents"
          && url.searchParams.get("scope") === "history"
          && url.searchParams.get("limit") === "200"
        );
      }),
    ).toBe(true);
  });

  test("background refresh does not reset a manually selected empty page", async () => {
    render(<Home />);

    await advance(250);
    await advance(250);
    await openBill();
    expect(screen.getByText("Source page 1 / 2")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(screen.getByText("Source page 2 / 2")).toBeInTheDocument();

    await advance(3000);
    await advance(250);

    expect(screen.getByText("Source page 2 / 2")).toBeInTheDocument();
  });

  test("printed source columns are the default and normalized columns remain available", async () => {
    render(<Home />);

    await advance(250);
    await advance(250);
    await openBill();

    expect(screen.getByRole("button", { name: "Printed columns" })).toHaveClass("active");
    expect(screen.getByRole("columnheader", { name: "Co-pay %" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "10" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Normalized" }));

    expect(screen.getByRole("columnheader", { name: "Rate" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "Not printed" })).toBeInTheDocument();
  });

  test("printed rows use server ordinals across multiple tables", async () => {
    const second = structuredClone(sourceTables.tables[0]);
    second.id = "p1-t2-s1";
    second.table_id = "p1-t2";
    second.rows[0].id = "p1-t2-s1-r1";
    second.rows[0].order = 0;
    second.rows[0].ordinal = 151;
    sourcePayload = {
      ...structuredClone(sourceTables),
      total: 2,
      tables: [
        {
          ...structuredClone(sourceTables.tables[0]),
          rows: [
            {
              ...structuredClone(sourceTables.tables[0].rows[0]),
              order: 149,
              ordinal: 150,
            },
          ],
        },
        second,
      ],
    };

    render(<Home />);
    await advance(250);
    await advance(250);
    await openBill();

    expect(screen.getByRole("cell", { name: "150" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "151" })).toBeInTheDocument();
  });

  test("structured validation issues navigate to their exact printed row", async () => {
    reviewPayload = {
      ...structuredClone(review),
      issues_open: 1,
      issues: [{
        id: "issue-1",
        code: "unlinked_financial_source_row",
        severity: "blocking",
        message: "Unmapped financial value needs review",
        page_number: 1,
        table_id: "p1-t1",
        table_type: "semantic_validation",
        source_row_id: "p1-t1-s1-r1",
        canonical_row_id: null,
        field: "net_amount",
        related_source_row_ids: [],
        related_canonical_row_ids: [],
        reason_codes: ["unlinked_financial_source_row"],
        status: "open" as const,
        resolution_reason: null,
      }],
    };

    render(<Home />);
    await advance(250);
    await advance(250);
    await openBill();

    expect(screen.getByText("Unmapped financial value needs review")).toBeInTheDocument();
    expect(screen.getByText(/printed p1-t1-s1-r1/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "open" }));
    await advance(250);

    expect(
      vi.mocked(fetch).mock.calls.some(([input]) => {
        const url = new URL(String(input), "http://localhost");
        return url.pathname.endsWith("/source-tables")
          && url.searchParams.get("anchor_row_id") === "p1-t1-s1-r1";
      }),
    ).toBe(true);
    expect(screen.getByRole("button", { name: "Printed columns" })).toHaveClass("active");
  });

  test("selects visible normalized rows and submits one bulk rejection", async () => {
    render(<Home />);
    await advance(250);
    await advance(250);
    await openBill();

    fireEvent.click(screen.getByRole("button", { name: "Normalized" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select Consultation" }));
    expect(screen.getByText("1 selected")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reject selected" }));
    fireEvent.change(screen.getByPlaceholderText("Why are these rows being changed?"), {
      target: { value: "Duplicate summary row" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Reject selected rows" }));
    await advance(0);

    expect(bulkPayload).toEqual({
      row_ids: ["row-1"],
      action: "reject",
      reason: "Duplicate summary row",
    });
  });

  test("edits a rejected row without resubmitting its unchanged disposition", async () => {
    normalizedRowsPayload.rows[0] = {
      ...normalizedRowsPayload.rows[0],
      review_disposition: "rejected",
      bulk_action: null,
    };
    render(<Home />);
    await advance(250);
    await advance(250);
    await openBill();

    fireEvent.click(screen.getByRole("button", { name: "Normalized" }));
    fireEvent.click(screen.getAllByText("Consultation")[0]);
    fireEvent.click(screen.getByRole("button", { name: "Review row ↗" }));
    expect(screen.getByLabelText("Disposition")).toHaveValue("rejected");
    expect(screen.queryByRole("button", { name: "Reject row" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Description"), {
      target: { value: "Corrected rejected consultation" },
    });
    fireEvent.change(screen.getByPlaceholderText("What did you verify or change?"), {
      target: { value: "Corrected description while retaining rejection" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save correction" }));
    await advance(0);

    expect(rowPatchPayload).toEqual({
      changes: { description: "Corrected rejected consultation" },
      reason: "Corrected description while retaining rejection",
    });
  });

  test("opens hospital alias setup from an unmapped printed header", async () => {
    render(<Home />);
    await advance(250);
    await advance(250);
    await openBill();

    fireEvent.click(screen.getByRole("button", { name: "Co-pay %" }));

    expect(screen.getByRole("heading", { name: "Column aliases" })).toBeInTheDocument();
    expect(screen.getByText(/Link this grounded bill identity/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Link hospital" })).toBeInTheDocument();
  });

  test("links a hospital with both observed registry revisions", async () => {
    render(<Home />);
    await advance(250);
    await advance(250);
    await openBill();

    fireEvent.click(screen.getByRole("button", { name: "Co-pay %" }));
    fireEvent.change(screen.getByPlaceholderText("How was this hospital identity verified?"), {
      target: { value: "Verified grounded hospital identity" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Link hospital" }));
    await advance(0);

    expect(hospitalLinkPayload).toEqual({
      create: true,
      hospital_id: null,
      registry_revision: 0,
      profile_revision: 4,
      reason: "Verified grounded hospital identity",
    });
  });

  test("selects grounded alias candidates and submits the source digest", async () => {
    reviewHospitalId = "hospital-1";
    render(<Home />);
    await advance(250);
    await advance(250);
    await openBill();

    fireEvent.click(screen.getByRole("button", { name: "Co-pay %" }));
    fireEvent.change(screen.getByLabelText("Normalized field"), {
      target: { value: "discount" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Preview affected rows" }));
    await advance(0);
    fireEvent.change(screen.getByPlaceholderText("Why does this header map to this field?"), {
      target: { value: "Verified co-pay discount column" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Apply 1 selected and teach hospital" }),
    );
    await advance(0);

    expect(aliasApplyPayload).toEqual({
      hospital_id: "hospital-1",
      source_label: "Co-pay %",
      canonical_field: "discount",
      source_digest: "d".repeat(64),
      registry_revision: 3,
      selected_candidate_ids: ["candidate-1"],
      reason: "Verified co-pay discount column",
    });
  });

  test("legacy printed-column explanation remains visible in normalized mode", async () => {
    sourcePayload = {
      ...structuredClone(sourceTables),
      available: false,
      unavailable_reason: "legacy_result",
      total: 0,
      tables: [],
    };

    render(<Home />);
    await advance(250);
    await advance(250);
    await openBill();

    expect(
      screen.getByText(
        "Printed columns are unavailable for this older extraction. Reprocess this bill to enable them.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Printed columns" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Normalized" })).toHaveClass("active");
  });

  test("new empty extraction explains that no printed table was detected", async () => {
    sourcePayload = {
      ...structuredClone(sourceTables),
      available: false,
      unavailable_reason: "no_source_tables",
      total: 0,
      tables: [],
    };

    render(<Home />);
    await advance(250);
    await advance(250);
    await openBill();

    expect(
      screen.getByText(
        "No printed table structure was detected; normalized rows are shown.",
      ),
    ).toBeInTheDocument();
  });

  test("completed bills never use the PDF filename as a hospital-name fallback", async () => {
    historyHospitalName = null;
    render(<Home />);

    await advance(250);
    await advance(250);

    expect(screen.getAllByText("Hospital not identified").length).toBeGreaterThan(0);
    expect(screen.getAllByText("bill.pdf").length).toBeGreaterThan(0);
  });

  test("legacy approvals are shown as ineffective and exports stay hidden", async () => {
    historyJobOverride = {
      ...job,
      certification_status: "legacy_uncertified",
    };
    reviewPayload = {
      ...structuredClone(review),
      approval: {
        status: "approved",
        approved_at: "2026-08-01T00:00:00Z",
        review_revision: 1,
      },
      approval_effective: false,
      approval_blockers: ["legacy_uncertified"],
      export_eligible: false,
    };

    render(<Home />);
    await advance(250);
    await openBill();

    expect(screen.getByText(/Legacy uncertified extraction/)).toBeInTheDocument();
    expect(screen.getByText(/must be reprocessed and certified/)).toBeInTheDocument();
    expect(screen.queryByText("Review sealed.")).not.toBeInTheDocument();
    expect(screen.queryByText("CSV ↗")).not.toBeInTheDocument();
  });

  test("tampered certification is visibly blocked", async () => {
    historyJobOverride = {
      ...job,
      certification_status: "invalid",
    };
    reviewPayload = {
      ...structuredClone(review),
      approval_effective: false,
      approval_blockers: ["certification_invalid"],
      export_eligible: false,
    };

    render(<Home />);
    await advance(250);
    await openBill();

    expect(screen.getByText(/Certification invalid/)).toBeInTheDocument();
    expect(screen.queryByText("CSV ↗")).not.toBeInTheDocument();
  });

  test("failed extraction displays its retained structured validation report", async () => {
    historyJobOverride = {
      ...job,
      status: "failed",
      certification_status: null,
      error: "extraction_integrity_failed",
    };
    validationPayload = {
      validation_version: "extraction_validation_v5",
      status: "failed",
      issues: [{
        id: "fatal-envelope",
        code: "extraction_result_contract_invalid",
        severity: "fatal",
        message: "The extraction envelope is invalid",
        page_number: null,
        table_id: null,
        source_row_id: null,
        canonical_row_id: null,
        field: "rows",
        reason_codes: [],
        status: "open",
        resolution_reason: null,
      }],
    };

    render(<Home />);
    await advance(250);
    await openBill();
    await advance(250);

    expect(screen.getByText("The extraction envelope is invalid")).toBeInTheDocument();
    expect(screen.getByText("field rows")).toBeInTheDocument();
  });

  test("opens on the dashboard even when completed history exists", async () => {
    render(<Home />);
    await advance(250);

    expect(screen.getByText("Evidence dashboard")).toBeInTheDocument();
    expect(screen.getByText("Drop hospital bills here")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Dashboard" })).toHaveAttribute("aria-current", "page");
  });

  test("returns to the dashboard from an evidence review", async () => {
    render(<Home />);
    await advance(250);
    await openBill();
    expect(screen.getByText("Source page 1 / 2")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Dashboard" }));

    expect(screen.getByText("Drop hospital bills here")).toBeInTheDocument();
  });

  test("opens and dismisses the mobile navigation drawer", async () => {
    render(<Home />);
    await advance(250);

    fireEvent.click(screen.getByRole("button", { name: "Open navigation" }));
    expect(screen.getByLabelText("Primary navigation")).toHaveClass("open");

    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.getByLabelText("Primary navigation")).not.toHaveClass("open");
  });

  test("shows active profile hospitals in the training directory", async () => {
    render(<Home />);
    await advance(250);

    fireEvent.click(screen.getByRole("button", { name: /Hospitals/ }));
    await advance(0);

    expect(screen.getAllByText("Dr. Kamakshi Memorial Hospital").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Vijaya Group of Hospitals").length).toBeGreaterThan(0);
    expect(screen.getByText("2 active layout profiles")).toBeInTheDocument();
  });

  test("aborts an active bill and removes it from the queue", async () => {
    activeJobOverride = {
      ...job,
      status: "queued",
      hospital_name: null,
      hospital_confidence: null,
      hospital_name_source: null,
    };
    vi.stubGlobal("confirm", vi.fn(() => true));
    render(<Home />);
    await advance(250);

    fireEvent.click(screen.getByRole("button", { name: /Active/ }));
    await advance(0);
    fireEvent.click(screen.getByRole("button", { name: "Abort" }));
    await advance(0);

    expect(abortRequests).toBe(1);
    expect(screen.queryByRole("button", { name: "Abort" })).not.toBeInTheDocument();
  });
});
