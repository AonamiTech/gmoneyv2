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
  approval: null,
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
  let historyHospitalName: string | null = "Test Hospital";
  let abortRequests = 0;
  let bulkPayload: Record<string, unknown> | null = null;
  let sourcePayload: Omit<typeof sourceTables, "unavailable_reason"> & {
    unavailable_reason: "legacy_result" | "no_source_tables" | null;
  } = sourceTables;

  beforeEach(() => {
    vi.useFakeTimers();
    activeRequests = 0;
    activeJobOverride = null;
    historyHospitalName = "Test Hospital";
    abortRequests = 0;
    bulkPayload = null;
    sourcePayload = structuredClone(sourceTables);
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.includes("/health/ready")) {
          return json({
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
            documents: [{ ...job, hospital_name: historyHospitalName }],
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
        if (url.endsWith("/api/v2/hospitals/trained")) {
          return json({
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
        if (url.includes("/rows?")) return json(structuredClone(rowsResult));
        if (url.endsWith("/review")) return json(structuredClone(review));
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
