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

describe("evidence page navigation", () => {
  let activeRequests = 0;
  let sourcePayload: Omit<typeof sourceTables, "unavailable_reason"> & {
    unavailable_reason: "legacy_result" | "no_source_tables" | null;
  } = sourceTables;

  beforeEach(() => {
    vi.useFakeTimers();
    activeRequests = 0;
    sourcePayload = structuredClone(sourceTables);
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
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
          return json({
            total: activeRequests === 1 ? 0 : 1,
            offset: 0,
            limit: 200,
            has_more: false,
            documents: activeRequests === 1 ? [] : [{ ...job }],
          });
        }
        if (url.includes("scope=history")) {
          return json({
            total: 1,
            offset: 0,
            limit: 50,
            has_more: false,
            documents: [{ ...job }],
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

  test("background refresh does not reset a manually selected empty page", async () => {
    render(<Home />);

    await advance(250);
    await advance(250);
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

    expect(screen.getByRole("button", { name: "Printed columns" })).toHaveClass("active");
    expect(screen.getByRole("columnheader", { name: "Co-pay %" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "10" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Normalized" }));

    expect(screen.getByRole("columnheader", { name: "Rate" })).toBeInTheDocument();
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

    expect(screen.getByRole("cell", { name: "150" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "151" })).toBeInTheDocument();
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

    expect(
      screen.getByText(
        "No printed table structure was detected; normalized rows are shown.",
      ),
    ).toBeInTheDocument();
  });
});
