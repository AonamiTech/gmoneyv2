import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export function GET() {
  return NextResponse.json(
    { release_revision: process.env.GMONEY_BUILD_REVISION ?? "unknown" },
    { headers: { "Cache-Control": "no-store" } },
  );
}
