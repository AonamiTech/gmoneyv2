import { NextResponse } from "next/server";

import { releaseRevision } from "@/lib/release";

export const dynamic = "force-dynamic";

export function GET() {
  return NextResponse.json(
    { release_revision: releaseRevision() },
    { headers: { "Cache-Control": "no-store" } },
  );
}
