import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { releaseRevision } from "./release";

describe("releaseRevision", () => {
  it("uses the immutable manifest instead of a runtime override", () => {
    const path = join(mkdtempSync(join(tmpdir(), "gmoney-release-")), "release.json");
    const revision = "a".repeat(40);
    writeFileSync(
      path,
      JSON.stringify({
        manifest_version: "gmoney_release_v1",
        revision,
        revision_required: true,
      }),
    );
    process.env.GMONEY_BUILD_REVISION = "b".repeat(40);

    expect(releaseRevision(path)).toBe(revision);
  });

  it("rejects a malformed production manifest", () => {
    const path = join(mkdtempSync(join(tmpdir(), "gmoney-release-")), "release.json");
    writeFileSync(
      path,
      JSON.stringify({
        manifest_version: "gmoney_release_v1",
        revision: "unknown",
        revision_required: true,
      }),
    );

    expect(() => releaseRevision(path)).toThrow("invalid baked GMoney release manifest");
  });
});
