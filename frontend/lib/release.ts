import { readFileSync } from "node:fs";

export const RELEASE_MANIFEST_PATH = "/etc/gmoney/release.json";

type ReleaseManifest = {
  manifest_version: "gmoney_release_v1";
  revision: string;
  revision_required: boolean;
};

export function releaseRevision(path = RELEASE_MANIFEST_PATH): string {
  try {
    const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
    if (!parsed || typeof parsed !== "object") {
      throw new Error("release manifest is not an object");
    }
    const manifest = parsed as Partial<ReleaseManifest>;
    const validRevision = /^[0-9a-f]{40}$/.test(manifest.revision ?? "");
    const validUnknown =
      manifest.revision === "unknown" && manifest.revision_required === false;
    if (
      manifest.manifest_version !== "gmoney_release_v1" ||
      typeof manifest.revision_required !== "boolean" ||
      (!validRevision && !validUnknown)
    ) {
      throw new Error("invalid baked GMoney release manifest");
    }
    return manifest.revision as string;
  } catch (error) {
    const missing = error instanceof Error && "code" in error && error.code === "ENOENT";
    if (missing && process.env.NODE_ENV !== "production") {
      return process.env.GMONEY_BUILD_REVISION ?? "unknown";
    }
    throw error;
  }
}
