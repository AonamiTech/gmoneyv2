import { readFileSync } from "node:fs";

const path = "/etc/gmoney/release.json";
const payload = JSON.parse(readFileSync(path, "utf8"));
const validRevision = /^[0-9a-f]{40}$/.test(payload.revision);
const validUnknown = payload.revision === "unknown" && payload.revision_required === false;

if (
  payload.manifest_version !== "gmoney_release_v1" ||
  typeof payload.revision_required !== "boolean" ||
  (!validRevision && !validUnknown)
) {
  throw new Error("invalid baked GMoney release manifest");
}
