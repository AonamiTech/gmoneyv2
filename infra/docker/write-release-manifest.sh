#!/bin/sh
set -eu

revision="${GMONEY_BUILD_REVISION:-unknown}"
required="${GMONEY_REQUIRE_BUILD_REVISION:-0}"

case "$required" in
  0) required_json=false ;;
  1) required_json=true ;;
  *) echo "GMONEY_REQUIRE_BUILD_REVISION must be 0 or 1" >&2; exit 1 ;;
esac

if [ "$revision" != "unknown" ] && ! printf '%s' "$revision" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "GMONEY_BUILD_REVISION must be a full lowercase 40-character Git SHA" >&2
  exit 1
fi
if [ "$required" = "1" ] && [ "$revision" = "unknown" ]; then
  echo "GMONEY_BUILD_REVISION is required for this image" >&2
  exit 1
fi

install -d -m 0755 /etc/gmoney
printf '{"manifest_version":"gmoney_release_v1","revision":"%s","revision_required":%s}\n' \
  "$revision" "$required_json" > /etc/gmoney/release.json
chmod 0444 /etc/gmoney/release.json
