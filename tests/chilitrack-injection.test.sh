#!/bin/sh
set -eu

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

default_id="c432d2ba-2529-4a99-91e5-d07bd4bcbbbe"
override_id="11111111-2222-4333-8444-555555555555"
second_override_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
umami_id="8f50186e-74cb-4d63-978d-bd4987e04b2b"
{
  printf '<script src="https://analytics.a14a.org/script.js" data-website-id="%s"></script>\n' "$umami_id"
  printf '<script id="chilitrack-analytics" data-website-id="%s" data-domains="mlxgpt.com"></script>\n' "$default_id"
} > "$tmp_dir/index.html"

CHILITRACK_HTML_ROOT="$tmp_dir" \
NEXT_PUBLIC_CHILITRACK_WEBSITE_ID="$override_id" \
  sh docker/40-chilitrack-website-id.sh

grep -q "$override_id" "$tmp_dir/index.html"
if grep -q "$default_id" "$tmp_dir/index.html"; then
  echo "default ChiliTrack website ID was not replaced" >&2
  exit 1
fi
grep -q "$umami_id" "$tmp_dir/index.html"
grep -q 'data-domains="mlxgpt.com"' "$tmp_dir/index.html"

# A second startup must replace the current override, not depend on the
# original default ID still being present in the writable container layer.
CHILITRACK_HTML_ROOT="$tmp_dir" \
NEXT_PUBLIC_CHILITRACK_WEBSITE_ID="$second_override_id" \
  sh docker/40-chilitrack-website-id.sh

grep -q "$second_override_id" "$tmp_dir/index.html"
if grep -q "$override_id" "$tmp_dir/index.html"; then
  echo "ChiliTrack website ID was not replaced on a repeated startup" >&2
  exit 1
fi
grep -q "$umami_id" "$tmp_dir/index.html"

if CHILITRACK_HTML_ROOT="$tmp_dir" \
  NEXT_PUBLIC_CHILITRACK_WEBSITE_ID='not-a-uuid' \
  sh docker/40-chilitrack-website-id.sh >/dev/null 2>&1; then
  echo "invalid ChiliTrack website ID was accepted" >&2
  exit 1
fi

printf '<script id="chilitrack-analytics"></script>\n' > "$tmp_dir/index.html"
if CHILITRACK_HTML_ROOT="$tmp_dir" \
  NEXT_PUBLIC_CHILITRACK_WEBSITE_ID="$override_id" \
  sh docker/40-chilitrack-website-id.sh >/dev/null 2>&1; then
  echo "ChiliTrack tag without data-website-id was accepted" >&2
  exit 1
fi

printf '<script id="chilitrack-analytics" data-website-id="%s"></script><script id="chilitrack-analytics" data-website-id="%s"></script>\n' \
  "$default_id" "$default_id" > "$tmp_dir/index.html"
if CHILITRACK_HTML_ROOT="$tmp_dir" \
  NEXT_PUBLIC_CHILITRACK_WEBSITE_ID="$override_id" \
  sh docker/40-chilitrack-website-id.sh >/dev/null 2>&1; then
  echo "duplicate ChiliTrack tags were accepted" >&2
  exit 1
fi
