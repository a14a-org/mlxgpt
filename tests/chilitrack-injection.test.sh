#!/bin/sh
set -eu

repo_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"

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

# The shipped tag must carry the URL-redaction attributes, and the startup
# rewrite must not drop them. ChiliTrack redacts only when it is told to:
# without these, location.href ships verbatim -- query string and fragment
# included -- in the RUM payload and in error breadcrumbs.
#
# tests/e2e/url-redaction.spec.ts pins the same two attributes in Playwright,
# but that spec runs in the `build` job behind `ruff check`, so a lint failure
# on unrelated Python skips it. Asserting here as well keeps the pin executing
# in the container job, which exercises the real site/index.html through the
# production entrypoint.
cp "$repo_root/site/index.html" "$tmp_dir/index.html"
CHILITRACK_HTML_ROOT="$tmp_dir" \
NEXT_PUBLIC_CHILITRACK_WEBSITE_ID="$override_id" \
  sh docker/40-chilitrack-website-id.sh

for attribute in 'data-exclude-search="true"' 'data-exclude-hash="true"'; do
  if ! grep 'id="chilitrack-analytics"' "$tmp_dir/index.html" | grep -q -- "$attribute"; then
    echo "ChiliTrack tag in site/index.html is missing $attribute" >&2
    exit 1
  fi
done
