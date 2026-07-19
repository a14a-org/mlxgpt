#!/bin/sh
set -eu

website_id="${NEXT_PUBLIC_CHILITRACK_WEBSITE_ID:-c432d2ba-2529-4a99-91e5-d07bd4bcbbbe}"
html_root="${CHILITRACK_HTML_ROOT:-/usr/share/nginx/html}"

if ! printf '%s' "$website_id" | grep -Eq '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'; then
  echo "Invalid NEXT_PUBLIC_CHILITRACK_WEBSITE_ID; expected a UUID" >&2
  exit 1
fi

find "$html_root" -type f -name '*.html' | while IFS= read -r html_file; do
  tag_count="$(grep -o 'id="chilitrack-analytics"' "$html_file" | wc -l | tr -d ' ')"
  if [ "$tag_count" -eq 0 ]; then
    continue
  fi
  if [ "$tag_count" -ne 1 ]; then
    echo "Expected exactly one ChiliTrack analytics tag in $html_file" >&2
    exit 1
  fi

  temporary_file="${html_file}.chilitrack.tmp"
  sed "/id=\"chilitrack-analytics\"/ s/data-website-id=\"[^\"]*\"/data-website-id=\"$website_id\"/" "$html_file" > "$temporary_file"
  if ! grep 'id="chilitrack-analytics"' "$temporary_file" | grep -q "data-website-id=\"$website_id\""; then
    rm -f "$temporary_file"
    echo "ChiliTrack analytics tag in $html_file is missing a replaceable data-website-id" >&2
    exit 1
  fi
  mv "$temporary_file" "$html_file"
done
